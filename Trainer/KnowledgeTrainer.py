import threading
from concurrent.futures import ThreadPoolExecutor

from Trainer.inference import cold_start_inference, llm_inference_category
from utils.data import KnowledgeBase, DatasetLoader, Example, Rationale
from utils.llm import LLM
import Levenshtein

from logger import logger


class Trainer:
    def __init__(self, args, train_dataset: DatasetLoader, valid_dataset: DatasetLoader, test_dataset: DatasetLoader,
                 llm: LLM, knowledge_base: KnowledgeBase = KnowledgeBase()):
        self.args = args
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.llm = llm
        self.knowledge_base = knowledge_base

        self.lock = threading.Lock()

    def cold_start(self):
        """
        """
        dataset = cold_start_inference(self.args, self.llm, self.train_dataset)

        for data in dataset:
            knowledge_texts = []
            for r in data.rationale:
                knowledge_texts += r.extract_knowledge()  # 这儿也没有根据prediction和label的一致性选择正确的rule
                self.knowledge_base.update_knowledge(knowledge_texts=knowledge_texts,
                                                     question=data.question,
                                                     rationale=r,
                                                     score=1)  # cold start阶段只要能搭建初始库即可，score基本无所谓

        self.knowledge_base.build_conceptual_memory()
        self.knowledge_base.save_knowledge_memory(f"./data/{self.args.dataset}/rule_base_cold")
        logger.info("完成cold start")

    def forward(self, example):
        response = llm_inference_category(args=self.args,
                                          knowledge_base=self.knowledge_base,
                                          llm=self.llm,
                                          train_prompt=self.args.train_prompt,
                                          input_text=example.question)

        print("response:\n", response)
        new_rationale = example.parse_response(response, self.args)

        if new_rationale['rationale'] != "" and new_rationale['prediction'] != "":
            new_rationale['prediction'] = Rationale.clean_prediction(new_rationale['prediction'])
            rationale_instance = Rationale(rationale=new_rationale['rationale'],
                                           prediction=new_rationale['prediction'])
            score = self.score(new_rationale['prediction'], example.gold_label)

            if score > 0.5:
                print('++++' * 50)
                print("new_rationale:\n", new_rationale)
                example.update_rationale(rationale_instance)  # 做了inplace的更新

            return rationale_instance, score

        return None, None

    @staticmethod
    def score(pred: str, gold: str) -> float:
        """
        比对prediction和gold_label打分，用于调整Rule confidence
        """
        pred, gold = pred.strip().lower(), gold.strip().lower()
        edit_distance = Levenshtein.distance(pred, gold)
        if edit_distance == 0:
            score = 1
        else:
            score = 1 - 2 * edit_distance / max(len(pred), len(gold))
        return score

    def backward(self, example: Example,
                 rationale: Rationale,
                 score: float):
        if rationale:
            knowledge_texts = rationale.extract_knowledge_texts()

            if score > 0.5:
                print("occurred_rules:", knowledge_texts)
                self.knowledge_base.update_knowledge(knowledge_texts=knowledge_texts,
                                                     question=example.question,
                                                     rationale=rationale,
                                                     score=score)
                print('oooo' * 50)
            else:
                print("occurred_rules:", f"score较低（{score}），没有rationale;")

    def train_step(self, example: Example):
        rationale, score = self.forward(example)

        with self.lock:
            self.backward(example, rationale, score)

        return {'rationale': rationale, 'score': score}

    def train(self):
        save_path = f"{self.args.save_dir}/train_loss.txt"
        with open(save_path, 'w', encoding="utf8"):
            pass

        for ep in range(self.args.epoch):  # 这里最好是epoch
            with ThreadPoolExecutor(max_workers=100) as executor:
                futures = [executor.submit(self.train_step, example) for example in self.train_dataset]
                futures = [future for future in futures if future.result() is not None]

                losses = [future.result()['score'] for future in futures if future.result()['score'] is not None]
                rationales = [future.result()['rationale'] for future in futures
                              if future.result()['rationale'] is not None]

                losses = [(loss+1)/2 for loss in losses]
                # None对应样例、-1对应输出没有rationale的样例
                logger.info(f"epoch{ep}的平均score为：{sum(losses) / len(losses)}")  # 如果像正常的微调

                with open(save_path, 'a', encoding="utf8") as f:
                    f.write(f"epoch{ep}的平均score为：{sum(losses) / len(losses)}\n")
                # 其实训练集的信息是会被过拟合记住的，所以那个要求sample rule的时候不能用来源question的规则
                # 这条限制，是可以保留或者说可控的。这种过拟合也比参数微调方便控制
                # 如果现在加的话，就是从knowledge的source里面控制了

                with open(f"{self.args.save_dir}/rationales_epoch{ep}.txt", 'w', encoding="utf8") as f:
                    for rationale in rationales:
                        f.write(str(rationale) + '\n')

            self.knowledge_base.build_conceptual_memory()
            self.knowledge_base.broadcast_knowledge_info()  # 每个epoch统一平均，避免并行带来的不同步

            self.knowledge_base.save_knowledge_memory(knowledge_memory_path=f"{self.args.save_dir}/rule_map_{ep}",
                                                      vectorizer_path=f"{self.args.save_dir}/vectorizer_{ep}.pkl")

        self.knowledge_base.save_knowledge_memory(knowledge_memory_path=f"{self.args.save_dir}/rule_map_final",
                                                  vectorizer_path=f"{self.args.save_dir}/vectorizer_final.pkl")
