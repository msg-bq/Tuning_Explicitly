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
            for r in data.rationales:
                knowledge_texts += r.get_knowledge_texts()  # 这儿也没有根据prediction和label的一致性选择正确的rule
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
            knowledge_texts = rationale.get_knowledge_texts(rationale.rationale)

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


    def eval(self):
        """
        将模型置于evaluate模式，包括：# 其实某种意义上，可以说读入rule_base应该专门有一个load函数，而不是放在这里。train.rationale可能也应该在load里读入
        1. 将llm置于evaluate模式
        2. 读入rule_base
        3. 读入train_dataset的rationale，并生成demos。由于此刻train里面还没保存，所以就先读入demos_epoch{epoch-1}作为替代
        """
        # 1. 将llm置于evaluate模式
        # self.llm.eval() 这行代码暂时还无法生效

        # 2. 读入rule_base
        if len(self.rule_base) == 0:  # 这个我觉得是，只有纯测试的时候才需要读入。平时的话规则库本来就在训练过程中有了
            # 如果还想考虑一个特殊情况的话，就是checkpoint。但是这个我觉得也没必要，因为checkpoint的load阶段就应该读入了
            rule_base_final_path = f"{self.args.save_dir}/rule_base_final"
            self.rule_base.read_rules(rule_base_final_path)

        # 3. 读入train_dataset的rationale
        # 目前还没保存训练集，所以直接读取demos和added_rules
        with open(f"{self.args.save_dir}/demos_epoch{0}", encoding="utf8") as f:
            self.eval_demos = "".join([l for l in f.readlines()])

        save_path = f"{self.args.save_dir}/demos_eval"
        with open(save_path, 'w', encoding="utf8") as f:
            f.write(str(self.eval_demos))

        with open(f"{self.args.save_dir}/added_rule_epoch{0}", encoding="utf8") as f:
            self.eval_added_rules = "".join([l for l in f.readlines()])

    def eval_step(self, example: Example):
        # 预留一个后处理demos的函数，是hjq写的

        _, response = llm_n_shot_CoT(self.llm, self.eval_added_rules,
                                     input_text=example.question, demos=self.eval_demos, model=self.args.llm_model)
        rationale = example.parse_response(response, self.args)
        prediction = Rationale.clean_prediction(rationale['prediction'])
        print(rationale['prediction'])
        print(prediction, example.gold_label)
        return prediction, example.gold_label

    def evaluate(self, is_valid=False, special_datasets: DatasetLoader = None):
        """
        验证集和测试集的评估
        """
        eval_type = "valid" if is_valid else "test"
        datasets = special_datasets if special_datasets else self.valid_dataset if is_valid else self.test_dataset

        # if is_valid: # valid用于训练阶段的测试
        #     demos = demo_cluster(self.args, self.train_dataset) #目前这个手段，还不能做到逐阶段优化5-shot
        #     # 因为demo_cluster里用的rationale，只是random.choice了一个rationale，而不是根据score来选
        #     demos, added_rules = n_shot_prompt(self.args, rules=self.rule_base.sample_rules(), demos=demos)
        # else:
        #     pass

        correct_cnt = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.eval_step, example) for example in datasets]
            for future in futures:
                prediction, gold_label = future.result()
                if prediction == gold_label:
                    correct_cnt += 1

        logger.info(f"{eval_type}集上的准确率为：{correct_cnt / len(datasets)}")