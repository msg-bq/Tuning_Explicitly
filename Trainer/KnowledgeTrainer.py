import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Union

from Trainer.inference import cold_start_inference, llm_inference_category
from utils.data import KnowledgeBase, DatasetLoader, Example, Rationale
from utils.llm import LLM
import Levenshtein

from logger import logger


class Trainer:
    def __init__(self, args, train_dataset: DatasetLoader, valid_dataset: DatasetLoader, test_dataset: DatasetLoader,
                 llm: LLM, knowledge_base: KnowledgeBase = KnowledgeBase(), metric: Callable = None):
        self.args = args
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.score = metric if metric is not None else self._default_metric
        assert isinstance(self.score, Callable), "metric should be callable"

        self.cur_ep = -1

        self.lock = threading.Lock()

        self.test_prompt_hook: Union[Callable, None] = None  # 如果多的话就像pytorch学习叭

        if self.valid_dataset is None and self.args.eval:
            self.valid_dataset = self.train_dataset

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
                                                     score=1)  # cold start要求完全一致才录入

        self.knowledge_base.build_conceptual_memory()
        self.knowledge_base.save_knowledge_memory(f"./data/{self.args.dataset}/rule_base_cold")
        logger.info("完成cold start")

    def forward(self, example):
        if self.test_prompt_hook is not None:
            input_text = self.test_prompt_hook(example.question)
        else:
            input_text = example.question

        response = llm_inference_category(args=self.args,
                                          knowledge_base=self.knowledge_base,
                                          llm=self.llm,
                                          train_prompt=self.args.train_prompt,
                                          input_text=input_text,
                                          temperature=0.3)

        print("response:\n", response)
        new_rationale = example.parse_response(response, self.args)

        if new_rationale['rationale'] != "" and new_rationale['prediction'] != "":
            new_rationale['prediction'] = Rationale.clean_prediction(new_rationale['prediction'])
            rationale_instance = Rationale(rationale=new_rationale['rationale'],
                                           prediction=new_rationale['prediction'])
            example.update_rationale(rationale_instance)

            score = self.score(new_rationale['prediction'], example.gold_label)
            if score < 0.5:
                print(1)

            return rationale_instance, score

        return None, -1

    @staticmethod
    def _default_metric(pred: str, gold: str) -> float:
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
        # todo: 暂时不考虑rationale为空的情况
        if rationale:
            knowledge_texts = rationale.get_knowledge_texts()
            self.knowledge_base.update_knowledge(knowledge_texts=knowledge_texts,
                                                 question=example.question,
                                                 rationale=rationale,
                                                 score=score)
        else:
            warnings.warn(
                f"example {example.question}没有rationale，忽略该样本")

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
            self.cur_ep = ep
            with ThreadPoolExecutor(max_workers=40) as executor:
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

            self.knowledge_base.save_knowledge_memory(knowledge_memory_path=f"{self.args.save_dir}/knowledge_base_{ep}",
                                                      vectorizer_path=f"{self.args.save_dir}/vectorizer_{ep}.pkl")

        self.knowledge_base.save_knowledge_memory(knowledge_memory_path=f"{self.args.save_dir}/knowledge_base_final",
                                                  vectorizer_path=f"{self.args.save_dir}/vectorizer_final.pkl")

    def eval(self):
        """
        将模型置于evaluate模式，包括：# 其实某种意义上，可以说读入rule_base应该专门有一个load函数，而不是放在这里。train.rationale可能也应该在load里读入
        1. 将llm置于evaluate模式
        2. 读入rule_base
        3. 读入train_dataset的rationale，并生成demos。由于此刻train里面还没保存，所以就先读入demos_epoch{epoch-1}作为替代
        """
        _version = self.cur_ep if self.cur_ep >= 0 else 'final'

        # 1. 将llm置于evaluate模式
        # self.llm.eval() 这行代码暂时还无法生效

        # 2. 读入rule_base
        if len(self.knowledge_base) == 0:  # 这个我觉得是，只有纯测试的时候才需要读入。平时的话规则库本来就在训练过程中有了
            # 如果还想考虑一个特殊情况的话，就是checkpoint。但是这个我觉得也没必要，因为checkpoint的load阶段就应该读入了
            knowledge_memory_path = f"{self.args.save_dir}/knowledge_base_{_version}"
            vectorizer_path = f"{self.args.save_dir}/vectorizer_{_version}.pkl"
            self.knowledge_base.load_knowledge_memory(knowledge_memory_path=knowledge_memory_path,
                                                      vectorizer_path=vectorizer_path)

            kb = KnowledgeBase()
            kb.set_knowledge_memory(self.knowledge_base.get_inference_knowledge_memory())
            self.knowledge_base = kb

        # 3. 读入train_dataset的rationale
        pass

    def eval_step(self, example: Example):
        response = llm_inference_category(args=self.args,
                                          knowledge_base=self.knowledge_base,
                                          llm=self.llm,
                                          train_prompt=self.args.train_prompt,
                                          input_text=example.question,
                                          mode='eval')

        rationale = example.parse_response(response, self.args)
        prediction = Rationale.clean_prediction(rationale['prediction'])

        return prediction, example.gold_label

    def evaluate(self, is_valid=False, special_datasets: DatasetLoader = None):
        """
        验证集和测试集的评估
        """
        eval_type = "valid" if is_valid else "test"
        datasets = special_datasets if special_datasets else self.valid_dataset if is_valid else self.test_dataset

        correct_cnt = 0
        with ThreadPoolExecutor(max_workers=200) as executor:
            futures = [executor.submit(self.eval_step, example) for example in datasets]
            for future in futures:
                print("做了n个")
                prediction, gold_label = future.result()
                print("prediction， gold", prediction, gold_label)
                prediction = prediction.replace("-in-law", "").replace("step-", "").replace("step", "")
                gold_label = gold_label.replace("-in-law", "").replace("step-", "").replace("step", "")
                if prediction.lower() == gold_label.lower():
                    correct_cnt += 1
                    print("做对一个")

        logger.info(f"{eval_type}集上的准确率为：{correct_cnt / len(datasets)}")

    def test(self,
             save_path: str,
             use_epoch_file: Union[int, str] = None,
             knowledge_memory_path: str = None,
             vectorizer_path: str = None,
             special_datasets: DatasetLoader = None
             ):
        """提示词也可以酌情调整，做个hook"""

        assert use_epoch_file or (knowledge_memory_path and vectorizer_path)

        if use_epoch_file:
            knowledge_memory_path = f"{save_path}/knowledge_base_{use_epoch_file}"
            vectorizer_path = f"{save_path}/vectorizer_{use_epoch_file}.pkl"

        if knowledge_memory_path and vectorizer_path:
            self.knowledge_base.load_knowledge_memory(knowledge_memory_path=knowledge_memory_path,
                                                      vectorizer_path=vectorizer_path)

        # km = self.knowledge_base.get_inference_knowledge_memory()
        # kb = deepcopy(self.knowledge_base)
        # kb.set_knowledge_memory(knowledge_memory=km)
        # self.knowledge_base = kb

        self.args.force_check_rate = 1.0

        # 临时保存
        save_path2 = r"D:\Downloads\tmp.txt"
        knowledge_memory = self.knowledge_base.get_inference_knowledge_memory()
        dct = {}
        for key, value in knowledge_memory.items():
            value = value[0]
            dct[key] = [{'rule_text': value.content,
                         'correct': 10,
                         'wrong': 0}]

        with open(save_path2, 'w', encoding="utf8") as f:
            f.write(str(dct))

        self.evaluate(is_valid=False, special_datasets=special_datasets)
