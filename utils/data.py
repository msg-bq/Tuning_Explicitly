import math
import pickle
import re
import string
from collections import defaultdict
from random import choice, choices
from typing import List, Dict, Optional, Union, Tuple, overload, Set, Callable
import Levenshtein
import numpy as np
import pandas as pd
import operator
import random

from sentence_transformers import SentenceTransformer

from utils.ExtraNameSpace import PredictionCleanNameSpace, KnowledgeExtractionNameSpace
import utils.extract_knowledge

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class KnowledgeSource:
    def __init__(self, knowledge: str, question: str, rationale: str, socre: float):
        self.question = question
        self.rationale = rationale
        self.score = socre

        self.related_context = self._get_related_context(knowledge)

    def _get_related_context(self, knowledge) -> str:
        """
        :return: 从question和rationale的组合中，获取和当前knowledge强相关的context，用于memorization
        默认策略为rationale中knowledge对应那行前面的字符
        这个函数虽然会带来高一个量级的时间复杂度，但比起代码改动便捷等是值得的
        """
        contexts = self.rationale.split('\n')
        for context in contexts:
            if knowledge in context:
                return context[:context.index(knowledge)]

        raise ValueError(f"{knowledge} not in rationale")

class Knowledge:
    def __init__(self, content: str):
        self.content = content

        # self.confidence = 0
        self.success_used = 0 # 有了分场景的successed_used，那其实总的也没必要了
        self.success_unused = 0 # unused是不可能分concept处理出来的
        self.failure_used = 0
        self.failure_unused = 0
        self.source = set()

    # def _load_knowledge(self, knowledge: Dict):
    #     if 'content' not in knowledge:
    #         raise KeyError("The content is not in the dict")
    #
    #     self.content = knowledge['content']
    #     self.confidence = knowledge.get('confidence', 0.0)
    #     self.success_used = knowledge.get('success_used', 0)
    #     self.success_unused = knowledge.get('success_unused', 0)
    #     self.failure_used = knowledge.get('failure_used', 0)
    #     self.failure_unused = knowledge.get('failure_unused', 0)
    #     self.source_questions = knowledge.get('source_questions', set())
    #
    #     return self


# def _update_knowledge_score(given_knowledge: Set[Knowledge], extracted_knowledge: Set[Knowledge],
#                             question: 'Example', score: float) -> float:
#     for knowledge in given_knowledge & extracted_knowledge:
#         knowledge.confidence += 0.1
#     for knowledge in given_knowledge - extracted_knowledge:
#         knowledge.confidence -= 0.001
#
#     for knowledge in extracted_knowledge:
#         knowledge.confidence += score
#
#     for knowledge in given_knowledge & extracted_knowledge:
#         knowledge.source_questions.add(question.question)
#
#     for knowledge in extracted_knowledge:
#         if score > 0:
#             knowledge.success_used += 1
#         else:
#             knowledge.failure_used += 1
#     for knowledge in given_knowledge - extracted_knowledge:
#         if score > 0:
#             knowledge.success_unused += 1
#         else:
#             knowledge.failure_unused += 1
#
#     return score


class KnowledgeBase:
    def __init__(self):
        self._content_to_instance: Dict[str, Knowledge] = dict()
        self._knowledge_memory: Dict[str, List[Knowledge]] = defaultdict(list)

    def get_knowledge_by_content(self, content: str) -> Optional[Knowledge]:
        instance = self._content_to_instance.get(content)

        if instance is None:
            raise KeyError(f"{content} not in knowledge base")

        return instance


    @staticmethod
    # def _split_knowledge_text(knowledge: Union[str, List[str], List[Knowledge]]) -> List[str]:
    #     """
    #     如果是str，默认是以\n分隔的
    #     """
    #     if isinstance(knowledge, str):
    #         return [k.strip() for k in knowledge.split('\n') if k.strip()]
    #     elif isinstance(knowledge, list):
    #         knowledge_list = [k.strip() if isinstance(k, str) else k for k in knowledge]
    #         knowledge_list = [k for k in knowledge_list if k]
    #
    #         return knowledge_list

    def _find_knowledge_instance(self, knowledge: Union[str, List[str], List[Knowledge]], question: 'Example') \
    #         -> Set[Knowledge]:
    #     knowledge = self._split_knowledge_text(knowledge) if isinstance(knowledge, str) else knowledge
    #     knowledge = [k.content if isinstance(k, Knowledge) else k for k in knowledge]
    #
    #     knowledge_instances = set([self._content_to_instance[k] if k in self._content_to_instance
    #                                else self._add_knowledge(k, question.question) for k in knowledge])
    #
    #     return knowledge_instances
    #
    # def update_knowledge(self, added_knowledge: Union[str, List[str], List[Knowledge]],
    #                      new_knowledge: Union[str, List[str], List[Knowledge]],
    #                      question: 'Example', score: float) -> List[Knowledge]:
    #     """
    #     需要字符串匹配，找到就返回，找不到就创建+返回
    #     :param added_knowledge: 答题时从knowledgebase中抽取的规则
    #     :param new_knowledge: 答题时从rationale中抽取的规则
    #     :param question: 问题
    #     """
    #     given_knowledge = self._find_knowledge_instance(added_knowledge, question)
    #     extracted_knowledge = self._find_knowledge_instance(new_knowledge, question)
    #
    #     _update_knowledge_score(given_knowledge, extracted_knowledge, question, score)
    #
    #     return list(extracted_knowledge)
    #
    # def __add_knowledge(self, knowledge: str, question: str, score: float) -> Knowledge:
    #     knowledge_instance = Knowledge(content=knowledge, question=question, confidence=score)
    #     self._content_to_instance[knowledge] = knowledge_instance
    #
    #     return knowledge_instance
    #
    # def _add_knowledge(self, knowledge: Union[List[str], str],
    #                    questions: Union[List[str], str],
    #                    scores: Union[List[float], float] = None) -> Union[Knowledge, List[Knowledge]]:
    #     """
    #     这里需要一个添加knowledge的函数，包括将字符串转为str+查重+添加
    #     这个函数只add，不检查是否存在
    #     """
    #     if not scores:
    #         scores = 1.0 if isinstance(knowledge, str) else [1.0] * len(knowledge)
    #
    #     if isinstance(knowledge, str):
    #         return self.__add_knowledge(knowledge, questions, scores)
    #     elif isinstance(knowledge, list):
    #         if isinstance(questions, str):
    #             questions = [questions] * len(knowledge)
    #
    #         new_knowledge_instances = [self.__add_knowledge(knowledge, question, score) for
    #                                    knowledge, question, score in zip(knowledge, questions, scores)]
    #         return new_knowledge_instances
    #
    # def __len__(self):
    #     return len(self._content_to_instance)

#     def _read_knowledges(self, knowledges: Union[List[Dict], List[str]]):
#         for knowledge_dict in knowledges:
#             if isinstance(knowledge_dict, str):
#                 knowledge_dict = eval(knowledge_dict)
#
#             knowledge = Knowledge(content="", question="")._load_knowledge(knowledge_dict)
#             self._content_to_instance[knowledge.content] = knowledge  # 这里应该有一个存在就不读入了的函数。
#             # 或者说这里本就应该调取update来完成存储。不过暂时先这样，因为目前的update还不够灵活
#
#     def read_knowledge(self, knowledge_base_path: str):
#         """
#         读入的是完整的knowledges，{'knowledge': "Guillermina is Christopher's daughter.", 'confidence': -22.23923742923761, 'success_used': 0, 'success_unused': 97, 'failure_used': 17, 'failure_unused': 674}
#         """
#         with open(knowledge_base_path, 'r') as f:
#             knowledge = [l for l in f.readlines() if l.strip()]
#             self._read_knowledges(knowledge)
#
#     def load_vectorizer(self, vectorizer_path: str):
#         with open(vectorizer_path, 'rb') as file:
#             self.vectorizer = pickle.load(file)
#
#     def broadcast_knowledge_info(self):
#         """可能存在的同步需求"""
#         pass
#
#     def save(self, save_path: str):
#         with open(save_path, 'w') as f:
#             out = [k.__dict__ for k in self._content_to_instance.values()]
#             f.write('\n'.join([str(o) for o in sorted(out, key=lambda x: x['confidence'])]))
#
#     def save_knowledge_memory(self, knowledge_memory_path: str, vectorizer_path: str):
#         with open(knowledge_memory_path, 'w', encoding='utf8') as f:
#             out = {k: v for k, v in self._knowledge_memory.items()}
#             f.write(str(out))
#
#         with open(vectorizer_path, 'wb') as file:
#             pickle.dump(self.vectorizer, file)
#
#     def update_knowledge_memory(self, concepts_chain: List[tuple], knowledge_chain: List[str], rationale: str,
#                                 question: str, result: str, score: float):
#         for concepts, knowledge in zip(concepts_chain, knowledge_chain):
#             if concepts not in self._knowledge_memory:
#                 self._knowledge_memory[concepts] = []
#             found = False
#             for i in range(len(self._knowledge_memory[concepts])):
#                 if self._knowledge_memory[concepts][i]['knowledge_text'] == knowledge:  # 奇怪的复杂度
#                     # score = 1 if score > 0.5 else -1
#                     score = 1 / 10 if score > 0.5 else 0
#                     # self._knowledge_memory[concepts][i]['confidence'] += score if score > 0 else 2*score
#
#                     if result == 'correct':
#                         self._knowledge_memory[concepts][i]['correct'] += 1
#                     elif result == 'wrong':
#                         self._knowledge_memory[concepts][i]['wrong'] += 1
#                     else:
#                         raise ValueError
#
#                     self._knowledge_memory[concepts][i]['confidence'] = self._knowledge_memory[concepts][i][
#                                                                             'correct'] / (
#                                                                                 self._knowledge_memory[concepts][i][
#                                                                                     'wrong'] + 10)
#
#                     self._knowledge_memory[concepts][i]['rationale'].append(rationale)
#                     self._knowledge_memory[concepts][i]['question'].append(question)
#                     # self._knowledge_memory[concepts].sort(key=operator.itemgetter('correct'), reverse=True)
#                     found = True
#                     break
#             if not found:
#                 if result == 'correct':
#                     _new = {'knowledge_text': knowledge, 'correct': 1, 'wrong': 0, 'confidence': score,
#                             'rationale': [rationale], 'question': [question]}
#                 elif result == 'wrong':
#                     _new = {'knowledge_text': knowledge, 'correct': 0, 'wrong': 1, 'confidence': score,
#                             'rationale': [rationale], 'question': [question]}
#                 else:
#                     raise ValueError
#                 self._knowledge_memory[concepts].append(_new)
#                 # append到最后不需要额外排序
#
#     def get_knowledge_memory(self):
#         return self._knowledge_memory
#
#     @classmethod
#     def _extract_key_concepts(self, doc_list: Union[str, List[str]], vectorizer, topn=2) -> List[Tuple[str, tuple]]:
#         def _sort_coo(coo_matrix):
#             tuples = zip(coo_matrix.col, coo_matrix.data)
#             return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
#
#         if isinstance(doc_list, str):
#             doc_list = [doc_list]
#
#         if not hasattr(vectorizer, "reversed_vocabulary"):
#             vectorizer.reversed_vocabulary = {v: k for k, v in vectorizer.vocabulary_.items()}
#
#         doc_concepts = []
#
#         for doc in doc_list:
#             tf_idf_vector = vectorizer.transform([doc])
#             sorted_items = _sort_coo(tf_idf_vector.tocoo())
#
#             sorted_items = [(vectorizer.reversed_vocabulary[idx], score) for
#                             idx, score in sorted_items]
#
#             sorted_items = [(word, doc.lower().index(word.lower())) for word, score in sorted_items]
#             sorted_items = sorted(sorted_items, key=lambda x: x[1])
#
#             key_concepts = [word for word, _ in sorted_items[:topn]]
#             # for idx, score in sorted_items[:topn]:
#             #     key_concepts.append(vectorizer.reversed_vocabulary[idx])
#
#             doc_concepts.append((doc, tuple(key_concepts)))
#
#         return doc_concepts
#
#     def _build_conceptual_memory(self, doc_list: List[str]):
#         """
#         这里算法可选，能达成为句子提供分类名词的需求即可，我们目前采取的策略是tf idf获取keywords作为概念
#         """
#
#         def calculate_tf_idf(doc_list: List[str]):
#             vectorizer = TfidfVectorizer(stop_words='english')
#             vectorizer.fit_transform(doc_list)
#
#             return vectorizer
#
#         self.vectorizer = calculate_tf_idf(doc_list=doc_list)
#
#     def build_conceptual_memory(self):
#         """
#         实现功能的函数是_build_conceptual_memory，换记忆方法时需要重载。而此函数很多是为了日志、改分等对齐而写的
#         """
#         learned_info = self.backward_buffer
#
#         doc_list = []
#         for example in learned_info:
#             lines = [l.strip() for l in example['rationale'].split('\n')]
#             doc_list.extend(lines)
#
#         if not doc_list:
#             self.backward_buffer = []
#             return
#
#         # print("doc_list", doc_list)
#         self._build_conceptual_memory(doc_list=doc_list)
#
#         topn = 2
#         for example in learned_info:
#             knowledges = example['knowledges']
#
#             filter_lines = []
#             for knowledge in knowledges:
#                 # for line, concepts in doc_concepts: # 复杂度非常不妥当
#                 for line in doc_list:
#                     if knowledge in line:
#                         filter_lines.append(line[:line.index(knowledge)])
#                         break
#
#             assert len(filter_lines) == len(knowledges), "{} != {}".format(len(filter_lines), len(knowledges))
#
#             doc_concepts = self._extract_key_concepts(doc_list=filter_lines, vectorizer=self.vectorizer, topn=topn)
#
#             self.update_knowledge_memory(concepts_chain=[c for _, c in doc_concepts],
#                                          knowledge_chain=knowledges,
#                                          question=example['question'],
#                                          rationale=example['rationale'],
#                                          result=example['result'],
#                                          score=example['score'])
#
#         self.backward_buffer = []
#
#
class Rationale:  # 修正到只有两个属性
    """
    top-N的结果，rationale和prediction
    """

    def __init__(self, rationale: str, prediction: str):
        self.rationale = rationale.strip()
        self.prediction = self.clean_prediction(prediction)
        self.knowledge_texts = self.extract_knowledge() # 这里就不转换knowledge_text为knowledge instance了，感觉没必要
        # 就靠knowledge source存储即可，分析

    @classmethod
    @KnowledgeExtractionNameSpace.register("Example")
    def extract_knowledge(cls) -> set[str]:
        pass

    @classmethod
    @PredictionCleanNameSpace.register("Example")
    def clean_prediction(cls, prediction: str) -> str:
        return prediction

    def update(self, new_rationale: Dict[str, str]):
        """
        做对了更新覆盖，错了不变
        """
        for key in new_rationale.keys():
            if key in self.__dict__.keys():
                self.__dict__[key] = new_rationale[key]
            else:
                raise KeyError("The key is not in the DatasetLoader")

        return self

    def __eq__(self, other):
        if isinstance(other, Rationale):
            return self.__dict__ == other.__dict__
        else:
            raise AttributeError("Incorrect attribute!")

    def __repr__(self):
        return str({"rationale": self.rationale, "prediction": self.prediction})


class Example:
    def __init__(self, question: str, gold_label: str, answers: List[Tuple[Rationale, float]] = None, *args, **kwargs):
        self.question = question.strip()
        self.gold_label = gold_label.strip()
        self.answers: List[Tuple[Rationale, float]] = [] if answers is None else answers

    def update_rationale(self, rationale: Union[Dict[str, str], Rationale, List]):
        new_rationale_instance = None

        if isinstance(rationale, dict):
            new_rationale_instance = Rationale(rationale=rationale['rationale'], prediction=rationale['prediction'])
        elif isinstance(rationale, Rationale):
            new_rationale_instance = rationale
        elif isinstance(rationale, list):
            for k in rationale:
                self.update_rationale(k)
            return

        self.rationale.append(new_rationale_instance)

    def __eq__(self, other):
        if isinstance(other, Example):
            return self.__dict__ == other.__dict__
        else:
            raise AttributeError("Incorrect attribute!")

    def _merge_arrtibute(self, attr: str, other_attr_value: Union[List, str]):
        """
        合并两个属性，这里只是简单的合并，不做去重
        """
        if getattr(self, attr, None) is None:
            raise AttributeError("The attribute is not in the Example")

        if isinstance(getattr(self, attr), list):
            if isinstance(other_attr_value, list):
                getattr(self, attr).extend(other_attr_value)
            elif isinstance(other_attr_value, str) or isinstance(other_attr_value, Rationale):
                getattr(self, attr).append(other_attr_value)
            else:
                raise TypeError("Incorrect type.")

        elif isinstance(getattr(self, attr), str):
            if isinstance(other_attr_value, str):
                setattr(self, attr, other_attr_value)
            else:
                raise TypeError("Incorrect type.")

    def _check_QA(self, other_QA: Tuple[str, str]):
        if self.question and self.question.strip() != other_QA[0].strip():
            raise Warning("The question is not the same.")

        if self.gold_label and self.gold_label.strip() != other_QA[1].strip():
            raise Warning("The gold_label is not the same.")

        return True

    def adjust_merge_example_to_rationale(self, merge_example: Dict[str, str]) -> Dict[
        str, Union[str, Rationale, List[Rationale]]]:
        """
        将merge_example中的rationale部分(包括rationale和prediction两个key)转换为Rationale类
        """
        rationale = merge_example.pop('rationale', None)
        prediction = merge_example.pop('prediction', None)
        rationale_class = Rationale(rationale=rationale, prediction=prediction)

        merge_example['rationale'] = rationale_class

        return merge_example

    def update(self, example: [str, Dict[str, str], 'Example'], args=None):
        """
        根据example里面的更新self对应的值。请注意，这个函数用于更新某个样例的情况，而不建议将样例1改变为样例2（注意到
        list我们是直接append，而不是替换的）
        当待修改的question和gold_label有任一不同时，会抛出警告
        """
        if 'rationale' in example and 'prediction' in example:
            merge_example = self.adjust_merge_example_to_rationale(example)
        else:
            merge_example = example

        if isinstance(example, str):  #parse出来一个dict
            self.parse_response(example, args)
        elif isinstance(example, Example):
            merge_example = example.to_dict()

        if isinstance(merge_example, dict):
            self._check_QA((merge_example.get('question', None), merge_example.get('gold_label', None)))

            for key in merge_example.keys():
                if key in self.__dict__:
                    self._merge_arrtibute(key, merge_example[key])
                else:
                    raise KeyError("The key is not in the DatasetLoader")

        else:
            raise TypeError("Incorrect type.")

        return self

    def to_dict(self):
        return {'question': self.question, 'gold_label': self.gold_label,
                'rationale': [k.rationale for k in self.rationale],
                'prediction': [k.prediction for k in self.rationale]}

    def parse_response(self, response: str, args=None) -> Dict[str, str]:
        """
        这里只会传入A：后面生成的部分
        """
        question_name = self.question.split('\n')[-1].strip()[10:].strip()[:-6].lower()
        pred_trigger = args.pred_trigger.lower() if args else "the answer is"

        if pred_trigger in response.lower():
            response_lst = response.lower().split(pred_trigger)
            length = len(response)
            prediction = response[length - len(response_lst[-1]):].strip()
            rationale = response[:-(len(response_lst[-1]) + len(pred_trigger))].strip()
        elif question_name in response.lower():
            response_lst = response.lower().split(question_name)
            length = len(response)
            prediction = response[length - len(response_lst[-1]):].strip()
            rationale = response[:-(len(response_lst[-1]) + len(question_name))].strip()
        else:
            rationale = response.strip()
            prediction = response.strip()

        return {'question': self.question, 'gold_label': self.gold_label,
                'rationale': rationale, 'prediction': prediction}

    def Top_k_rationale(self, k: int = 1):
        """
        返回score排名 top-k的rationale
        """
        return choices(self.rationale, k=k)

    def __repr__(self):
        return str({"question": self.question, "gold_label": self.gold_label, "rationale": self.rationale.__repr__()})

    def __hash__(self):
        return hash((self.question, self.gold_label))


class DatasetLoader:  # 命名上和torch的多加了个set
    def __init__(self, data: List[Example]):
        # {question, gold_label: data_instance}
        self._question_label_2_data_instance = dict()
        self._data_instance_list = []

        for e in data:
            self._data_instance_list.append(e)

        self._build_index()

    def __len__(self):
        return len(self._data_instance_list)

    def __getitem__(self, item):
        return self._data_instance_list[item]

    def __repr__(self):
        return " ".join([str(self._question_label_2_data_instance[d]) for d in self._question_label_2_data_instance])

    def __iter__(self):
        self._iter_index = -1
        return self

    def __next__(self):
        self._iter_index += 1
        if self._iter_index >= len(self._data_instance_list):
            raise StopIteration()
        return self._data_instance_list[self._iter_index]

    def _build_index(self):
        for data_instance in self._data_instance_list:
            question, gold_label = data_instance.question, data_instance.gold_label
            key = (question, gold_label)
            if key in self._question_label_2_data_instance:
                raise KeyError("The question and gold_label is already in the DatasetLoader")
            else:
                self._question_label_2_data_instance[key] = data_instance

    @overload
    def find(self, key: Tuple[str, str]) -> Optional[Example]:
        ...

    @overload
    def find(self, key: Tuple[str, str], default) -> Optional[Example]:
        ...

    def find(self, key, *args):
        if key in self._question_label_2_data_instance:
            return self._question_label_2_data_instance[key]
        else:
            if args:
                return args[0]
            else:
                raise KeyError("The question and gold_label pair is not in the DatasetLoader")
