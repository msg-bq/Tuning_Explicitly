import json
import os.path
import pickle
from collections import defaultdict
from copy import deepcopy
from random import choices
from typing import List, Dict, Optional, Union, Tuple, overload, Set, Any

import numpy as np

from utils.ExtraNameSpace import PredictionCleanNameSpace, KnowledgeExtractionNameSpace
import utils.extract_knowledge

from sklearn.feature_extraction.text import TfidfVectorizer

knowledge_info = Union[Dict[str, str], 'KnowledgeSource']


class KnowledgeSource:
    def __init__(self, knowledge: str, question: str, rationale: 'Rationale', score: float):
        self.question = question
        self.rationale = rationale
        self.score = score

        assert isinstance(knowledge, str), "knowledge should be str, not %s" % type(knowledge)
        self.related_context = self._get_related_context(knowledge)
        self.concepts = None  # 对应context的类别，用concepts或categorys代替

    def _get_related_context(self, knowledge: str) -> str:
        """
        :return: 从question和rationale的组合中，获取和当前knowledge强相关的context，用于memorization
        默认策略为rationale中knowledge对应那行前面的字符
        这个函数虽然会带来高一个量级的时间复杂度，但比起代码改动便捷等是值得的
        """
        contexts = self.rationale.rationale.split('\n')
        for context in contexts:
            if knowledge in context:
                return context[:context.index(knowledge)]  # todo: 每个样例这里只考虑了第一次出现。如果后续修改的话
            # 最好是从related_context的输入端就进行控制，即knowledge本身抽取时记录对应的context，再处理
            # 但暂时校验是充分的，这一点没有特别的必要

        raise ValueError(f"{knowledge} not in rationale")

    def dumps(self):
        return json.dumps({'question': self.question,
                           'rationale': self.rationale.dumps(),
                           'score': self.score,
                           'context': self.related_context})

    @staticmethod
    def loads(dct: Dict) -> 'KnowledgeSource':
        return KnowledgeSource(knowledge=dct['knowledge'],
                               question=dct['question'],
                               rationale=Rationale.loads(dct['rationale']),
                               score=dct['score'])


class Knowledge:
    def __init__(self, content: str, related_info: Union[knowledge_info, List[knowledge_info]] = None):
        self.content = content
        # self.confidence = 0
        self.sources: Set[KnowledgeSource] = set()

        if related_info:
            self.load_source(related_info)

        # unused这个变量，我们依赖于correct+wrong的数量这个参数可以替代，所以不需要保留，这样也就失去了prompt+knowledge的最后一层必要了
        # 当然prompt里面+KB仍然还是不反对的，只是我们现在无所谓了

    def load_source(self, related_info: knowledge_info):
        if isinstance(related_info, dict):
            self.sources.add(KnowledgeSource(knowledge=self.content,
                                             question=related_info['question'],
                                             rationale=related_info['rationale'],
                                             score=related_info['score']))

        elif isinstance(related_info, KnowledgeSource):
            self.sources.add(related_info)

        elif isinstance(related_info, list):
            for k in related_info:
                self.load_source(k)

        else:
            raise TypeError

    def get_source_context(self) -> List[str]:
        sources = self.sources
        return [s.related_context for s in sources]

    def update_source(self, question: str = None, rationale: 'Rationale' = None, score: float = None,
                      sources: Union[KnowledgeSource, List[KnowledgeSource]] = None):
        """
        :param question:
        :param rationale:
        :param score:
        :param sources:  允许以KnowledgeSource的格式传入和更新，或者只传入前三个
        :return:
        """
        assert (question is not None and score is not None) or sources, ("must have question and score or sources, "
                                                                         "question: %s, score: %s, sources: %s" % (
                                                                             question, score, sources))

        self.sources.add(KnowledgeSource(knowledge=self.content,
                                         question=question,
                                         rationale=rationale,
                                         score=score))

        if sources:
            if isinstance(sources, list):
                self.sources = self.sources | set(sources)
            elif isinstance(sources, KnowledgeSource):
                self.sources.add(sources)
            else:
                raise TypeError

    def get_confidence(self) -> float:
        return self.get_success_num() / (len(self.sources) + 10)

    @staticmethod
    def _is_success(source: KnowledgeSource):
        """
        先采取默认策略，＞0.8的认为回答正确
        """
        return source.score > 0.8

    def get_success_num(self, concepts=None) -> int:
        sources = self.sources.copy()
        sources = [s for s in sources if s.concepts == concepts] if concepts else sources
        return len([s for s in sources if self._is_success(s)])

    def get_failure_num(self, concepts=None) -> int:
        sources = self.sources.copy()
        sources = [s for s in sources if s.concepts == concepts] if concepts else sources
        return len(sources) - self.get_success_num(concepts)

    def dumps(self):
        return {'content': self.content,
                'source_questions': [s.dumps() for s in self.sources]}

    @staticmethod
    def loads(knowledge: Dict) -> 'Knowledge':
        if 'content' not in knowledge:
            raise KeyError("The content is not in the dict")

        return Knowledge(content=knowledge['content'],
                         related_info=[KnowledgeSource.loads({**eval(s), **{'knowledge': knowledge['content']}})
                                       for s in knowledge['source_questions']])


class KnowledgeBase:
    def __init__(self):
        self._content_to_instance: Dict[str, Knowledge] = dict()
        self._knowledge_memory: Dict[Union[str, tuple], List[Knowledge]] = defaultdict(list)

    def __len__(self):
        return len(self._content_to_instance)

    def get_knowledge_by_content(self, content: str) -> Optional[Knowledge]:
        instance = self._content_to_instance.get(content)

        if instance is None:
            raise KeyError(f"{content} not in knowledge base")

        return instance

    @staticmethod
    def _split_knowledge_text(knowledge: str) -> List[str]:
        """
        如果是str，默认是以\n分隔的
        """
        if isinstance(knowledge, str):
            return [k.strip() for k in knowledge.split('\n') if k.strip()]

        raise TypeError("knowledge in '_split_knowledge_text' func should be str")

    def _find_knowledge_instance(self, knowledge: Union[str, List[Union[str, Knowledge]]]) -> Set[Knowledge]:
        knowledge = self._split_knowledge_text(knowledge) if isinstance(knowledge, str) else knowledge
        knowledge = [k.content if isinstance(k, Knowledge) else k for k in knowledge]  # 不假设"Knowledge类型=在KB里"

        knowledge_instances = set([self._content_to_instance[k] if k in self._content_to_instance
                                   else self._add_knowledge(k) for k in knowledge])

        return knowledge_instances

    def update_knowledge(self, knowledge_texts: Union[str, List[str], List[Knowledge]],
                         question: str, rationale: 'Rationale', score: float) -> List[Knowledge]:
        """
        需要字符串匹配，找到就返回，找不到就创建+返回
        :param knowledge_texts: 答题时从rationale中抽取的规则
        :param rationale:
        :param score:
        :param question: 问题
        本函数现在还不支持batch，连带着后面的find_instance之类的都不支持
        """
        if isinstance(knowledge_texts, str):
            knowledge_texts = [knowledge_texts]

        knowledge_instances = self._find_knowledge_instance(knowledge_texts)

        for k in knowledge_instances:
            k.update_source(question=question,
                            rationale=rationale,
                            score=score)

        return list(knowledge_instances)

    def __add_knowledge(self, knowledge: str) -> Knowledge:
        knowledge_instance = Knowledge(content=knowledge)
        self._content_to_instance[knowledge] = knowledge_instance

        return knowledge_instance

    def _add_knowledge(self, knowledge: Union[List[str], str]) -> Union[Knowledge, List[Knowledge]]:
        """
        这里需要一个添加knowledge的函数，包括将字符串转为str+查重+添加
        这个函数只add，不检查是否存在
        输入输出的type保持一致，均为iteration或普通数值类型
        """
        if isinstance(knowledge, str):
            return self.__add_knowledge(knowledge)
        else:
            return [self.__add_knowledge(k) for k in knowledge]

    # def _append_knowledge_source(self, knowledge_source: Union[KnowledgeSource, List[KnowledgeSource]]):
    #     pass

    def broadcast_knowledge_info(self):
        """可能存在的同步需求"""
        pass

    def _save_knowledge(self, save_path: str):
        with open(save_path, 'w') as f:
            out = [k.dumps() for k in self._content_to_instance.values()]
            for o in out:
                f.write(json.dumps(o) + '\n')

    def _load_knowledge(self, load_path: str):
        with open(load_path, 'r') as f:
            for line in f.readlines():
                knowledge = Knowledge.loads(json.loads(line))
                self._content_to_instance[knowledge.content] = knowledge

    def save_knowledge_memory(self, knowledge_memory_path: str, vectorizer_path: str = None):
        self._save_knowledge(knowledge_memory_path)

        if vectorizer_path and hasattr(self, 'vectorizer') and self.vectorizer is not None:
            with open(vectorizer_path, 'wb') as file:
                pickle.dump(self.vectorizer, file)

    def load_knowledge_memory(self, knowledge_memory_path: str, vectorizer_path: str = None):
        self._load_knowledge(knowledge_memory_path)

        if vectorizer_path:
            with open(vectorizer_path, 'rb') as file:
                self.vectorizer = pickle.load(file)

            learned_info = self._get_memory_learned_info()
            self._update_knowledge_memories(learned_info=learned_info, topn=2)

    def _update_knowledge_memory(self, concepts_chain: List[Tuple[str, tuple]],
                                 knowledge_chain: Union[Knowledge, List[Knowledge]]):
        """
        这个函数暂时封装的意义不大，只是看之后有无其他函数也需要调用
        """
        if isinstance(knowledge_chain, Knowledge):
            knowledge_chain = [knowledge_chain] * len(concepts_chain)

        assert len(concepts_chain) == len(knowledge_chain)

        for (context, concept), knowledge in zip(concepts_chain, knowledge_chain):
            if knowledge not in self._knowledge_memory[concept]:  # 这个复杂度倒是不值得set带来的麻烦
                self._knowledge_memory[concept].append(knowledge)

    def get_knowledge_memory(self):
        return self._knowledge_memory

    @classmethod
    def _extract_key_concepts_tfidf(cls, doc_list: Union[str, List[str]], vectorizer, topn=2) -> List[Tuple[str, tuple]]:
        def _sort_coo(coo_matrix):
            tuples = zip(coo_matrix.col, coo_matrix.data)
            return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

        if isinstance(doc_list, str):
            doc_list = [doc_list]

        if not hasattr(vectorizer, "reversed_vocabulary"):
            vectorizer.reversed_vocabulary = {v: k for k, v in vectorizer.vocabulary_.items()}

        doc_concepts = []

        for doc in doc_list:
            tf_idf_vector = vectorizer.transform([doc])
            sorted_items = _sort_coo(tf_idf_vector.tocoo())

            sorted_items = [(vectorizer.reversed_vocabulary[idx], score) for
                            idx, score in sorted_items]

            sorted_items = [(word, doc.lower().index(word.lower())) for word, score in sorted_items]
            sorted_items = sorted(sorted_items, key=lambda x: x[1])

            key_concepts = [word for word, _ in sorted_items[:topn]]

            doc_concepts.append((doc, tuple(key_concepts)))

        return doc_concepts

    @classmethod
    def _extract_key_concepts_hyperplane(cls, doc_list: str | list[str], vectorizer) -> list[tuple[str, tuple]]:
        # encode
        complaint_embeddings = cls.encoder.encode(list(data['review']))

        n = len(doc_list)  # Number of documents
        n_planes = int(np.log2(n / 16))  # hack: 这里也有个特殊的参数
        n_repeats = 25  # Number of hash tables



    @classmethod
    def extract_key_concepts(cls, doc_list: Union[str, List[str]], vectorizer, func_name='hyperplane',
                             *args, **kwargs) -> list[tuple[str, tuple]]:
        funcs_dict = {'tfidf': cls._extract_key_concepts_tfidf,
                      'hyperplane': cls._extract_key_concepts_hyperplane}

        func = funcs_dict.get(func_name, cls._extract_key_concepts_tfidf)

        return func(doc_list, vectorizer, *args, **kwargs)

    def _build_conceptual_memory(self, doc_list: List[str]):
        """
        这里算法可选，能达成为句子提供分类名词的需求即可，我们目前采取的策略是tf idf获取keywords作为概念
        """

        def calculate_tf_idf(doc_lst: List[str]):
            vectorizer = TfidfVectorizer(stop_words='english')
            vectorizer.fit_transform(doc_lst)

            return vectorizer

        self.vectorizer = calculate_tf_idf(doc_lst=doc_list)

    def _get_memory_learned_info(self) -> List[Knowledge]:
        """
        为memorization准备相关文本，比如对于base版本的knowledge base，这里对应所有knowledge即可（related_context内置进去了）
        对于加了并查集去查的knowledge base，取每个类的root节点的knowledge instance

        一般来说主要传递的是knowledge+能获取对应context、score信息的入参（所以其他地方修改实现的话，这里也简单调整下
        """
        return list(self._content_to_instance.values())

    def _update_knowledge_memories(self, learned_info: List[Knowledge], topn: int = 2):
        for knowledge in learned_info:
            contexts = knowledge.get_source_context()
            doc_concepts = self.extract_key_concepts(doc_list=contexts,
                                                     vectorizer=self.vectorizer,
                                                     topn=topn)

            # 这个数据结构有点奇怪，应该在一个更优雅的地方把sources改了，或者就在上面
            for (_, concepts), source in zip(doc_concepts, knowledge.sources):
                source.concepts = concepts

            self._update_knowledge_memory(concepts_chain=doc_concepts,
                                          knowledge_chain=knowledge)

    def _extract_doc_list_from_learned_info(self) -> Tuple[List[str], List[Knowledge]]:
        learned_info: List[Knowledge] = self._get_memory_learned_info()

        doc_list = []
        for knowledge in learned_info:
            doc_list.extend(knowledge.get_source_context())

        return doc_list, learned_info

    @staticmethod
    def stop_words_filter(doc_list: List[str], stop_words: List[str]):
        """此函数留空，暂时没有纳入算法"""
        # tokenizer, 这里用split代替
        filter_doc_list = []
        for doc in doc_list:
            word_list = doc.split()
            filter_word_list = [word for word in word_list if word not in stop_words]
            if word_list:
                filter_doc_list.append(' '.join(filter_word_list))

        return filter_doc_list


    def build_conceptual_memory(self):
        """
        实现功能的函数是_build_conceptual_memory，换记忆方法时需要重载。而此函数很多是为了日志、改分等对齐而写的
        """
        doc_list, learned_info = self._extract_doc_list_from_learned_info()

        if not doc_list:
            return

        # 只是实现在这里，但本次投稿、训练时候先不加
        # doc_list = self.stop_words_filter(doc_list=doc_list,
        #                                   stop_words=['sentence', 'phrase', 'retrieve',
        #                                               'have', 'answer'])
        # 对于tf-idf，从源头删除就够了，因为inference时，train不存在的会被丢弃

        self._build_conceptual_memory(doc_list=doc_list)

        self._update_knowledge_memories(learned_info=learned_info, topn=2)

    @staticmethod
    def _sort_knowledge_memory(knowledge_memory: Dict[str, List[Knowledge]]) \
            -> Dict[Union[str, tuple], List[Knowledge]]:
        """
        :return: 对knowledge memory做排序，用于推理阶段
        """
        for k in knowledge_memory:
            knowledge_memory[k] = sorted(knowledge_memory[k],
                                         key=lambda x: x.get_confidence(),
                                         reverse=True)

        return knowledge_memory

    @staticmethod
    def _filter_knowledge_memory(knowledge_memory: Dict[str, List[Knowledge]]) \
            -> Dict[Union[str, tuple], List[Knowledge]]:
        """
        推理时过滤掉质量过低的knowledge
        """
        for k in knowledge_memory:
            knowledge_memory[k] = [v for v in knowledge_memory[k] if v.get_success_num(k) > v.get_failure_num(k)/2]

        knowledge_memory = {k: v for k, v in knowledge_memory.items() if v}

        return knowledge_memory

    def get_inference_knowledge_memory(self) -> Dict[Union[str, tuple], List[Knowledge]]:
        return self._sort_knowledge_memory(self._filter_knowledge_memory(self.get_knowledge_memory()))

    def set_knowledge_memory(self, knowledge_memory: Dict[str, List[Knowledge]]):
        self._knowledge_memory = knowledge_memory

    def __deepcopy__(self, memo):
        """
        创建一个KB的副本
        """
        cls = self.__class__
        new_kb = cls.__new__(cls)
        memo[id(self)] = new_kb
        for k, v in self.__dict__.items():
            setattr(new_kb, k, deepcopy(v, memo))

        return new_kb


class Rationale:  # 修正到只有两个属性
    """
    top-N的结果，rationale和prediction
    """

    def __init__(self, rationale: str, prediction: str, *args, **kwargs):
        self.rationale = rationale.strip()
        self.prediction = self.clean_prediction(prediction)
        self.knowledge_texts = list(self.get_knowledge_texts())  # 这里就不转换knowledge_text为knowledge instance了，感觉没必要
        # 就靠knowledge source存储即可，分析

    @staticmethod
    @KnowledgeExtractionNameSpace.register("Example")
    def extract_knowledge_texts(rationale) -> List[str]:
        pass

    def get_knowledge_texts(self) -> Union[Set[str], List[str]]:
        if hasattr(self, 'knowledge_texts') and self.knowledge_texts:
            return self.knowledge_texts

        knowledge_texts = self.extract_knowledge_texts(self.rationale)

        self.knowledge_texts = set(knowledge_texts)

        return self.knowledge_texts

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

    def dumps(self):
        return json.dumps(self.__dict__)

    @staticmethod
    def loads(json_str) -> 'Rationale':
        return Rationale(**json.loads(json_str))


class Example:
    """
    给dataloader作为样例的，所以不包含score这个字段
    """

    def __init__(self, question: str, gold_label: str, rationales: List[Rationale] = None, *args, **kwargs):
        self.question = question.strip()
        self.gold_label = gold_label.strip()
        self.rationales = rationales if rationales else []

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

        self.rationales.append(new_rationale_instance)

    def __eq__(self, other):
        if isinstance(other, Example):
            return self.__dict__ == other.__dict__
        else:
            raise AttributeError("Incorrect attribute!")

    def _merge_attribute(self, attr: str, other_attr_value: Union[List, str]):
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

    def _check_QA(self, other_qa: Tuple[str, str]):
        if self.question and self.question.strip() != other_qa[0].strip():
            raise Warning("The question is not the same.")

        if self.gold_label and self.gold_label.strip() != other_qa[1].strip():
            raise Warning("The gold_label is not the same.")

        return True

    @staticmethod
    def _adjust_merge_example_to_rationale(merge_example: Dict[str, Union[str, Rationale]]) \
            -> Dict[str, Union[str, Rationale, List[Rationale]]]:
        """
        将merge_example中的rationale部分(包括rationale和prediction两个key)转换为Rationale类
        """
        rationale = merge_example.pop('rationale')
        prediction = merge_example.pop('prediction')
        rationale_class = Rationale(rationale=rationale, prediction=prediction)

        merge_example['rationales'] = rationale_class

        return merge_example

    def update(self, example: [str, Dict[str, str], 'Example'], args=None):
        """
        根据example里面的更新self对应的值。请注意，这个函数用于更新某个样例的情况，而不建议将样例1改变为样例2（注意到
        list我们是直接append，而不是替换的）
        当待修改的question和gold_label有任一不同时，会抛出警告
        """
        if 'rationale' in example and 'prediction' in example:
            merge_example = self._adjust_merge_example_to_rationale(example)
        else:
            merge_example = example

        if isinstance(example, str):  # parse出来一个dict
            self.parse_response(example, args)
        elif isinstance(example, Example):
            merge_example = example.to_dict()

        if isinstance(merge_example, dict):
            self._check_QA((merge_example.get('question'), merge_example.get('gold_label')))

            for key in merge_example.keys():
                if key in self.__dict__:
                    self._merge_attribute(key, merge_example[key])
                else:
                    raise KeyError("The key is not in the DatasetLoader")

        else:
            raise TypeError("Incorrect type.")

        return self

    def to_dict(self):
        return {'question': self.question, 'gold_label': self.gold_label,
                'rationale': [k.rationale for k in self.rationales],
                'prediction': [k.prediction for k in self.rationales]}

    def parse_response(self, response: str, args=None) -> Dict[str, str]:
        """
        这里只会传入A：后面生成的部分
        """
        question_name = self.question.split('\n')[-1].strip()[10:].strip()[
                        :-6].lower()  # this special case only serve for
        # CLUTRR, and it doesn't work for the other benchmarks, so it is not a big deal, and you can adjust it for free.
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
        return choices(self.rationales, k=k)

    def __repr__(self):
        return str({"question": self.question, "gold_label": self.gold_label, "rationale": self.rationales.__repr__()})

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
