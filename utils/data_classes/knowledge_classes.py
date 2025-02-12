import ast
import json
import math
import warnings
from random import choices
from typing import Optional, Union, overload

from utils.ExtraNameSpace import PredictionCleanNameSpace, KnowledgeExtractionNameSpace


knowledge_info = Union[dict[str, str], 'KnowledgeSource']


class Category:
    """
    接近于一个abstract class
    """
    def __init__(self, category, original_key):
        assert not isinstance(category, Category), ("原则上我们不限制category的类型，但需要支持Category type的情况很少"
                                                    "而不小心在key的构造时刻多套一层Category却容易发生，因此这里暂时拒绝"
                                                    "Category类型作为输入")
        self.category = category  # todo: 之后把原文弄进来
        self.original_key = original_key if isinstance(original_key, set) else {original_key}

    def __hash__(self):
        return hash(self.category)

    def __eq__(self, other):
        return type(self) is type(other) and self.category == other.category

    def __str__(self):
        return f"Category: {self.category}, OriginalKey: {self.original_key}"

    def __getstate__(self):
        """控制对象的序列化方式"""
        return {'category': self.category, 'original_key': self.original_key}

    def __setstate__(self, state):
        """控制对象的反序列化方式"""
        self.category = state['category']
        self.original_key = state['original_key']

    def dumps(self) -> tuple[str, tuple[str, ...]]:
        return self.category, tuple(self.original_key)

    @staticmethod
    def loads(value: str | tuple[str, tuple[str, ...]]) -> 'Category':
        if isinstance(value, str):
            value = ast.literal_eval(value)
        category, original_key = value
        return Category(category=category, original_key=set(original_key))

    def append_original_key(self, key):
        self.original_key.add(key)


class KnowledgeSource:
    def __init__(self, knowledge: str, question: str, rationale: 'Rationale', score: float):
        self.question = question
        self.rationale = rationale
        self.score = score

        assert isinstance(knowledge, str), "knowledge should be str, not %s" % type(knowledge)
        self.related_context = self._get_related_context(knowledge)
        self.concepts: Category | tuple[Category, ...] | None = None  # 对应context的类别，用concepts或categories代替

    def _get_related_context(self, knowledge: str) -> str:
        """
        :return: 从question和rationale的组合中，获取和当前knowledge强相关的context，用于memorization
        默认策略为rationale中knowledge对应那行前面的字符
        这个函数虽然会带来高一个量级的时间复杂度，但比起代码改动便捷等是值得的
        """
        contexts = self.rationale.rationale.split('\n')
        for context in contexts:
            if knowledge in context:
                return context[:context.index(knowledge)]
                # todo: 每个样例这里只考虑了第一次出现。如果后续修改的话
            # 最好是从related_context的输入端就进行控制，即knowledge本身抽取时记录对应的context，再处理
            # 但暂时校验是充分的，这一点没有特别的必要

        raise ValueError(f"{knowledge} not in rationale")

    @staticmethod
    def __stop_words_filter(text: str, stop_words: list[str]):  # fixme: 回头看放在哪里
        """此函数留空，暂时没有纳入算法"""
        # tokenizer, 这里用split代替
        word_list = text.split()
        filter_word_list = [word for word in word_list if word.lower() not in stop_words]
        if word_list:
            return ' '.join(filter_word_list)

        return "null"

    def dumps(self):
        return json.dumps({'question': self.question,
                           'rationale': self.rationale.dumps(),
                           'score': self.score,
                           'context': self.related_context})

    @staticmethod
    def loads(dct: dict) -> 'KnowledgeSource':
        return KnowledgeSource(knowledge=dct['knowledge'],
                               question=dct['question'],
                               rationale=Rationale.loads(dct['rationale']),
                               score=dct['score'])


class Knowledge:
    def __init__(self, content: str, related_info: Union[knowledge_info, list[knowledge_info]] = None):
        self.content = content
        # self.confidence = 0
        self.sources: set[KnowledgeSource] = set()

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

    def get_source_context(self) -> list[str]:
        sources = self.sources
        return [s.related_context for s in sources]

    def update_source(self, question: str = None, rationale: 'Rationale' = None, score: float = None,
                      sources: Union[KnowledgeSource, list[KnowledgeSource]] = None):
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

    @staticmethod
    def _is_same_categories(x_categories: Category | list[Category] | tuple[Category],
                            y_categories: Category | list[Category] | tuple[Category]) -> bool:
        """
        对比两个categories是否是一致的，用于筛选某个特定category的knowledge有哪些
        如果是list，则50%以上的一致即可
        XXX: 上面是最初的设想。但现在我们注意到一个问题是，n个超平面其实是为了召回。那...没有50%的说法，只要A在B之内即可
        """
        def _is_iter_category(value):
            if (isinstance(value, list) or isinstance(value, tuple)) and value and isinstance(value[0], Category):
                return True
            return False

        if isinstance(x_categories, Category) and isinstance(y_categories, Category):
            return x_categories == y_categories
        elif isinstance(x_categories, Category) and _is_iter_category(y_categories):
            return x_categories in y_categories
        elif _is_iter_category(x_categories) and _is_iter_category(y_categories):
            need_succ_num = math.ceil(len(x_categories) / 2)
            for x_cat in x_categories:
                if need_succ_num == 0:
                    return True
                if x_cat in y_categories:
                    need_succ_num -= 1
        elif _is_iter_category(x_categories) and isinstance(y_categories, Category):
            warnings.warn("It seems that the order of categories is not considered.")

        return False

    def get_success_num(self, concepts=None) -> int:
        sources = self.sources.copy()
        sources = [s for s in sources if self._is_same_categories(concepts, s.concepts)] if concepts else sources
        return len([s for s in sources if self._is_success(s)])

    def get_failure_num(self, concepts=None) -> int:
        sources = self.sources.copy()
        sources = [s for s in sources if self._is_same_categories(concepts, s.concepts)] if concepts else sources
        return len(sources) - self.get_success_num(concepts)

    def dumps(self):
        return {'content': self.content,
                'source_questions': [s.dumps() for s in self.sources]}

    @staticmethod
    def loads(knowledge: dict) -> 'Knowledge':
        if 'content' not in knowledge:
            raise KeyError("The content is not in the dict")

        return Knowledge(content=knowledge['content'],
                         related_info=[KnowledgeSource.loads({**eval(s), **{'knowledge': knowledge['content']}})
                                       for s in knowledge['source_questions']])


class Rationale:  # 修正到只有两个属性
    """
    top-N的结果，rationale和prediction
    """

    def __init__(self, rationale: str, prediction: str, *_, **__):
        self.rationale = rationale.strip()
        self.prediction = self.clean_prediction(prediction)
        self.knowledge_texts = list(self.get_knowledge_texts())  # 这里就不转换knowledge_text为knowledge instance了，感觉没必要
        # 就靠knowledge source存储即可，分析

    @staticmethod
    @KnowledgeExtractionNameSpace.register("Example")
    def extract_knowledge_texts(rationale) -> list[str]:
        pass

    def get_knowledge_texts(self) -> Union[set[str], list[str]]:
        if hasattr(self, 'knowledge_texts') and self.knowledge_texts:
            return self.knowledge_texts

        knowledge_texts = self.extract_knowledge_texts(self.rationale)

        self.knowledge_texts = set(knowledge_texts)

        return self.knowledge_texts

    @classmethod
    @PredictionCleanNameSpace.register("Example")
    def clean_prediction(cls, prediction: str) -> str:
        return prediction

    def update(self, new_rationale: dict[str, str]):
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

    def __init__(self, question: str, gold_label: str, rationales: list[Rationale] = None, *_, **__):
        self.question = question.strip()
        self.gold_label = gold_label.strip()
        self.rationales = rationales if rationales else []

    def update_rationale(self, rationale: Union[dict[str, str], Rationale, list]):
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

    def _merge_attribute(self, attr: str, other_attr_value: Union[list, str]):
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

    def _check_QA(self, other_qa: tuple[str, str]):
        if self.question and self.question.strip() != other_qa[0].strip():
            raise Warning("The question is not the same.")

        if self.gold_label and self.gold_label.strip() != other_qa[1].strip():
            raise Warning("The gold_label is not the same.")

        return True

    @staticmethod
    def _adjust_merge_example_to_rationale(merge_example: dict[str, Union[str, Rationale]]) \
            -> dict[str, Union[str, Rationale, list[Rationale]]]:
        """
        将merge_example中的rationale部分(包括rationale和prediction两个key)转换为Rationale类
        """
        rationale = merge_example.pop('rationale')
        prediction = merge_example.pop('prediction')
        rationale_class = Rationale(rationale=rationale, prediction=prediction)

        merge_example['rationales'] = rationale_class

        return merge_example

    def update(self, example: [str, dict[str, str], 'Example'], args=None):
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

    def parse_response(self, response: str, args=None) -> dict[str, str]:
        """
        这里只会传入A：后面生成的部分
        """
        question_name = self.question.split('\n')[-1].strip()[10:].strip()[
                        :-6].lower()  # this special case only serve for
        # CLUTRR, and it doesn't work for the other benchmarks, so it is not a big deal, and you can adjust it for free.
        # To avoid errors, if question_name is None, we will give it a special name.
        if not question_name:
            question_name = "$NOT_FIND_QUESTION_NAME"

        pred_trigger = args.pred_trigger.lower() if args and hasattr(args, 'pred_trigger') else "the answer is"

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
    def __init__(self, data: list[Example]):
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
    def find(self, key: tuple[str, str]) -> Optional[Example]:
        ...

    @overload
    def find(self, key: tuple[str, str], default) -> Optional[Example]:
        ...

    def find(self, key, *args):
        if key in self._question_label_2_data_instance:
            return self._question_label_2_data_instance[key]
        else:
            if args:
                return args[0]
            else:
                raise KeyError("The question and gold_label pair is not in the DatasetLoader")
