import json
import warnings
from copy import deepcopy
from typing import Literal

from .build_categorize_func import build_categorize_func, CategoryFunc
from .build_categorize_model import build_categorize_model
from .conceptual_memory import Category, KnowledgeMemory, MultiKnowledgeMemory
from .knowledge_classes import Knowledge, Rationale


_KNOWLEDGE_MEMORY_TYPE = KnowledgeMemory | MultiKnowledgeMemory


class KnowledgeBase:
    def __init__(self, mode: str = None, build_conceptual_memory_method: str = 'tfidf'):
        self._content_to_instance: dict[str, Knowledge] = dict()
        self._knowledge_memory: _KNOWLEDGE_MEMORY_TYPE | None = None
        # fixme: For grandson\'s brother, we have "  这个例子对应的context和knowledge都是错的，检查一下是multi这个class的问题还是
        # 确实没有
        self._inference_knowledge_memory: _KNOWLEDGE_MEMORY_TYPE | None = None
        self.mode: Literal['train', 'eval', None] = None
        self.build_conceptual_memory_method: str = build_conceptual_memory_method  # 用哪个算法构建conceptual memory

    def __len__(self):
        return len(self._content_to_instance)

    def get_knowledge_by_content(self, content: str) -> Knowledge:
        instance = self._content_to_instance.get(content)

        if instance is None:
            raise KeyError(f"{content} not in knowledge base")

        return instance

    @staticmethod
    def _split_knowledge_text(knowledge: str) -> list[str]:
        """
        如果是str，默认是以\n分隔的
        """
        if isinstance(knowledge, str):
            return [k.strip() for k in knowledge.split('\n') if k.strip()]

        raise TypeError("knowledge in '_split_knowledge_text' func should be str")

    def _find_knowledge_instance(self, knowledge: str | list[str | Knowledge]) -> set[Knowledge]:
        knowledge = self._split_knowledge_text(knowledge) if isinstance(knowledge, str) else knowledge
        knowledge = [k.content if isinstance(k, Knowledge) else k for k in knowledge]  # 不假设"Knowledge类型=在KB里"

        knowledge_instances = set([self._content_to_instance[k] if k in self._content_to_instance
                                   else self._add_knowledge(k) for k in knowledge])

        return knowledge_instances

    def update_knowledge(self, knowledge_texts: str | list[str] | list[Knowledge],
                         question: str, rationale: Rationale, score: float) -> list[Knowledge]:
        """
        需要字符串匹配，找到就返回，找不到就创建+返回
        :param knowledge_texts: 答题时从rationale中抽取的规则
        :param rationale:
        :param score:
        :param question: 问题
        本函数现在还不支持batch，连带着后面的find_instance之类的都不支持
        """
        if self.mode == 'eval':
            raise ValueError("Shouldn't update knowledge in eval mode")

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

    def _add_knowledge(self, knowledge: list[str] | str) -> Knowledge | list[Knowledge]:
        """
        这里需要一个添加knowledge的函数，包括将字符串转为str+查重+添加
        这个函数只add，不检查是否存在
        输入输出的type保持一致，均为iteration或普通数值类型
        """
        if isinstance(knowledge, str):
            return self.__add_knowledge(knowledge)
        else:
            return [self.__add_knowledge(k) for k in knowledge]

    # def _append_knowledge_source(self, knowledge_source: Union[KnowledgeSource, list[KnowledgeSource]]):
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
        self._save_knowledge(knowledge_memory_path)  # fixme: 这一行是存_content_to_instance用的，但主要也还是给人看的
        if vectorizer_path:
            self._knowledge_memory.save(path=vectorizer_path)  # fixme: 先用vectorizer的路径

    def load_knowledge_memory(self, knowledge_memory_path: str, vectorizer_path: str = None):
        self._load_knowledge(knowledge_memory_path)

        if vectorizer_path:
            # _type = MultiKnowledgeMemory if os.path.basename(vectorizer_path).startswith('multi') \
            #     else KnowledgeMemory
            _type = KnowledgeMemory  # hack: 这里先固定下
            self._knowledge_memory = _type.load(path=vectorizer_path)  # hack: 这里面选type目前没有任何影响，两个
            # class的load函数一样，得到的结果自然也是一样的

            learned_info = self._get_memory_learned_info()
            self._update_knowledge_memories(learned_info=learned_info)

    def _update_knowledge_memory(self,
                                 concepts_chain: list[Category] | list[tuple[Category, ...]],
                                 knowledge_chain: Knowledge | list[Knowledge]):
        """
        这个函数暂时封装的意义不大，只是看之后有无其他函数也需要调用
        """
        if isinstance(knowledge_chain, Knowledge):
            knowledge_chain = [knowledge_chain] * len(concepts_chain)
        elif isinstance(knowledge_chain, list):
            raise NotImplementedError

        assert len(concepts_chain) == len(knowledge_chain)

        for concept, knowledge in zip(concepts_chain, knowledge_chain):
            if isinstance(self._knowledge_memory, KnowledgeMemory):
                if concept not in self._knowledge_memory:
                    self._knowledge_memory[concept] = [knowledge]
                else:
                    if knowledge not in self._knowledge_memory[concept]:  # 这个复杂度倒是不值得set带来的麻烦
                        self._knowledge_memory[concept].append(knowledge)
            elif isinstance(self._knowledge_memory, MultiKnowledgeMemory):
                # hack: 这里面有个问题是，要求输入的key必须是个get_category出来的值，否则就会有问题。但这个对使用者会提出额外的困难
                assert isinstance(concept, tuple) and isinstance(concept[0], Category)
                self._knowledge_memory.append_knowledge(concept, knowledge)
            else:
                raise TypeError(f"knowledge_memory is wrong type: {type(self._knowledge_memory)}")

    def get_knowledge_memory(self, inner_call: bool = False):
        """
        :param inner_call: 表明当前调用是由KnowledgeBase内部函数调取，而非外部使用。内部调用不需要触发warning
        :return: 当前的knowledge memory
        """
        if self.mode == 'eval' and not inner_call:
            warnings.warn("In eval mode, the knowledge memory is not updated. "
                          "And we suggest call get_inference_knowledge_memory in eval mode.")
        return self._knowledge_memory

    def _get_memory_learned_info(self) -> list[Knowledge]:
        """
        为memorization准备相关文本，比如对于base版本的knowledge base，这里对应所有knowledge即可（related_context内置进去了）
        对于加了并查集去查的knowledge base，取每个类的root节点的knowledge instance

        一般来说主要传递的是knowledge+能获取对应context、score信息的入参（所以其他地方修改实现的话，这里也简单调整下
        """
        return list(self._content_to_instance.values())

    @staticmethod
    def __get_topk_context(knowledge_sources: list[str], top_percent: float = 0.5) -> list[str]:
        """
        :param knowledge_sources:
        :param top_percent: 选取数量占比前百分之top_percent的source
        :return: 选取的topk context
        """
        source_count = dict()
        for source in knowledge_sources:
            source_count[source] = source_count.get(source, 0) + 1

        source_count = sorted(source_count.items(), key=lambda x: x[1], reverse=True)
        chosen_sources = []
        threshold = len(knowledge_sources) * top_percent
        for source, count in source_count:
            chosen_sources.append(source)
            threshold -= count
            if threshold <= 0:
                break

        return chosen_sources

    def _update_knowledge_memories(self, learned_info: list[Knowledge]):
        for knowledge in learned_info:
            # todo: 这个数据结构有点奇怪，应该在一个更优雅的地方把sources改了，或者就在上面
            # fixme: contexts也是for knowledge.sources，后面应该改了这四行
            chosen_categories = []
            top_contexts = self.__get_topk_context(knowledge_sources=knowledge.get_source_context(),
                                                   top_percent=1)  # hack: 0.5超参
            # 这里确实意味着相关文本，或者相关文本的数量也作为一个参数？这个是不能替代similarity，或者是similarity的加权和的权？
            # 我目前感觉好像是一回事儿

            for source in knowledge.sources:
                context: str = source.related_context
                category: Category | tuple[Category, ...] = self._knowledge_memory.get_category(key=context)
                source.concepts = category  # fixme: hash func更新的时候需要有办法更新，
                # 不过好像可以用dct[key]直接更新下面所有的value
                # fixme: 这里既然每个source.concepts给了一个list[Category]，所以要注意后面要能用到这个list版本的，而不是直接match
                # 比如注意刚才的50%
                if context in top_contexts:
                    chosen_categories.append(category)

            self._update_knowledge_memory(concepts_chain=chosen_categories,
                                          knowledge_chain=knowledge)

    def _extract_doc_list_from_learned_info(self) -> tuple[list[str], list[Knowledge]]:
        learned_info: list[Knowledge] = self._get_memory_learned_info()

        doc_list = []
        for knowledge in learned_info:
            doc_list.extend(knowledge.get_source_context())

        return doc_list, learned_info

    @staticmethod
    def stop_words_filter(doc_list: list[str], stop_words: list[str]):
        """此函数留空，暂时没有纳入算法"""
        # tokenizer, 这里用split代替
        filter_doc_list = []
        for doc in doc_list:
            word_list = doc.split()
            filter_word_list = [word for word in word_list if word not in stop_words]
            if word_list:
                filter_doc_list.append(' '.join(filter_word_list))

        return filter_doc_list

    def _build_conceptual_memory(self, doc_list: list[str], fn_name: str = 'tfidf'):
        categorize_model = build_categorize_model(doc_list=doc_list, func_name=fn_name)
        categorize_funcs = build_categorize_func(categorize_model=categorize_model, func_name=fn_name)

        if isinstance(categorize_funcs, CategoryFunc):
            self._knowledge_memory = KnowledgeMemory(categorize_funcs)
        elif isinstance(categorize_funcs, list) and isinstance(categorize_funcs[0], CategoryFunc):
            self._knowledge_memory = MultiKnowledgeMemory(categorize_funcs)
        else:
            raise TypeError("categorize_funcs must be CategoryFunc or list[CategoryFunc]."
                            "Now, it is {}".format(type(categorize_funcs)))

    def build_conceptual_memory(self):
        """
        实现功能的函数是_build_conceptual_memory，换记忆方法时需要重载。而此函数很多是为了日志、改分等对齐而写的
        """
        if self.mode == 'eval':
            raise ValueError("Should not build conceptual_memory in eval mode")

        doc_list, learned_info = self._extract_doc_list_from_learned_info()

        if not doc_list:
            return

        # 只是实现在这里，但本次投稿、训练时候先不加
        # doc_list = self.stop_words_filter(doc_list=doc_list,
        #                                   stop_words=['sentence', 'phrase', 'retrieve', 'we',
        #                                               'have', 'answer'])
        # 对于tf-idf，从源头删除就够了，因为inference时，train不存在的会被丢弃

        self._build_conceptual_memory(doc_list=doc_list, fn_name=self.build_conceptual_memory_method)

        self._update_knowledge_memories(learned_info=learned_info)

    @staticmethod
    def _sort_knowledge_memory(knowledge_memory: _KNOWLEDGE_MEMORY_TYPE) -> _KNOWLEDGE_MEMORY_TYPE:
        """
        :return: 对knowledge memory做排序，用于推理阶段
        """
        if isinstance(knowledge_memory, KnowledgeMemory):
            for k in knowledge_memory:
                knowledge_memory[k] = sorted(knowledge_memory[k],
                                             key=lambda x: MultiKnowledgeMemory.rank_func_knowledge(k, x),
                                             # x.get_confidence(),
                                             reverse=True)
            return knowledge_memory
        elif isinstance(knowledge_memory, MultiKnowledgeMemory):
            for table in knowledge_memory.tables:
                for k in table:
                    table[k] = sorted(table[k],  # todo: 没考虑过相似性
                                      key=lambda x: MultiKnowledgeMemory.rank_func_knowledge(k, x),
                                      # x.get_confidence(),
                                      reverse=True)
            return knowledge_memory
        else:
            raise TypeError("knowledge_memory must be KnowledgeMemory or MultiKnowledgeMemory")

    @staticmethod
    def _filter_knowledge_memory(knowledge_memory: _KNOWLEDGE_MEMORY_TYPE) -> _KNOWLEDGE_MEMORY_TYPE:
        """
        推理时过滤掉质量过低的knowledge
        """
        threshold_num = 1

        if isinstance(knowledge_memory, KnowledgeMemory):
            filtered_knowledge_memory = KnowledgeMemory(knowledge_memory.cat_func)
            for k in knowledge_memory:
                filtered_knowledge = [v for v in knowledge_memory[k] if v.get_success_num(k) > v.get_failure_num(k)
                                      / threshold_num]
                if filtered_knowledge:
                    filtered_knowledge_memory[k] = filtered_knowledge
            return filtered_knowledge_memory

        elif isinstance(knowledge_memory, MultiKnowledgeMemory):
            filtered_multi_knowledge_memory = MultiKnowledgeMemory(knowledge_memory.cat_funcs)
            for i, table in enumerate(knowledge_memory.tables):
                for k in table:
                    filtered_knowledge = [v for v in table[k] if v.get_success_num(k) > v.get_failure_num(k)
                                          / threshold_num]
                    if filtered_knowledge:
                        filtered_multi_knowledge_memory.tables[i][k] = filtered_knowledge
            return filtered_multi_knowledge_memory

        else:
            raise TypeError("knowledge_memory must be KnowledgeMemory or MultiKnowledgeMemory")

    def get_inference_knowledge_memory(self) -> _KNOWLEDGE_MEMORY_TYPE:
        if self.mode == 'eval' and self._inference_knowledge_memory:
            return self._inference_knowledge_memory

        if self.mode == 'train':
            warnings.warn("This function costs many resources to refine and sort the knowledge in the memory."
                          "We suggest call get_knowledge_memory in the train mode.")

        self._inference_knowledge_memory = (
            self._sort_knowledge_memory(self._filter_knowledge_memory(self.get_knowledge_memory(inner_call=True))))

        return self._inference_knowledge_memory

    def set_knowledge_memory(self, knowledge_memory: dict[str, list[Knowledge]]):  # XXX: 这个函数现在是不没用了
        self._knowledge_memory = knowledge_memory

    def __deepcopy__(self, memo):  # XXX: 修改memory格式后，这个其实不一定可靠了，比如memory可能也要维护一个deepcopy
        # 但现在考虑的是似乎不需要deepcopy了
        """
        创建一个KB的副本
        """
        cls = self.__class__
        new_kb = cls.__new__(cls)
        memo[id(self)] = new_kb
        for k, v in self.__dict__.items():
            setattr(new_kb, k, deepcopy(v, memo))

        return new_kb

    def eval(self):
        self.mode = 'eval'
        self._inference_knowledge_memory = self.get_inference_knowledge_memory()  # todo: 需要一个train,并联动起来

    def train(self):
        self.mode = 'train'
        self._inference_knowledge_memory = None
