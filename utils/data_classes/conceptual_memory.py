import pickle

from .build_categorize_func import CategoryFunc
from utils.data_classes.knowledge_classes import Knowledge, Category

# ============hack=================
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
encode_model = SentenceTransformer('all-MiniLM-L6-v2')
CACHE_RANK = {}


class KnowledgeMemory(dict):
    def __init__(self,
                 categorize_func: CategoryFunc,  # todo: 确认下plane有无问题，为什么cat的返回值一样
                 *args, **kwargs):
        """
        :param categorize_func: 对example进行分类，可以类比于hash func
        注：因为是memory，我们强制value是list[knowledge]
        """
        super().__init__(*args, **kwargs)
        self.cat_func = categorize_func
        self.CAT_CACHE: dict[str, Category] = {}

    def __setitem__(self, key, value: Knowledge | list[Knowledge]):
        category = self.get_category(key)

        if isinstance(value, Knowledge):
            super().__setitem__(category, [value])  # 限制赋值只能附[knowledge]进来
        else:
            super().__setitem__(category, value)  # 认为value是list[knowledge]
        # if category not in self:
        #     super().__setitem__(category, value)
        # else:
        #     super().__getitem__(category).extend(value)

    def __getitem__(self, key) -> list[Knowledge]:
        category = self.get_category(key)
        return super().__getitem__(category)

    def __contains__(self, key):
        category = self.get_category(key)
        return super().__contains__(category)

    def __getstate__(self):
        """控制pickle序列化时不保存 CAT_CACHE"""
        state = self.__dict__.copy()  # 复制对象的所有属性
        state.pop("CAT_CACHE", None)  # 移除 CAT_CACHE

        return state

    def __setstate__(self, state):
        """控制pickle反序列化时恢复对象"""
        self.__dict__.update(state)

    def get_category(self, key):
        if not hasattr(self, 'CAT_CACHE'):  # fixme: 这个是权宜之计，但有点拖慢这种常用函数的效率了
            self.CAT_CACHE = {}  # 反序列化后，重新初始化 CAT_CACHE

        if key in self.CAT_CACHE:
            category = self.CAT_CACHE[key]
        else:
            if isinstance(key, Category):
                category = key
            else:
                category = Category(category=self.cat_func(key), original_key=key)
                assert not isinstance(self.cat_func(key), Category)
                self.CAT_CACHE[key] = category

        if not isinstance(key, Category):
            category.append_original_key(key)  # XXX: 这里有机会检查下

        return category

    def get(self, key, *args, **kwargs):
        category = self.get_category(key)
        return super().get(category, *args, **kwargs)

    def save(self, path: str = "knowledge_memory.pkl"):
        """
        使用 pickle 将本对象全部保存到指定文件。
        :param path: 保存路径，默认保存为 knowledge_memory.pkl
        """
        # del KnowledgeMemory.CAT_CACHE
        # with open(path, "wb") as f:
        #     pickle.dump(self, f)

        """
                使用 pickle 将本对象全部保存到指定文件。
                :param path: 保存路径，默认保存为 knowledge_memory.pkl
                """
        cat_func_path = path
        items_path = '.'.join(path.split('.')[:-1]) + '.km'  # hack: 之后input改成dir

        with open(cat_func_path, "wb") as f:
            pickle.dump(self.cat_func, f)

        items = {c.dumps(): k for c, k in self.items()}
        with open(items_path, "wb") as f:
            pickle.dump(items, f)

    @staticmethod
    def load(path: str = "knowledge_memory.pkl"):
        """
        从指定的 pickle 文件中加载并返回 KnowledgeMemory 对象。
        :param path: pickle 文件路径
        :return: 反序列化得到的 KnowledgeMemory 实例
        """
        cat_func_path = path
        items_path = '.'.join(path.split('.')[:-1]) + '.km'  # hack: 之后input改成dir

        with open(cat_func_path, "rb") as f:
            cat_func: CategoryFunc = pickle.load(f)

        obj = KnowledgeMemory(categorize_func=cat_func)

        with open(items_path, "rb") as f:
            items: dict[tuple[str,  tuple[str, ...]], list[Knowledge]] = pickle.load(f)

        for c, k in items.items():
            cat = Category.loads(c)
            obj[cat] = k

        return obj


def _rank_func_similarity(key: str | Category, knowledge: Knowledge) -> float:
    if not hasattr(knowledge, 'rank_score'):
        knowledge.rank_score: dict[tuple[str, ...], float] = {}

    if isinstance(key, str):
        key = [key]
    elif isinstance(key, Category):
        key = list(key.original_key)
    else:
        raise TypeError(f"Invalid type for context: {type(key)}")

    key = tuple(key)

    if key in knowledge.rank_score:
        return knowledge.rank_score[key]

    if key in CACHE_RANK:
        key_embedding = CACHE_RANK[key]
    else:
        key_embedding = encode_model.encode(key, convert_to_tensor=True)
        CACHE_RANK[key] = key_embedding

    knowledge_texts = tuple(knowledge.get_source_context())  # 对吧，应该就是一回事儿
    if knowledge_texts in CACHE_RANK:
        knowledge_embeddings = CACHE_RANK[knowledge_texts]
    else:
        knowledge_embeddings = encode_model.encode(knowledge_texts, convert_to_tensor=True)
        CACHE_RANK[knowledge_texts] = knowledge_embeddings

    similarity_scores = util.cos_sim(key_embedding, knowledge_embeddings)
    similarity_scores = sum(similarity_scores[0]) / len(similarity_scores[0])
    if similarity_scores > 0.9:
        print("相似度", key, knowledge_texts)
        print(util.cos_sim(key_embedding, knowledge_embeddings))
        similarity_score = similarity_scores  # 2*  hack: 这里也应该可以作为个超参或者重载
        # 这里就是，有的时候similarity更关键，有的时候confidence更关键。而且要考虑到confidence处于0-1的相对低的区域
        # （比如现在还有为了平滑的坟墓+10）
    elif similarity_scores > 0.6:
        similarity_score = similarity_scores
    else:
        similarity_score = 0
    # similarity_score = 2 * float(similarity_scores)

    return similarity_score


class MultiKnowledgeMemory:
    def __init__(self, categorize_funcs: list[CategoryFunc]):
        """
        :param categorize_funcs: 一个哈希函数列表，每个哈希函数都能把输入映射成一个 int
        """
        assert len(categorize_funcs), "must have at least one categorize_func"
        self.cat_funcs = categorize_funcs
        self.tables: list[KnowledgeMemory] = [
            KnowledgeMemory(categorize_func=categorize_func) for categorize_func in self.cat_funcs
        ]

    def __setitem__(self, key, value: Knowledge | list[Knowledge]):
        if isinstance(key, Category):
            raise TypeError("MultiKnowledgeMemory 不支持直接使用 Category 作为 key")

        if isinstance(key, tuple) and isinstance(key[0], Category):
            assert len(key) == len(self.tables)
            for table, k in zip(self.tables, key):
                table[k] = value
        else:
            for table in self.tables:
                table[key] = value

    def __getitem__(self, key) -> list[Knowledge]:
        """
        对一个 query，分别用所有哈希函数计算hash值，然后到对应的表中取出所有item合并返回。
        """
        if isinstance(key, Category):
            raise TypeError("MultiKnowledgeMemory 不支持直接使用 Category 作为 key")

        values = []
        if isinstance(key, tuple) and isinstance(key[0], Category):
            assert len(key) == len(self.tables)
            for table, k in zip(self.tables, key):  # fixme: 这里好像也不对，这里是能面向从get_category里面获得的key
                # 但是从iter() / keys()里面拿到的key，是每个table自己的。不过我们现在在避免对MultiKnowledgeMemory进行iter
                # 所以我暂时先把iter和keys的实现给注释掉
                values.extend(table.get(k, []))
        else:
            for table in self.tables:
                values.extend(table.get(key, []))

        # if values:
        return values  # todo: 这里的values不能支持append等操作，因为不是原始的list了

        # raise KeyError(f"\"{key}\" not in {MultiKnowledgeMemory}")  # fixme: 这里无法区分空list和真的没有key

    def __contains__(self, key):
        if self._is_category(key):
            if isinstance(key, tuple):
                return any(k in table for k, table in zip(key, self.tables))
        else:
            return any(key in table for table in self.tables)

        raise ValueError(f"\"{key}\" is not a valid key")

    # def __iter__(self):
    #     keys = set()  # XXX: 这个实现可能会有问题，multi的话，每个key怎么定义很难评价
    #     for i, table in enumerate(self.tables):
    #         keys = keys | set(table.keys())  # 考虑改成(i, key)
    #
    #     for key in keys:
    #         yield key

    @staticmethod
    def _is_category(key):
        if isinstance(key, Category):
            return True
        elif isinstance(key, tuple) and isinstance(key[0], Category):
            return True
        return False

    def get_category(self, key) -> tuple[Category, ...]:
        return key if self._is_category(key) else (
            tuple([table.get_category(key) for table in self.tables]))

    def save(self, path: str = "multi_knowledge_memory.pkl"):
        """
        将整个 MultiKnowledgeMemory 对象（以及内部的所有表、哈希函数）保存到文件。

        保存分为两部分：
          - 主文件（meta）：保存 cat_funcs 列表以及内部表的数量
          - 数据文件（items）：保存一个字典，每个 key 为对应表的索引，value 为该表中保存的数据，
            数据中每个键为 Category 对象经 dumps() 后的字符串，确保数据与表的对应关系不混乱。
        """
        meta_path = path
        # 将后缀改为 .mkm 用于保存内部表的数据
        items_path = '.'.join(path.split('.')[:-1]) + '.mkm'

        # 保存元信息：cat_funcs 与内部表的数量
        cat_funcs = self.cat_funcs
        with open(meta_path, "wb") as f:
            pickle.dump(cat_funcs, f)

        # 用一个 dict 保存每个内部表的数据，key 为表的索引
        tables_data = {}
        for idx, table in enumerate(self.tables):
            # 将 table 内部的 key（Category 对象）通过 dumps() 转换为字符串
            table_items = {cat.dumps(): value for cat, value in table.items()}
            tables_data[idx] = table_items

        with open(items_path, "wb") as f:
            pickle.dump(tables_data, f)

    @staticmethod
    def load(path: str = "multi_knowledge_memory.pkl"):
        """
        从文件读入一个 MultiKnowledgeMemory 对象，并返回实例。

        加载步骤：
          1. 从 meta 文件中加载 cat_funcs 和 num_tables 信息。
          2. 从 items 文件中加载保存的各个内部表的数据（以字典形式保存，key 为表的索引）。
          3. 检查保存的表数量是否与 cat_funcs 数量一致，不一致则抛出错误。
          4. 按照索引顺序将数据恢复到各个内部的 KnowledgeMemory 表中。
        """
        meta_path = path
        items_path = '.'.join(path.split('.')[:-1]) + '.mkm'

        # 先加载元信息
        cat_funcs = pickle.load(open(meta_path, "rb"))

        # 加载内部表的数据（字典：表索引 -> 数据）
        with open(items_path, "rb") as f:
            tables_data = pickle.load(f)

        # 根据 cat_funcs 重建 MultiKnowledgeMemory 对象
        obj = MultiKnowledgeMemory(categorize_funcs=cat_funcs)

        # 按照索引恢复每个内部表的内容
        for idx, table in enumerate(obj.tables):
            table_items = tables_data.get(idx)
            if table_items is None:
                raise ValueError(f"缺少索引为 {idx} 的保存数据！")
            for cat_str, value in table_items.items():
                # 将保存的字符串还原为 Category 对象
                cat = Category.loads(cat_str)
                table[cat] = value

        return obj

    # def save(self, folder_path: str):
    #     """
    #     将 MultiKnowledgeMemory 保存到一个文件夹中：
    #       1. 如果文件夹不存在则创建之。
    #       2. 在该文件夹内保存一个 meta 文件（meta.pkl），存放 cat_funcs 及表的数量等元信息。
    #       3. 逐个保存每个内部的 KnowledgeMemory 对象，每个表单独保存，
    #          文件名格式为 "knowledge_memory_i.pkl"，i 表示第 i 个表。
    #          每个 KnowledgeMemory 对象的保存会生成两个文件（主文件和 .km 文件）。
    #     """
    #     # 如果目标文件夹不存在则创建
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     # 保存 meta 信息
    #     meta_path = os.path.join(folder_path, "meta.pkl")
    #     meta = {"cat_funcs": self.cat_funcs, "num_tables": len(self.tables)}
    #     with open(meta_path, "wb") as f:
    #         pickle.dump(meta, f)
    #
    #     # 逐个保存每个内部 KnowledgeMemory 对象
    #     for i, table in enumerate(self.tables):
    #         # 构造每个表的主文件保存路径，如 folder_path/knowledge_memory_0.pkl
    #         table_file = os.path.join(folder_path, f"knowledge_memory_{i}.pkl")
    #         table.save(table_file)
    #
    # @staticmethod
    # def load(folder_path: str) -> 'MultiKnowledgeMemory':
    #     """
    #     从指定的文件夹加载 MultiKnowledgeMemory 对象：
    #       1. 从 meta 文件（meta.pkl）中读取 cat_funcs 及表的数量。
    #       2. 逐个加载文件夹内保存的每个 KnowledgeMemory 对象，
    #          文件名格式为 "knowledge_memory_i.pkl"（对应的 .km 文件会自动加载）。
    #     """
    #     meta_path = os.path.join(folder_path, "meta.pkl")
    #     with open(meta_path, "rb") as f:
    #         meta = pickle.load(f)
    #     cat_funcs = meta["cat_funcs"]
    #     num_tables = meta.get("num_tables", len(cat_funcs))
    #
    #     # 利用 cat_funcs 构建 MultiKnowledgeMemory 对象
    #     obj = MultiKnowledgeMemory(cat_funcs=cat_funcs)
    #
    #     # 按照顺序加载每个内部表的数据
    #     for i in range(num_tables):
    #         table_file = os.path.join(folder_path, f"knowledge_memory_{i}.pkl")
    #         loaded_table = KnowledgeMemory.load(table_file)
    #         obj.tables[i] = loaded_table
    #
    #     return obj

    # def keys(self):
    #     return self.__iter__()

    def get_first_value(self, key: str) -> list[Knowledge]:  # todo: 这边应该是get_sorted_value。然后作为str的key，
        # 能在几个table里找到，也应该是similarity选取的一部分
        values = []
        assert isinstance(key, str), f"key must be a string, now it is {type(key)} {key}"  # fixme: 这里不能支持Category？
        for i, table in enumerate(self.tables):
            vs: list[Knowledge] = table.get(key, [])
            vs = self.filter_and_sort(key, vs)

            # ===========
            # hack: 临时测试用  # fixme: 不能在这边filter，否则可能vs为空
            # import re
            # pattern = r"for (.*)'s(.*), we (have|retrieve)"
            # result = re.findall(pattern, key.lower())
            # if result:
            #     key_words = [result[0][0], result[0][1]]
            #     vs = [k for k in vs if k.content.startswith(key_words[0]+'\'s')
            #           and k.content.split('\'s')[1].startswith(key_words[1]+' ')]
            # else:
            #     warnings.warn(f"key {key} not found in table {i}")
            # ===========
            if vs:
                values.append(vs[0])  # XXX: 这个[0]默认了这个函数仅在sort_knowledge_memory后被调取

        # values = sorted(values,  # todo: sort knowledge memory时候不考虑相似性，还可以说大家是一类的
        #                 # 但multi这里好像值得考虑，哪怕是用满足几个table这个条件。因为multi大家本就不是一类的
        #                 key=lambda x: self.rank_func_knowledge(key, x),  # x.get_confidence(),
        #                 reverse=True)
        values = self.filter_and_sort(key, values)

        return values

    @staticmethod
    def rank_func_knowledge(key: str | Category, knowledge: Knowledge) -> float:
        """
        这个函数面向的输入是KnowledgeMemory的，而非Multi。不过好像只有Multi里遍历KnowledgeMemory的时候需要
        所以暂时放到了Multi里面
        :param key: retrieve前的context，但可能被category过了
        :param knowledge: 待排序的knowledge
        """
        confidence_score = knowledge.get_confidence()
        return confidence_score

        similarity_score = _rank_func_similarity(key, knowledge)

        score = confidence_score + similarity_score
        if hasattr(knowledge, 'rank_score'):
            knowledge.rank_score[key] = score
        else:
            knowledge.rank_score = {key: score}

        return score

    def filter_and_sort(self, key: str, knowledge: list[Knowledge]):
        knowledge = [k for k in knowledge if _rank_func_similarity(key, k) >= 0.7]  # XXX: 1.5是个超参
        # 这里设计有问题，这应当是一个独立的、专做文本检索或排序的，只不过此时的排序要求它综合考虑confidence和similarity或着说relevance
        return sorted(knowledge,
                      key=lambda x: self.rank_func_knowledge(key, x),
                      reverse=True)

    def values(self) -> list[Knowledge]:
        values = []
        for table in self.tables:
            values.extend(table.values())
        return values

    def item(self):
        pass  # fixme: NotImplementedError

    def get(self, key, default=None):
        """Returns the value for key if key is in the dictionary, else calls default if callable,
            otherwise returns default.
        """
        if key in self:
            return self[key]
        return default() if callable(default) else default

    def append_knowledge(self, categories: list[Category], knowledge: Knowledge) -> None:
        for c, table in zip(categories, self.tables):
            if c not in table:
                table[c] = [knowledge]
            else:
                if knowledge not in table[c]:  # 这个复杂度倒是不值得set带来的麻烦
                    table[c].append(knowledge)

    def extend_knowledge(self, categories: list[Category], knowledge: list[Knowledge]) -> None:
        for c, table in zip(categories, self.tables):
            if c not in table:
                table[c] = knowledge  # 尽管KnowledgeMemory有限制机制，但能在外部保证正确性还是在外部保证下
            else:
                if knowledge not in table[c]:
                    table[c].extend(knowledge)
