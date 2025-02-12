from functools import partial
from collections import defaultdict

import numpy as np

from rake_nltk import Rake


class CategoryFunc:
    """
    接近于一个abstract class
    """
    def __init__(self, category_func: callable):
        self.cat_func = category_func  # XXX: 其实Category放在这里，而非KnowledgeMemory的get_category里面也不错的

    def __call__(self, *args, **kwargs):
        return self.cat_func(*args, **kwargs)


# =============tf-idf=======================
def __category_func_tfidf(doc, topn, categorize_model):
    tf_idf_vector = categorize_model.transform([doc])

    coo_matrix = tf_idf_vector.tocoo()
    tuples = zip(coo_matrix.col, coo_matrix.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    sorted_items = [(categorize_model.reversed_vocabulary[idx], score) for
                    idx, score in sorted_items]

    sorted_items = [(word, doc.lower().index(word.lower())) for word, score in sorted_items]
    sorted_items = sorted(sorted_items, key=lambda x: x[1])

    key_concepts = " ".join([word for word, _ in sorted_items[:topn]])

    return key_concepts  # 不建议return tuple，会被识别为generator


def _build_categorize_func_tfidf(categorize_model, **kwargs) -> CategoryFunc:
    """
    :param categorize_model: 对应tf-idf的词表模型
    """
    if not hasattr(categorize_model, "reversed_vocabulary"):  # XXX: 思考下现在的封装是否会干扰reversed_vocab的更新？
        categorize_model.reversed_vocabulary = {v: k for k, v in categorize_model.vocabulary_.items()}

    topn_para = kwargs.get("topn", 2)  # fixme: 这个参数现在没有合适的传入方式，原来KnowledgeBase里的topn的参数我移除了

    return CategoryFunc(category_func=partial(__category_func_tfidf, topn=topn_para, categorize_model=categorize_model))


# =============hyperplane=======================
_ENCODE_CACHE = defaultdict(dict)  # 缓存运行过的encoder.code

import hashlib


def __stop_words_filter(text: str, stop_words: list[str]):  # fixme: 回头看放在哪里
    """此函数留空，暂时没有纳入算法"""
    # tokenizer, 这里用split代替
    word_list = text.split()
    filter_word_list = [word for word in word_list if word not in stop_words]
    if word_list:
        return ' '.join(filter_word_list)

    return "null"

def __hash_vector(inp, encoder, plane: np.ndarray):
    if isinstance(inp, str):
        inp = __stop_words_filter(inp,
                                  stop_words=['sentence', 'phrase', 'retrieve', 'we',
                                              'have', 'answer', 'for'])

    plane_hash = hashlib.sha256(plane.tobytes()).hexdigest()  # XXX: 这里虽然有哈希碰撞的可能性，但后果和概率都不高（plane也就
    # 不超两位数。所以暂时不管它，想校验也很容易，生成plane时候查一下即可
    if inp not in _ENCODE_CACHE[plane_hash]:
        _ENCODE_CACHE[plane_hash][inp] = encoder.encode(inp)
    v = _ENCODE_CACHE[plane_hash][inp]

    # Dot vector with randomly generated planes
    dot_product = np.dot(v.T, plane)  # ( 1 , 768 ) X (768, 11)
    if len(dot_product.shape) == 1:  # a或b为1D-array时，dot不返回矩阵而是返回另一个ND-array的sum，所以需要纠正一下
        dot_product = np.array([dot_product])
    # get the sign of the dot product (1,11) shaped vector
    sign_of_dot_product = np.sign(dot_product)

    h = np.squeeze(sign_of_dot_product >= 0)
    if not h.shape:
        h = np.array([h])

    hash_value = 0

    n_plane = plane.shape[1]
    for i in range(n_plane):
        # increment the hash value by 2^i * h_i
        hash_value += np.power(2, i) * h[i]

    hash_value = int(hash_value)

    return hash_value


def _build_categorize_func_hyperplane(categorize_model, **_) -> list[CategoryFunc]:
    """
    :param categorize_model: 给定的词向量模型和超平面
    """
    cat_encoder, planes = categorize_model

    return [CategoryFunc(category_func=partial(__hash_vector, encoder=cat_encoder, plane=p)) for p in planes]


# ==========rake===========
def __extract_keywords_rake(text: str, rake_obj: Rake, topn: int = 4) -> tuple[str, ...]:
    rake_obj.extract_keywords_from_text(text)
    keywords: list[tuple[float, str]] = rake_obj.get_ranked_phrases_with_scores()
    keywords: tuple[str, ...] = tuple(w[1] for w in keywords[:topn])

    return keywords


def _build_categorize_func_rake_nltk(categorize_model, **kwargs) -> CategoryFunc:
    r: Rake = categorize_model
    topn_para = kwargs.get("topn", 2)

    return CategoryFunc(category_func=partial(__extract_keywords_rake, rake_obj=r, topn=topn_para))


funcs_dict = {'tfidf': _build_categorize_func_tfidf,
              'hyperplane': _build_categorize_func_hyperplane,
              'rake_nltk': _build_categorize_func_rake_nltk}


def build_categorize_func(categorize_model, func_name='hyperplane', **kwargs) -> CategoryFunc | list[CategoryFunc]:
    func = funcs_dict.get(func_name, _build_categorize_func_tfidf)

    return func(categorize_model, **kwargs)

# https://towardsdatascience.com/locality-sensitive-hashing-in-nlp-1fb3d4a7ba9f
# 理想情况下，验一下输出要是hashable的
