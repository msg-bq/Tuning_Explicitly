import math
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from rake_nltk import Rake


def _build_categorize_model_tfidf(doc_list: list[str]):
    """
    这里算法可选，能达成为句子提供分类名词的需求即可，我们目前采取的策略是tf idf获取keywords作为概念
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit_transform(doc_list)

    return vectorizer


def _build_categorize_model_hyperplane(doc_list: list[str]):
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    if isinstance(doc_list, str):
        doc_list = [doc_list]

    # complaint_embeddings = encoder.encode('test')
    search_space_len = len(doc_list)
    # embedding_dims = complaint_embeddings.shape[0]
    embedding_dims = 384  # hack: 现在卡死了这个模型
    warnings.warn("The dims of the embedding is not known, please check the embedding dims")

    n_buckets = max(search_space_len / 10, 1)

    # Generate 11 planes randomly. This gives us a embedding_dims X n_planes dimensional matrix
    n_planes = max(math.ceil(math.log2(n_buckets)), 1)
    n_planes = 50
    n_repeats = max(math.ceil(search_space_len / n_planes / 10), 1)
    n_repeats = 2  # todo: 提成参数
    np.random.seed(42)
    planes_l = tuple([np.random.normal(size=(embedding_dims, n_planes)) for _ in range(n_repeats)])

    return encoder, planes_l


def _build_categorize_model_rake_nltk(doc_list):
    nltk.download('punkt_tab')
    nltk.download('stopwords')

    empty_stopwords = {"$^@^#&@&#*@"}
    folio_nl_stopwords = {'premise', 'hypothesis', 'hypo', 'premises', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                          'that', 'we', 'have', 'retrieve', 'should', 'the', 'for'}
    r = Rake(stopwords=folio_nl_stopwords)  # Rake的停用词不能设置为空（空的时候默认使用nltk的，所以需要给一个特殊值）

    return r


funcs_dict = {'tfidf': _build_categorize_model_tfidf,
              'hyperplane': _build_categorize_model_hyperplane,
              'rake_nltk': _build_categorize_model_rake_nltk}


def build_categorize_model(doc_list: list[str], func_name='hyperplane'):
    return funcs_dict.get(func_name, _build_categorize_model_tfidf)(doc_list)
