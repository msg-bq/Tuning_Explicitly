import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 下载停用词（如果尚未下载）
nltk.download('stopwords')
nltk.download('punkt')


def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))  # 获取英文停用词列表
    words = word_tokenize(sentence)  # 分词
    filtered_sentence = [word for word in words if word.lower() not in stop_words]  # 过滤停用词
    return ' '.join(filtered_sentence)

# 示例
# sentence = "This is an example showing how to remove stop words from a sentence."
# filtered_sentence = remove_stopwords(sentence)
# print(filtered_sentence)


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
