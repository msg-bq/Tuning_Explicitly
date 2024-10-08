import re
import string

from utils.ExtraNameSpace import PredictionCleanNameSpace


@PredictionCleanNameSpace.register("Default")
def clean_prediction(self, prediction: str) -> str:
    return prediction


@PredictionCleanNameSpace.register("CLUTRR")
def clean_prediction(self, prediction: str) -> str:
    """
    从Answer中提出特定数据集的答案
    这里传入的pred已经是最后一个The answer is后面的部分了
    """
    if prediction == "":
        return prediction

    pred_words = prediction.split()
    if len(pred_words) == 1:
        if pred_words[0][-1] in string.punctuation:
            return pred_words[0][:-1]

        return pred_words[0].strip()

    tags = ['<Begin>', '</End>', '<rule>', '<retrieved_rule>', '<new_rule>']
    for tag in tags:
        if tag in prediction:
            return clean_prediction(self, prediction.split(tag)[0])

    if pred_words[-1][-1] in string.punctuation:
        return pred_words[-1][:-1]

    return pred_words[-1].strip()

@PredictionCleanNameSpace.register("SST2")
def clean_prediction(self, prediction: str) -> str:
    prediction = prediction.strip().lower()
    if prediction == "":
        return prediction

    result = ""

    pattern1 = "the sentiment of the above review is (.*)"
    pattern2 = "the sentiment is (.*)"
    pattern3 = "the sentiment in this sentence is (.*)"

    pattern_list = [pattern1, pattern2, pattern3]

    for pattern in pattern_list:
        match = re.match(pattern, prediction)
        if match:
            result = match.group(1)
            result = result.split()[0]
            break

    if not result and 'negative' in prediction and 'positive' not in prediction:
        return "negative"
    elif not result and 'positive' in prediction and 'negative' not in prediction:
        return "positive"

    if not result:
        words = prediction.split()
        if len(words) == 1:
            result = prediction
        else:
            if words[-1].lower().startswith('sentiment'):
                result = words[-2]
            else:
                result = words[-1]

    if result[-1] in string.punctuation:
        return result[:-1]

    return result.strip()


@PredictionCleanNameSpace.register("LANG_8")
def clean_prediction(self, prediction: str) -> str:
    """
    ORIGINAL_SENTENCE这个特殊的返回值表明答案是正确的
    """
    original_prediction = prediction
    prediction = prediction.strip().lower()

    no_error_trigger = ["no grammar errors", "no grammatical errors"]
    if any(trigger in prediction for trigger in no_error_trigger):
        return "ORIGINAL_SENTENCE"

    def remove_quotation_marks(text):
        print("抽取出来的文本是：", text)
        """去除首尾的引号"""
        text = text.strip()
        if not text:
            return text

        if text[0] == ":":
            text = text[1:].strip()

        if not text:
            return text

        if text[0] == '"' and text[-1] == '"':
            return text[1:-1]
        elif text[0] == "“" and text[-1] == "”":
            return text[1:-1]
        elif text[0] == "'" and text[-1] == "'":
            return text[1:-1]
        return text

    pattern1 = "the revised sentence can be(.*)"
    pattern2 = "the revised sentence could be \"(.*)\""
    pattern3 = "the revised sentence is(.*)"
    pattern4 = "the revised sentence would be(.*)"
    pattern5 = "it could be revised as follows:(.*)"
    pattern6 = "however, a revised version of the sentence could be(.*)"
    pattern7 = "revised sentence:(.*)"
    pattern8 = "the revised sentence should be(.*)"
    pattern9 = "the correct sentence is(.*)" # 最好是用""把每个(.*)包起来，不过有少数确实没有引号
    pattern10 = "The revised sentence with corrected grammar would be:".lower()
    pattern11 = "The revised sentence is \"(.*)\"".lower()
    pattern12 = "therefore, the revised sentence is \"(.*)\""

    pattern_num = 12
    pattern_list = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8,
                    pattern9, pattern10, pattern11, pattern12]

    for pattern in pattern_list:
        # 以\n或结束符结束
        pattern += "(\n|$)"
        match = re.search(pattern, prediction)
        if match:
            result = remove_quotation_marks(match.group(1))
            if result:
                idx = original_prediction.lower().index(result)
                return original_prediction[idx:idx + len(result)]

    return original_prediction