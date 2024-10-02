import concurrent
from concurrent.futures import as_completed
from random import random
from typing import Optional, List, Union
import time
import tiktoken
import requests

url = "http://localhost:5027/v1/chat/completions"
headers = {"Content-Type": "application/json"}

encoding = tiktoken.get_encoding("cl100k_base")
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens

def cnt_tokens(message):
    '''
    统计message的token数
    '''
    if isinstance(message, list):
        cnt = 0
        for m in message:
            cnt += num_tokens_from_string(m['content'])
    elif isinstance(message, str):
        cnt = num_tokens_from_string(message)
    else:
        raise TypeError("message should be list or str")

    return cnt


def call_openchat(input_text: Union[List[str], str], model="openchat_3.5", is_gpt3=False, **kwargs) \
        -> Union[str, List[str]]:
    """
    lbq
    List[str] GPT存在历史，str 不存在历史
    """
    if 'topN' in kwargs:
        kwargs['n'] = kwargs.pop('topN')

    max_supported_tokens = 3000 # ≈3:4

    if isinstance(input_text, str):
        prompt = [{"role": "user", "content": " ".join(input_text.split(' ')[:max_supported_tokens])}]
    else:
        prompt = input_text


    if 'max_length' in kwargs:
        max_length = kwargs['max_length']
        max_length = min(max_length, max_supported_tokens - cnt_tokens(prompt))
        if max_length < 0:
            raise ValueError("max_length is too small")

    try_call = 10
    while try_call:
        try_call -= 1
        start_time = time.time()
        try:
            param = {'model': model,
                     'messages': prompt}
            param.update(kwargs)
            response = requests.post(url, json=param, headers=headers)
            completion = eval(response.text)
            
            # print("time:", time.time() - start_time)

            if len(completion['choices']) == 1:
                return completion['choices'][0]['message']['content'].strip()
            else:
                return [c['message']['content'].strip() for c in completion['choices']]

        except Exception as e:
            time.sleep(20 + 10 * random())
            print("openchat有错")
            print(e)

    if 'n' in kwargs and kwargs['n'] >= 1:
        return []
    else:
        return ''

if __name__ == '__main__':
    pass