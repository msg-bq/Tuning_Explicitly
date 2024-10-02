import concurrent
from concurrent.futures import as_completed
from random import random
from typing import Optional, List, Union
import time
from zhipuai import ZhipuAI
import tiktoken

key_list = ["bb0cacadd0f34f6a8ef0e79fe8b6859a.waTlNiSctvzNk7tw",
            "493b431d70f3a3cbeb422fb635d2fd9d.ESOID6BTundcrGPu"]
key_choose = 0

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


def call_glm(input_text: Union[List[str], str], model="glm-3-turbo", **kwargs) \
        -> Union[str, List[str]]:
    """
    lbq
    List[str] GPT存在历史，str 不存在历史
    """

    if 'topN' in kwargs:
        kwargs['n'] = kwargs.pop('topN')

    max_supported_tokens = 6000

    if isinstance(input_text, str):
        prompt = [{"role": "user", "content": " ".join(input_text.split(' ')[:max_supported_tokens])}]
    else:
        prompt = input_text


    global key_choose
    client = ZhipuAI(api_key=key_list[key_choose]) # 填写您自己的APIKey

    try_call = 10
    while try_call:
        try_call -= 1
        start_time = time.time()
        key_choose = (key_choose + 1) % len(key_list)
        try:
            result = client.chat.completions.create(
                        model=model,
                        messages=prompt, **kwargs)
                    # print("time:", time.time() - start_time)
            if len(result.choices) == 1:
                return result.choices[0].message.content.strip()
            else:
                return [c.message.content.strip() for c in result.choices]

            # else:
            #     completion = openai.ChatCompletion.create(
            #         model=model,
            #         messages=prompt,
            #         **kwargs
            #     )
            #     # print("time:", time.time() - start_time)

            #     if len(completion.choices) == 1:
            #         return completion.choices[0].message['content'].strip()
            #     else:
            #         return [c.message['content'].strip() for c in completion.choices]

        except Exception as e:
            print("sleep")
            time.sleep(20 + 10 * random())


    # if 'n' in kwargs and kwargs['n'] > 1:
    #     return []
    # else:
    #     return ""

if __name__ == '__main__':
    prompt = \
'''Context: Lisa went shopping with her son Joe and her brother Michael.\nQuestion: Joe is Michael's what?
Answer: Let's think step by step. If you use some rules in the reasoning process, please write them in "<rule>xxx<rule>" format individually before you draw every conclusion.'''

#     prompt = \
#     """Context: Florence and her husband Norman went to go ice skating with their daughter Marilyn. Marilyn's sister Janet could not go because she has a broken leg. Kecia went to the store with her sister Florence Chris loved going to the store with his mom Florence. She always bought him snacks Chris likes to visit his sister. Her name is Janet.
# Question: Kecia is Norman's what?
# Answer: Let's think step byx step. If you use some rules in the reasoning process, please write them with "<rule>xxx<rule>" format individually."""

    # print(prompt)
    # num_worker = 5
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
    #     sub_response = [executor.submit(call_openai, prompt, model="gpt-4-1106-preview") for _ in range(num_worker)]
    #
    #     for r in as_completed(sub_response):
    #         print(r.result())
    print(call_glm(prompt, model="glm-3-turbo"))