from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Optional, List

from utils.llm_models import *


def generate_func_mapping(model: str) -> callable:
    openai_prefix = ["davinci", "gpt"]
    openchat_prefix = ["openchat"]
    glm_prefix = ["glm"]

    prefix_func_mapping = {'gpt': (openai_prefix, call_openai),
                           'openchat': (openchat_prefix, call_openchat),
                           # 'glm': (glm_prefix, call_glm)
                           }

    for _, (prefixes, func) in prefix_func_mapping.items():
        for prefix in prefixes:
            if model.startswith(prefix):
                return func

    raise ValueError(f"model {model} is not supported.")


class LLM:
    def __init__(self, generate_func: Optional[callable] = None, max_workers: int = 5):
        self.generate_func = generate_func
        self.max_workers = max_workers

    def generate_single(self, input_text: str, **kwargs) -> str:
        if self.generate_func is None:
            raise ValueError("generate_func is None.")

        return self.generate_func(input_text, **kwargs)

    def generate_single_parallel(self, input_text: str, try_times: int = 3, **kwargs) -> List[str]:
        results = []

        workers = min(try_times, self.max_workers)
        generate_single_partial = partial(self.generate_single, **kwargs)

        with ThreadPoolExecutor(max_workers=workers) as self.executor:
            sub_response = [self.executor.submit(generate_single_partial, input_text) for _ in range(try_times)]

            for r in as_completed(sub_response):
                if not r.result():
                    continue

                response = r.result()
                if isinstance(response, str):
                    results.append(response)
                elif isinstance(response, list):
                    new_response = [r for r in response if r]
                    results.extend(new_response)
                else:
                    raise TypeError("Incorrect type.")

        return results

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate_single(prompt, **kwargs) for prompt in prompts]

    def eval(self):
        """
        原计划打算把循环和topN放在这里，但感觉没必要，放在eval_step里就好了
        """
        pass


if __name__ == '__main__':
    pass
    # llm = LLM(call_openai)
    # llm = LLM(call_openchat)
    # print(llm.generate_single('hello world'))
    # print(llm.generate_single_parallel("""write a poem about Alice's "fake" world""", topN=5))
