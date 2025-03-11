import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import yaml

# 获取当前文件的路径
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
# 获取上上级目录的路径
parent_directory = current_file_path.parent.parent
# 将上上级目录添加到模块搜索路径中
sys.path.append(str(parent_directory))

from utils import LLM, NameSpace
from utils.data_classes import Rationale
from utils.read_datasets import read_datasets

import utils.clean_prediction_func
import utils.read_funcs


def args_parse():
    parser = argparse.ArgumentParser(description="Rule-Finetune")

    parser.add_argument("--dataset", type=str, default="CLUTRR",
                        choices=["default", "CLUTRR", "SST2", "LANG_8", "SALAD", "SignalP"],  # default
                        # 包含一个通用的默认格式输入，暂时先不写
                        help="dataset used for experiment, should involve train, test at least")

    parser.add_argument("--llm_model", type=str,
                        # choices=["davinci", "gpt-3.5-turbo", "gpt-3.5-turbo-ca", "gpt-3.5-turbo-0613",
                        #          "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-4-turbo-2024-04-09",
                        #          "gpt-4o-ca", "glm-4-air", "claude-3-5-haiku-20241022",
                        #          "gemini-1.5-flash-latest"],
                        default="gpt-3.5-turbo-ca", help="language model used for experiment")

    parser.add_argument("--data_dir", type=str, default=None,
                        help="data dir used for experiment")

    parser.add_argument("--prompt_type", type=str, default='CLUTRR_test_prompt',
                        help="data dir used for experiment")

    parser.add_argument("--save_file", type=str, default=None)

    args = parser.parse_args()

    if not args.data_dir:
        args.data_dir = f"data/{args.dataset}"

    if not args.save_file:
        num_suffix = 0
        while os.path.exists(os.path.join("baseline_exp", args.dataset, args.prompt_type, str(num_suffix))):
            num_suffix += 1
        args.save_file = os.path.join("baseline_exp", args.dataset, args.prompt_type, str(num_suffix))

    NameSpace._args = args

    return args


def eval_step(args, llm: LLM, example, test_prompt: None | str | Callable[[str], str]):
    if isinstance(test_prompt, Callable):
        prompt = test_prompt(example.question)
    elif isinstance(test_prompt, str):
        prompt = test_prompt + "\n\n" + example.question
    elif test_prompt is None:
        prompt = example.question
    else:
        raise TypeError(f"test_prompt type {type(test_prompt)} not supported")

    response = llm.generate_single(input_text=prompt, model=args.llm_model)
    rationale = example.parse_response(response, args)
    prediction = Rationale.clean_prediction(rationale['prediction'])

    return prediction, example


def _get_prompt(args):
    if args.prompt_type == 'zero':
        return None

    if args.dataset == 'SALAD':
        from baseline_prompt.SALAD_baseline import salad_prompt_input
        return salad_prompt_input  # hack: 超参

    file_path = os.path.join('prompt_utils/', f'{args.dataset}.yaml')
    return yaml.load(open(file_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)[args.prompt_type]


def main():
    args = args_parse()
    _, _, test_dataset = read_datasets(args)
    llm = LLM(generate_func_or_name=args.llm_model)
    save_file = args.save_file
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_file, 'w'):
        pass

    test_prompt = _get_prompt(args)

    correct_cnt = 0

    with ThreadPoolExecutor(max_workers=200) as executor:  # hack: 超参
        futures = [executor.submit(eval_step, args, llm, example, test_prompt) for example in test_dataset]

        for future in futures:
            if args.dataset in ['CLUTRR', 'SALAD', 'SignalP']:
                prediction, example = future.result()
                prediction = prediction.replace("-in-law", "").replace("step-", "").replace("step", "")
                gold_label = example.gold_label.replace("-in-law", "").replace("step-", "").replace("step", "")

                with open(save_file, 'a', encoding="utf8") as f:
                    save_data = {'question': example.question,
                                 'prediction': prediction,
                                 'gold_label': gold_label}
                    f.write(json.dumps(save_data) + '\n')

                if prediction.lower() == gold_label.lower():
                    correct_cnt += 1
                else:
                    print("prediction， gold", prediction, gold_label)

    accuracy = correct_cnt / len(test_dataset)
    print(accuracy)


if __name__ == '__main__':
    main()
