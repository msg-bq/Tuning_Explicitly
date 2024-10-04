import os.path

from Trainer.KnowledgeTrainer import Trainer
from utils.data import KnowledgeBase
from utils.llm import LLM, generate_func_mapping
from utils.read_datasets import read_datasets, read_rationales
import argparse
from utils.ExtraNameSpace import NameSpace

from prompt import prompt_dict

from logger import logger


def args_parse():
    parser = argparse.ArgumentParser(description="Rule-Finetune")

    parser.add_argument("--dataset", type=str, default="CLUTRR",
                        choices=["default", "CLUTRR", "SST2", "LANG_8"],  # default包含一个通用的默认格式输入，暂时先不写
                        help="dataset used for experiment, should involve train, test at least")

    parser.add_argument("--train_dataset_size", type=int, default=200,
                        help="choose the first train_dataset_size examples from train dataset for training")

    parser.add_argument("--data_dir", type=str, default=None,
                        help="data dir used for experiment")

    parser.add_argument("--rationale_path", type=str, default=None,
                        help="rationale path used for experiment")

    parser.add_argument("--save_dir", type=str, default=None,
                        help="save dir used for experiment")

    parser.add_argument("--llm_model", type=str,
                        choices=["davinci", "gpt-3.5-turbo", "gpt-3.5-turbo-ca", "gpt-3.5-turbo-0613",
                                 "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-4-turbo-2024-04-09",
                                 "gpt-4o-ca"],
                        default="gpt-3.5-turbo-ca", help="language model used for experiment")

    parser.add_argument("--multi_thread", type=bool, default=True,
                        help="whether to use multi-thread to accelerate")

    parser.add_argument("--epoch", type=int, default=5,
                        help="epoch used for experiment")

    parser.add_argument("--cold_start_topN", type=int, default=5,
                        help="output topN results for every call LLM.generate in cold start phase")

    parser.add_argument("--cold_start_temperature", type=float, default=0.5,
                        help="temperature used in cold start phase")

    parser.add_argument("--cold_start_try_num", type=int, default=3,
                        help="the number of tries in cold start phase")

    parser.add_argument("--train_rule_num", type=int, default=50,
                        help="the number of rules used in training phase")

    parser.add_argument("--train", type=bool, default=True,
                        help="whether to train")

    parser.add_argument("--eval", type=bool, default=False,
                        help="whether to eval")

    parser.add_argument("--test", type=bool, default=False,
                        help="whether to test")

    parser.add_argument("--cold_start_num", type=int, default=200,
                        help="the number of examples chosen in cold start phase")

    parser.add_argument(
        "--encoder", type=str, default="all-MiniLM-L6-v2", help="which sentence-transformer encoder for clustering"
    )

    parser.add_argument("--cot_trigger_type", type=str, default='category_prompt',
                        choices=['category_prompt', 'lang8'],
                        help="zero-shot prompt for cold start phase")

    parser.add_argument("--train_prompt_type", type=str, default=None, choices=None,
                        help="instruction prompt for training phase with few-shot examples chosen automatically such as AutoCoT (NotImplemented), "
                             "or use cot_trigger_prompt when None. "
                             "Should use the same format as cot_trigger_prompt.")

    parser.add_argument("--force_check_rate", type=float, default=0.5,
                        help="used to decide whether to replace a rule with the one in rule_map, aims to control the "\
                        "frequency of rule usage")

    args = parser.parse_args()

    def get_prompt(prompt_dict, dataset: str, *args):
        elem = prompt_dict
        params = [dataset] + list(args)

        sign = True
        while elem and params:
            try:
                elem = elem.get(params.pop(0))
            except AttributeError:
                sign = False
                break

        if sign and elem:
            return elem
        elif dataset == "Default":
            raise AttributeError
        else:
            dataset = "Default"
            return get_prompt(prompt_dict, dataset, *args[0:1]) # 这里这个逻辑设计的非常奇怪

    args.cot_trigger = get_prompt(prompt_dict, args.dataset, 'CoT', args.cot_trigger_type)
    args.pred_trigger = get_prompt(prompt_dict, args.dataset, 'pred_trigger') # the format used should be same as cot_trigger
    args.train_prompt = get_prompt(prompt_dict, args.dataset, 'train_prompt', args.train_prompt_type) \
        if args.train_prompt_type else args.cot_trigger

    args.direct_answer_trigger_for_zeroshot_cot = args.pred_trigger

    if not args.data_dir:
        args.data_dir = f"./data/{args.dataset}"

    if not args.save_dir:
        num_suffix = 0
        while os.path.exists(f"./experiment/{args.dataset}/version_{num_suffix}"):
            file_list = os.listdir(f"./experiment/{args.dataset}/version_{num_suffix}")
            if file_list == ['args.txt']:
                break

            num_suffix += 1
        args.save_dir = f"./experiment/{args.dataset}/version_{num_suffix}"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.multi_thread:
        if os.path.exists(os.path.join(args.data_dir, "rationale/ZeroShotCoTParallel.jsonl")):
            args.rationale_path = os.path.join(args.data_dir, "rationale/ZeroShotCoTParallel.jsonl")
    else:
        if os.path.exists(os.path.join(args.data_dir, "rationale/ZeroShotCoT.jsonl")):
            args.rationale_path = os.path.join(args.data_dir, "rationale/ZeroShotCoT.jsonl")

    NameSpace._args = args

    logger.info(f"args: {args}")
    with open(os.path.join(args.save_dir, "args.txt"), 'w') as f:
        f.write(str(args))

    return args


def main():
    """
    1. 读取数据集
    2. 构造Trainer
        2.1 构造ZeroShotCoT
        2.2 抽取出RuleBase
        2.3 进行训练
    3. 评估
    """

    args = args_parse()

    # 1. 读取数据集
    train_dataset, valid_dataset, test_dataset = read_datasets(args)

    if args.rationale_path:
        train_dataset, valid_dataset, test_dataset = read_rationales(args,
                                                                     train_dataset=train_dataset,
                                                                     valid_dataset=valid_dataset,
                                                                     test_dataset=test_dataset)


    # 2. 构造Trainer
    # 2.1 构造ZeroShotCoT + # 2.2 抽取出RuleBase
    generate_func = generate_func_mapping(args.llm_model)
    llm_model = LLM(generate_func)

    cur_Trainer = Trainer(args, train_dataset, valid_dataset, test_dataset, llm_model,
                          knowledge_base=KnowledgeBase()) #topN是个小问题

    if args.train:    # 需要cold start的时候运行
        cur_Trainer.cold_start()  # 存Answer的时候就clean一下

    # 2.3 进行训练
    if args.train:
        cur_Trainer.train()

    # # 3. 评估

    # if args.eval:
    #     cur_Trainer.eval()
    #     cur_Trainer.evaluate(is_valid=True)
    #
    # if args.test:
    #     cur_Trainer.eval()
    #     cur_Trainer.evaluate(is_valid=False)


if __name__ == '__main__':
    main()