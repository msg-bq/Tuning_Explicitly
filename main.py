import os.path

from Trainer.KnowledgeTrainer import Trainer
from utils.data_classes.knowledge_base_classes import KnowledgeBase
from utils.llm import LLM
from utils.read_datasets import read_datasets, read_rationales
import argparse
from utils.ExtraNameSpace import NameSpace

from prompt import prompt_dict

from logger import logger

import _import_overload


def args_parse():
    parser = argparse.ArgumentParser(description="Rule-Finetune")

    parser.add_argument("--dataset", type=str, default="CLUTRR",
                        choices=["default", "CLUTRR", "SST2", "LANG_8", "SALAD", "FOLIO_NL", "SignalP"],  # default包含一个通用的默认格式输入，暂时先不写
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
                                 "gpt-3.5-turbo-0125", "gpt-4-1106-preview", "gpt-4-turbo-2024-04-09",
                                 "gpt-3.5-turbo", "glm-4-air"],
                        default="gpt-3.5-turbo-0125", help="language model used for experiment")

    parser.add_argument("--multi_thread", type=bool, default=True,
                        help="whether to use multi-thread to accelerate")

    parser.add_argument("--epoch", type=int, default=5,  # 同样由于4的价格，SignalP用了3
                        help="epoch used for experiment")

    parser.add_argument("--cold_start_topN", type=int, default=3,  # 类似的，SignalP用了5
                        help="output topN results for every call LLM.generate in cold start phase")

    parser.add_argument("--cold_start_temperature", type=float, default=0.3,
                        help="temperature used in cold start phase")

    parser.add_argument("--cold_start_try_num", type=int, default=2,
                        help="the number of tries in cold start phase")

    parser.add_argument("--train", type=bool, default=False,
                        help="whether to train")

    parser.add_argument("--eval", type=bool, default=False,
                        help="whether to eval")

    parser.add_argument("--test", type=bool, default=False,
                        help="whether to test")

    parser.add_argument("--cold_start_num", type=int, default=20,
                        help="the number of examples chosen in cold start phase")

    parser.add_argument(
        "--encoder", type=str, default="all-MiniLM-L6-v2", help="which sentence-transformer encoder for clustering"
    )

    parser.add_argument("--cot_trigger_type", type=str, default='CLUTRR',
                        choices=['CLUTRR', 'lang8', 'SALAD'],
                        help="zero-shot prompt for cold start phase")

    parser.add_argument("--train_prompt_type", type=str, default=None, choices=None,  # fixme: 现在的
                        # train实则使用了cot_trigger，名字容易引起误解
                        help="Instruction prompt for training phase with few-shot examples chosen automatically "
                             "such as AutoCoT (NotImplemented), "
                             "or use cot_trigger_prompt when None. "
                             "Should use the same format as cot_trigger_prompt.")

    parser.add_argument("--test_prompt_type", type=str, default="CLUTRR_test_prompt", choices=None,  # hack: 这里应该单独给个test
                        help="Instruction prompt for training phase or use cot_trigger_prompt when None. "
                             "It's better to use the same format as cot_trigger_prompt.")

    parser.add_argument("--force_check_rate", type=float, default=0.5,  # SignalP用了0.2，因为gpt4贵
                        help="used to decide whether to replace a rule with the one in rule_map, aims to control the "
                             "frequency of rule usage")

    parser.add_argument("--build_conceptual_memory_method", type=str, default="tfidf",
                        help="the method to build conceptual memory, please register your own algorithms in "
                             "build_categorize_model.py and build_categorize_func.py")

    parser.add_argument("--force_overwrite", type=bool, default=False,
                        help="whether to overwrite the existing preprocessed dataset"
                             "and ZeroShotCoT rationale")

    args = parser.parse_args()

    def get_prompt(prompt_dct, dataset: str, *other_args):
        elem = prompt_dct
        params = [dataset] + list(other_args)

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
            return get_prompt(prompt_dct, dataset, *other_args[0:1])  # XXX: 这里这个逻辑设计的非常奇怪

    args.cot_trigger = get_prompt(prompt_dict, args.dataset, 'CoT', args.cot_trigger_type)
    args.pred_trigger = get_prompt(prompt_dict, args.dataset,
                                   'pred_trigger')  # the format used should be same as cot_trigger
    args.train_prompt = get_prompt(prompt_dict, args.dataset, 'train_prompt', args.train_prompt_type) \
        if args.train_prompt_type else args.cot_trigger
    args.test_prompt = get_prompt(prompt_dict, args.dataset, args.test_prompt_type) \
        if args.test_prompt_type else args.cot_trigger

    args.direct_answer_trigger_for_zeroshot_cot = args.pred_trigger

    if not args.data_dir:
        args.data_dir = f"./data/{args.dataset}"

    def _is_incomplete_dir(dir_path: str) -> bool:
        """
        只有两个文件被认为不完整（其实就是args和空的train loss）
        """
        return len(os.listdir(dir_path)) <= 2
    # warnings.warn("We use knowledge_base_final to judge whether a dir is complete or not.")
    # return not os.path.exists(os.path.join(dir_path, "knowledge_base_final"))
    # todo: 这会导致我不能并行开n个

    if not args.save_dir:
        num_suffix = 1
        while os.path.exists(f"./experiment/{args.dataset}/version_{num_suffix}") and \
                not _is_incomplete_dir(f"./experiment/{args.dataset}/version_{num_suffix}"):
            file_list = os.listdir(f"./experiment/{args.dataset}/version_{num_suffix}")
            if len(file_list) <= 2:
                break

            num_suffix += 1
        if not args.train and args.test:
            num_suffix -= 1
        args.save_dir = f"./experiment/{args.dataset}/version_{num_suffix}"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)  # todo: 删一下没用到的version

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
    if not args.force_overwrite and args.rationale_path:
        train_dataset, valid_dataset, test_dataset = read_rationales(args,
                                                                     train_dataset=train_dataset,
                                                                     valid_dataset=valid_dataset,
                                                                     test_dataset=test_dataset)

    # 2. 构造Trainer
    # 2.1 构造ZeroShotCoT + # 2.2 抽取出RuleBase
    llm_model = LLM(generate_func_or_name=args.llm_model)

    cur_Trainer = Trainer(args, train_dataset, valid_dataset, test_dataset, llm_model,
                          knowledge_base=KnowledgeBase
                          (build_conceptual_memory_method=args.build_conceptual_memory_method))  # topN是个小问题

    if args.train:  # 需要cold start的时候运行
        cur_Trainer.cold_start()  # 存Answer的时候就clean一下
        # 2.3 进行训练
        cur_Trainer.train()

    # # 3. 评估

    # if args.eval:
    #     cur_Trainer.eval()
    #     cur_Trainer.evaluate(is_valid=True)

    if args.test:
        # args.save_dir = r'D:\Github\Tuning_Explicitly\experiment\CLUTRR\version_71'
        cur_Trainer.test(  # r'D:\Github\Tuning_Explicitly\experiment\CLUTRR\version_86',
            # r"D:\Github\Tuning_Explicitly\experiment\LANG_8\version_6",
            r'D:\Github\Tuning_Explicitly\experiment\CLUTRR\version_21',
            # r"D:\Github\Tuning_Explicitly\experiment\FOLIO_NL\version_154",  # 148是tfidf，47是hyperplane
            # args.save_dir,
            use_epoch_file='final')
        # 25是最普通的random200，配上inference 50
        # 26是inference 50训的，
        # 28也是
        # 34 200 48
        # 37 200 48
        # 69 200s
        # 71 2000 58
        # 72 381 0.575
        # 73, 74 top 200。74好像是 0.45
        # 76 2000 TOp we retrieve 0.45 然后给个上限的分析
        # 78 glm-3-turbo
        # 训练过程基本非常稳定，随便选一个version+重复3遍做最后的实验即可
        # 86是5000


# rake, 0.325, tfidf 0.4，中间0.355不知道是哪个。对应了39-41

# CLUTRR, tfidf, top=1.0, only confidence
# CLUTRR, hyperplane, top=0.5, confidence+similarity
# 其他数据集好像都是拿confidence+similarity顺手做的，没有关心细节参数，top当时可能是0.5。


if __name__ == '__main__':
    main()
