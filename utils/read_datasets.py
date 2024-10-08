import json
import os
from typing import List, Union, Tuple

from utils.ExtraNameSpace import DatasetsReaderNameSpace
from utils.data import Example, DatasetLoader
from operator import itemgetter

"""
为不同的数据集，准备不同的读入方式，输入都是一个dir，输出train, dev, test, 输出统一{'question', 'gold_label'}
最终使用的数据集，将被单独保存为train_preprocessed.jsonl, dev_preprocessed.jsonl, test_preprocessed.jsonl
"""


# 加一个去重
@DatasetsReaderNameSpace.register("Example")
def read_func() -> Tuple[list, list, list]:
    pass


def _read_preprocessed_data(path) -> List[dict]:     # 读取预处理
    with open(path, 'r', encoding="utf8") as f:
        data = [line.strip() for line in f.readlines()]

        for i in range(len(data)):
            data[i]: dict = eval(data[i])

        return data


def _save_preprocessed_data(data: List[dict], path):    # 保存预处理
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path, 'w', encoding="utf8") as f:
        for sample in data:
            f.write(json.dumps(sample) + '\n')


def _adjust_dataset_format(**kwags):
    """
    将read_datasets输出的内容，调整为DatasetLoader的格式
    """
    for key, value in kwags.items():
        if value:
            kwags[key] = DatasetLoader([Example(**sample) for sample in value])
    return kwags


def _duplicate_removal(dataset: Union[List, DatasetLoader]) -> DatasetLoader:
    """
    对train, dev, test的读入进行去除，rationale的可以全保留
    """
    new_dataset = []
    for sample in dataset:
        if sample in new_dataset:
            continue
        else:
            new_dataset.append(sample)

    return DatasetLoader(new_dataset)


def _duplicate_removal_multi(**kwargs):
    for key, value in kwargs.items():
        if value:
            kwargs[key] = _duplicate_removal(value)
    return kwargs


def read_datasets(args) -> (DatasetLoader, DatasetLoader, DatasetLoader):
    data_dir = args.data_dir

    # 存储在preprocessed_data下面
    train_path = os.path.join(data_dir, 'preprocessed_data/train_preprocessed.jsonl')
    test_path = os.path.join(data_dir, 'preprocessed_data/test_preprocessed.jsonl')
    valid_path = os.path.join(data_dir, 'preprocessed_data/valid_preprocessed.jsonl')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        train_dataset, valid_dataset, test_dataset = read_func(data_dir)
        if args.train_dataset_size and args.train_dataset_size > 0:
            train_dataset = train_dataset[:args.train_dataset_size]

        _save_preprocessed_data(train_dataset, train_path)
        _save_preprocessed_data(test_dataset, test_path)
        if valid_dataset:
            _save_preprocessed_data(valid_dataset, valid_path)

    train_dataset = _read_preprocessed_data(train_path)
    test_dataset = _read_preprocessed_data(test_path)
    if os.path.exists(valid_path):
        valid_dataset = _read_preprocessed_data(valid_path)
    else:
        valid_dataset = None

    train_dataset, valid_dataset, test_dataset = \
        itemgetter('train', 'valid', 'test')(_adjust_dataset_format(train=train_dataset,
                                                                    valid=valid_dataset,
                                                                    test=test_dataset))

    return train_dataset, valid_dataset, test_dataset


def read_rationales(args, **kwargs):
    """
    读取rationale，将其加入到对应的数据集中
    """
    rationale_path = args.rationale_path

    if os.path.exists(rationale_path):
        rationale_dataset = _read_preprocessed_data(rationale_path)
        """
        修改对应的值，伪代码是
        for sample in train_rationale:
            existed_sample = address_mapping[sample]
            existed_sample.update(sample)
        """
        for sample in rationale_dataset:
            e = Example(question=sample['question'],
                        gold_label=sample['gold_label'])
            key = (e.question, e.gold_label)

            for _, value in kwargs.items():
                if value:
                    existed_sample = value.find(key, None)
                    # 有两类命名不统一，之后都得改改
                    # 1个是gold_ans和gold_label，另一个是answer和prediction
                    if existed_sample:
                        existed_sample.update(sample, args)  # update(e)应该更好
                    break

    return itemgetter('train_dataset', 'valid_dataset', 'test_dataset')(kwargs)
