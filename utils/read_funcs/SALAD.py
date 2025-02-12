import json
import os.path
import random

from utils import DatasetsReaderNameSpace


def _read_SALAD(path: str) -> tuple[list[dict], list[dict]]:  # todo: 这里给个dataclass
    """
    自行拆分训练集和测试集
    """
    dataset: list[dict] = json.load(open(path, encoding='utf8'))
    new_dataset = []

    for data in dataset[2:-2]:  # hack: 一头一尾用于CoT举例子了
        if data['mcq']:  # 过滤空数据
            new_data = {'question': data['mcq'],
                        'gold_label': data['gt'],
                        'category': (data['1-category'], data['2-category'], data['3-category']),
                        'is_safe': True if data['mcq'].startswith('For each question you will choose the safe answer')
                        else False}

            new_dataset.append(new_data)

    # 先随机打乱数据集
    random.seed(42)
    random.shuffle(new_dataset)

    # 获取所有类别的唯一组合
    unique_categories = set(sample['category'] for sample in new_dataset)
    safe_samples = [sample for sample in new_dataset if sample['is_safe']]
    unsafe_samples = [sample for sample in new_dataset if not sample['is_safe']]

    # 选取保证类别覆盖的训练集
    train_set = []
    selected_categories = set()

    for sample in safe_samples:
        if sample['category'] not in selected_categories:
            train_set.append(sample)
            selected_categories.add(sample['category'])
        if len(selected_categories) == len(unique_categories):
            break

    selected_categories = set()  # XXX: 低质量代码
    for sample in unsafe_samples:
        if sample['category'] not in selected_categories:
            train_set.append(sample)
            selected_categories.add(sample['category'])
        if len(selected_categories) == len(unique_categories):
            break

    # 剩余数据
    remaining_samples = [sample for sample in new_dataset if sample not in train_set]

    # 填充训练集
    train_set += remaining_samples[:-200]  # 留出200个作为测试集
    test_set = remaining_samples[-200:]  # 取最后200个作为测试集

    return train_set, test_set


@DatasetsReaderNameSpace.register("SALAD")
def read_func(data_dir):
    train_data, test_data = _read_SALAD(os.path.join(data_dir, 'mcq_set.json'))

    return train_data, None, test_data
