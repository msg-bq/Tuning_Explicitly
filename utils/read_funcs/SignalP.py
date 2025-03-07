import os
import random

from Bio import SeqIO

from utils import DatasetsReaderNameSpace

_question_template_signalp = """You are an expert in genomics and bioinformatics.
Question: Can you identify if there is any signal peptides in the following sequence:
{sequence}
please giving the answer in "SP" or "NO_SP" to represent there seems like a signal peptides or no signal peptides.
Again, you should think step by step to analyze given sequence and then give "SP" or "NO_SP" to answer if there is any signal peptides."""


def _read_signalp(path: str) -> tuple[list[dict], list[dict]]:
    assert path.endswith('fasta')

    sp_dataset = []
    no_sp_dataset = []
    for record in SeqIO.parse(path, "fasta"):
        length = len(record.seq) / 2
        seq = record.seq[:int(length)]
        annotations = record.seq[int(length):]
        uniprot, kingdom, label, partition_no = record.id.split('|')
        if label not in ['NO_SP', 'SP']:
            continue  # XXX: 为了评分和文章阐述方便，我们只考虑无信号肽和一般信号肽

        data = {'uniprot': uniprot,
                'kingdom': kingdom,
                'gold_label': label,  # {'label': label,
                # 'annotations': str(annotations)},
                'partition_no': partition_no,
                'question': _question_template_signalp.format(sequence=str(seq))}

        if label == 'SP':
            sp_dataset.append(data)
        else:
            no_sp_dataset.append(data)

    # 随机拆分训练集和测试集，但控制下sp和no_sp的比例，原始数据集太悬殊了。大致sp>1/3
    random.shuffle(sp_dataset)
    random.shuffle(no_sp_dataset)

    dataset = []
    while sp_dataset and no_sp_dataset:
        if random.random() < 0.33:
            dataset.append(sp_dataset.pop())
        else:
            dataset.append(no_sp_dataset.pop())

    train_set = dataset[:-200]
    test_set = dataset[-200:]

    return train_set, test_set


@DatasetsReaderNameSpace.register("SignalP")
def read_func(data_dir):
    train_data, test_data = _read_signalp(os.path.join(data_dir, 'benchmark_set.fasta'))

    return train_data, None, test_data
