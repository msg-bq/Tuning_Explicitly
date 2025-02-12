from datasets import load_dataset

from utils import DatasetsReaderNameSpace


_folio_prompt = """Suppose you are one of the greatest AI scientists, logicians and mathematicians. Let us think step by step.
Read and analyze the "Premises" first, then using First-Order Logic (FOL) to judge whether the "Hypothesis" is True, False or Unknown. You should follow the format of above three examples to answer this question.
Please make sure your reasoning is directly deduced from the "Premises" other than introducing unsourced common knowledge and unsourced information by common sense reasoning. Giving your answer by "each reasoning steps with logic knowledge and judgement (True, False or Unknown)".

Premises: {premises}
Hypothesis: {hypothesis}"""


def _format_question(premises: str, hypothesis: str) -> str:
    premises = "\n".join([f"{i+1}. {p.strip()}." for i, p in enumerate(premises.split('\n'))])
    return _folio_prompt.format(premises=premises,
                                hypothesis=hypothesis)


def _read_FOLIO_NL(data_dir: str = None) -> tuple[list[dict], list[dict]]:
    data_dir = data_dir or "yale-nlp/FOLIO"
    ds = load_dataset(data_dir)
    train_data = ds['train']
    train_dataset = [{'question': _format_question(d['premises'], d['conclusion']),
                      'gold_label': d['label']} for d in train_data]

    test_data = ds['validation']
    test_dataset = [{'question': _format_question(d['premises'], d['conclusion']),
                     'gold_label': d['label']} for d in test_data]

    return train_dataset, test_dataset


@DatasetsReaderNameSpace.register("FOLIO_NL")
def read_func(data_dir: str = None) -> tuple[list[dict], list[dict], list[dict]]:
    train_data, test_data = _read_FOLIO_NL(data_dir)

    return train_data, None, test_data


if __name__ == '__main__':
    _read_FOLIO("../../data/FOLIO_NL")