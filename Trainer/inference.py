import os.path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed  # 我先把多线程都取消了
import random
from typing import List, Tuple
import json

from utils.data import DatasetLoader, Example, Rationale, KnowledgeBase
import utils.extract_knowledge
from utils.llm import LLM
from utils.ExtraNameSpace import ScoreNameSpace
import utils.clean_prediction_func
import utils.score


@ScoreNameSpace.register("Example")
def is_high_quality_prediction(prediction: str, gold_label: str) -> bool:
    pass


def cold_start_inference(args, llm: LLM, dataset: DatasetLoader):
    def select_dataset():
        """
        挑出没有rationale且应当获取的data
        """
        for data in dataset[:args.cold_start_num]:
            if data.rationales:  # 有可能rationale已经存在了，这个时候就不需要再生成了。但要注意的是，如果调整了rationale的录入格式
                continue

            if data.question.strip() in fail_examples:
                continue

            yield data

    rationale_dir = os.path.join(args.data_dir, "rationale")
    fail_file = os.path.join(rationale_dir, "fail.jsonl")

    fail_examples = []
    os.makedirs(rationale_dir, exist_ok=True)

    if not os.path.exists(fail_file):
        with open(fail_file, 'w'):
            pass

    with open(fail_file, 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            fail_examples.append(line['question'].strip())

    cold_start_path = os.path.join(rationale_dir, "ColdStart.jsonl")
    if not os.path.exists(cold_start_path):
        with open(cold_start_path, 'w'):
            pass

    def save_to_cold_start(example: Example):
        if example.rationales:
            final_dict = example.to_dict()
            if example.rationales:
                with open(cold_start_path, 'w') as f:
                    assert len(final_dict['rationale']) == len(final_dict['prediction'])
                    for r, p in zip(final_dict['rationale'], final_dict['prediction']):
                        tmp_dict = {'question': final_dict['question'].strip(),
                                    'gold_label': final_dict['gold_label'].strip(),
                                    'rationale': r.strip(), 'prediction': p.strip()}
                        f.write(json.dumps(tmp_dict) + '\n')

            return True

        with open(fail_file, 'a') as f:
            f.write(json.dumps({'question': example.question.strip()}) + '\n')

        return False

    if args.multi_thread:
        with ThreadPoolExecutor(max_workers=200) as executor:
            responses = [executor.submit(_cold_start_inference_single, args, llm, data) for data in select_dataset()]

            for r in as_completed(responses):
                save_to_cold_start(r.result())

    else:
        for data in select_dataset():
            example = _cold_start_inference_single(args, llm, data)
            save_to_cold_start(example)

    return dataset


def _cold_start_inference_single(args, llm: LLM, example: Example):
    rationales_answers_pair = _inference_single(llm=llm, input_text=example.question,
                                                cot_trigger=args.cot_trigger,
                                                direct_answer_trigger_for_zeroshot_cot=args.direct_answer_trigger_for_zeroshot_cot,
                                                llm_model=args.llm_model,
                                                temperature=args.cold_start_temperature,
                                                topN=args.cold_start_topN,
                                                try_times=args.cold_start_try_num,
                                                cold_start_phase=True)

    for r, pred in rationales_answers_pair:
        r = Rationale(rationale=r, prediction=pred)

        if is_high_quality_prediction(prediction=r.prediction.strip(), gold_label=pred.strip()):
            example.update_rationale(r)
            break

    return example


def _inference_single(llm: LLM, input_text: str, cot_trigger: str, direct_answer_trigger_for_zeroshot_cot: str,
                      llm_model: str, temperature: float, topN: int, try_times: int, cold_start_phase: bool = True) \
                     -> List[Tuple[str, str]]:
    """
    cold_start_phase和其他的区别是，要多一层direct_answer_trigger_for_zeroshot_cot
    """
    llm_input = cot_trigger + "\n" + input_text

    rationales_answers_pair = []
    rationales = llm.generate_single_parallel(input_text=llm_input, model=llm_model,
                                              temperature=temperature,
                                              topN=topN,
                                              try_times=try_times)

    if cold_start_phase:
        for r in rationales:
            z2 = input_text + "Answer: " + r + " " + direct_answer_trigger_for_zeroshot_cot
            pred = llm.generate_single(input_text=z2, model=llm_model, temperature=0.0)

            if pred:
                rationales_answers_pair.append((r, pred))

    else:
        for r in rationales:
            pred = Rationale.clean_prediction(r)
            rationales_answers_pair.append((r, pred))

    return rationales_answers_pair


def llm_inference_category(args,
                           knowledge_base: KnowledgeBase,
                           llm: LLM,
                           train_prompt: str,
                           input_text: str, **kwargs) -> str:
    prompt = train_prompt + '\n' + input_text.strip() + "\nAnswer:"

    input_length = len(prompt.split('\n'))
    current_line = 0  # 初始行数

    absent_set = set()
    knowledge_memory = knowledge_base.get_knowledge_memory()     # 只建议读取，不写入

    while True:
        response = llm.generate_single(input_text=prompt, **kwargs)
        whole_text = prompt + " " + response
        pending_lines = whole_text.split('\n')[input_length-1:] # 所有除去prompt的句子。每轮current_line不清零，所以不影响位置

        concepts = None
        sign = False
        while current_line < len(pending_lines):
            line = pending_lines[current_line]
            current_line += 1

            if args.pred_trigger.lower() in line.lower():
                break

            if "we retrieve" in line.lower():
                line = line[:line.lower().index("we retrieve")]

            concepts = knowledge_base.extract_key_concepts(doc_list=line,
                                                           vectorizer=knowledge_base.vectorizer)  # ["(A, B)"], 例外：A, B

            if not concepts:
                warnings.warn('No concepts found in line: ' + line)
                continue

            concepts = concepts[0]

            if concepts not in knowledge_memory or not knowledge_memory[concepts]:
                absent_set.add(concepts)
                continue

            if random.random() < args.force_check_rate:
                continue

            sign = True
            current_line -= 1

            break

        if current_line >= len(pending_lines):
            out = "\n".join(pending_lines)
            return out

        if sign:
            this_rule = random.choice(knowledge_memory[concepts]) # 随机，似乎不适合greedy
            line = pending_lines[current_line+1] # 因为回退了一行
            current_rule = extract_knowledge_texts(line)
            assert len(current_rule) == 1, "It's better to have only one rule in line: " + line

            line = line.replace(current_rule[0], this_rule['rule_text'])

            prompt = "\n".join(whole_text.split("\n")[:current_line+input_length-1]) + '\n' + line # 这个-1不一定对
            current_line += 1

            print('有替换！')
        else:
            print('无替换！')
