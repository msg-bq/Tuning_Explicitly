import os.path
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed  # 我先把多线程都取消了
import random
from typing import List, Tuple
import json

from utils.data import DatasetLoader, Example, Rationale, KnowledgeBase, Knowledge
from utils.llm import LLM
from utils.ExtraNameSpace import ScoreNameSpace
from utils.extract_knowledge import extract_knowledge_texts
import utils.extract_knowledge
import utils.clean_prediction_func
import utils.score
from utils.llm_models import call_openai


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

    cold_start_path = os.path.join(rationale_dir, "ZeroShotCoTParallel.jsonl")
    if not os.path.exists(cold_start_path):
        with open(cold_start_path, 'w'):
            pass

    def save_to_cold_start(example_tmp: Example):
        if example_tmp.rationales:
            final_dict = example_tmp.to_dict()
            if example_tmp.rationales:
                with open(cold_start_path, 'a') as fs:
                    assert len(final_dict['rationale']) == len(final_dict['prediction'])
                    for ration, pred in zip(final_dict['rationale'], final_dict['prediction']):
                        tmp_dict = {'question': final_dict['question'].strip(),
                                    'gold_label': final_dict['gold_label'].strip(),
                                    'rationale': ration.strip(), 'prediction': pred.strip()}
                        fs.write(json.dumps(tmp_dict) + '\n')

            return True

        with open(fail_file, 'a') as fs:
            fs.write(json.dumps({'question': example_tmp.question.strip()}) + '\n')

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
                                                top_n=args.cold_start_topN,
                                                try_times=args.cold_start_try_num,
                                                cold_start_phase=True)

    for r, pred in rationales_answers_pair:
        r = Rationale(rationale=r, prediction=pred)

        if is_high_quality_prediction(prediction=r.prediction.strip(),
                                      gold_label=example.gold_label.strip()):
            example.update_rationale(r)
            break

    return example


def _inference_single(llm: LLM, input_text: str, cot_trigger: str, direct_answer_trigger_for_zeroshot_cot: str,
                      llm_model: str, temperature: float, top_n: int, try_times: int, cold_start_phase: bool = True) \
                     -> List[Tuple[str, str]]:
    """
    cold_start_phase和其他的区别是，要多一层direct_answer_trigger_for_zeroshot_cot
    """
    llm_input = cot_trigger + "\n\n" + input_text

    rationales_answers_pair = []
    rationales = llm.generate_single_parallel(input_text=llm_input, model=llm_model,
                                              temperature=temperature,
                                              topN=top_n,
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


def add_space(s):
    """
    add space after every comma if there is no space after it
    """
    return re.sub(r'(?<=[,])(?=[^\s])', r' ', s)


def _tmp_adjust(args, knowledge_base: KnowledgeBase, train_prompt: str, input_text: str, **kwargs):
    knowledge_memory = knowledge_base.get_inference_knowledge_memory()
    knowledge_contents = [v[0].content for v in knowledge_memory.values() if v]
    # knowledge_contents = [k.content for v in knowledge_memory.values() for k in v]

    chosen_num = min(50, len(knowledge_contents))
    chosen_knowledge = random.sample(knowledge_contents, chosen_num)

    tmp_train_prompt = "Instruction: Following are several existed knowledge in knowledge base. When you answer the questions, try to use the provided knowledge whenever possible in \"we retrieve\" format. "\
    "Try not to invent knowledge by yourself unless necessary. But if so, you are permitted to"\
    "establish your own rules in \"we have\" format.\n"\
    "Knowledge Base:\n"\
    "brother's sister is sister."

    prompt = tmp_train_prompt + '\n'.join(chosen_knowledge) + '\n\n' + train_prompt + '\n\n' + input_text.strip() + "\nAnswer:"

    return prompt


def llm_inference_category(args,
                           knowledge_base: KnowledgeBase,
                           llm: LLM,
                           train_prompt: str,
                           input_text: str,
                           mode: str = "train",
                           **kwargs) -> str:

    assert mode in ["train", "eval"], "mode must be in ['train', 'eval']"

    # if mode == 'train':
    #     prompt = train_prompt + '\n\n' + input_text.strip() + "\nAnswer:"
    # else:
    prompt = _tmp_adjust(args, knowledge_base, train_prompt, input_text, **kwargs)

    input_length = len(prompt.split('\n'))
    current_line = 0  # 初始行数

    absent_set = set()
    knowledge_memory = knowledge_base.get_knowledge_memory() if mode == "train" \
        else knowledge_base.get_inference_knowledge_memory()

    try_cnt = 0
    max_tries = 15
    while True:
        try_cnt += 1
        print("prompt:", prompt)

        prompt = add_space(prompt)
        response = llm.generate_single(input_text=prompt, model=args.llm_model, **kwargs)
        response = response.replace("\n\n", "\n")
        whole_text = prompt + " " + response
        pending_lines: List[str] = whole_text.split('\n')[input_length-1:]  # 所有除去prompt的句子。每轮current_line不清零，所以不影响位置

        if not pending_lines:   # 针对输出仅一行
            return response

        if not hasattr(knowledge_base, 'vectorizer'):
            warnings.warn(
                "The knowledge_base has not yet execute memorization phase to build a vectorizer"
            )
            return response

        concepts = None
        sign = False
        while current_line < len(pending_lines):
            line = pending_lines[current_line]
            current_line += 1

            if args.pred_trigger.lower() in line.lower():
                break

            current_knowledge = extract_knowledge_texts(line)
            if not current_knowledge:
                continue

            line = line[:line.lower().index(current_knowledge[0].lower())]

            concepts = knowledge_base.extract_key_concepts(doc_list=line,
                                                           vectorizer=knowledge_base.vectorizer)  # ["(A, B)"], 例外：A, B

            if not concepts or not len(concepts):
                warnings.warn('No concepts found in line: ' + line)
                continue

            concepts = concepts[0][1]

            if concepts not in knowledge_memory or not knowledge_memory[concepts]:
                absent_set.add(concepts)
                continue

            if random.random() > args.force_check_rate:
                continue

            sign = True
            current_line -= 1

            break

        if current_line >= len(pending_lines) or try_cnt > max_tries:
            out = "\n".join(pending_lines)
            print("response: ", out)
            return out

        if sign:
            this_knowledge: Knowledge = random.choice(knowledge_memory[concepts]) if mode == "train" \
                else knowledge_memory[concepts][0]  # 随机，似乎不适合greedy
            line = pending_lines[current_line]
            current_knowledge = extract_knowledge_texts(line)

            if len(current_knowledge) > 1:
                warnings.warn("It's better to have only one knowledge in line: " + line)
            # otherwise, you should design a more specific replacement strategy

            current_knowledge = current_knowledge[0]
            line = line[:line.index(current_knowledge)+len(current_knowledge)]
            last_line = line
            line = line.replace(current_knowledge, this_knowledge.content)
            if line != last_line:
                print(f"进行一次有效替换：{last_line} → {line}")

            prompt = "\n".join(whole_text.split("\n")[:current_line+input_length-1]) + '\n' + line
            current_line += 1

            print('有替换！')
        else:
            print('无替换！')
