import os.path
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed  # 我先把多线程都取消了
import random
import json
from typing import Callable

from utils.data_classes import DatasetLoader, Example, Rationale, Knowledge, KnowledgeBase
from utils.data_classes.conceptual_memory import KnowledgeMemory, MultiKnowledgeMemory
from utils.llm import LLM
from utils.ExtraNameSpace import ScoreNameSpace
from utils.extract_knowledge import extract_knowledge_texts


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


def _inference_single(llm: LLM, input_text: str, cot_trigger: str | Callable[[str], str],
                      direct_answer_trigger_for_zeroshot_cot: str,
                      llm_model: str, temperature: float, top_n: int, try_times: int, cold_start_phase: bool = True) \
        -> list[tuple[str, str]]:
    """
    cold_start_phase和其他的区别是，要多一层direct_answer_trigger_for_zeroshot_cot
    """
    if isinstance(cot_trigger, str):
        llm_input = cot_trigger + "\n\n" + input_text
    elif isinstance(cot_trigger, Callable):
        llm_input = cot_trigger(input_text)
    else:
        raise TypeError("cot_trigger must be a string or a callable function")

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
    knowledge_memory: KnowledgeMemory | MultiKnowledgeMemory = knowledge_base.get_inference_knowledge_memory()
    knowledge_contents = list(set([v[0].content for v in knowledge_memory.values() if v]))

    chosen_num = min(50, len(knowledge_contents))
    chosen_knowledge = random.sample(knowledge_contents, chosen_num)

    tmp_train_prompt = ("Instruction: Following are several existed knowledge in knowledge base. When you answer the "
                        "questions, try to use the provided knowledge whenever possible in \"we retrieve\" format. "
                        "Try not to invent knowledge by yourself unless necessary. But if so, you are permitted to"
                        "establish your own rules in \"we have\" format.\n"
                        "Knowledge Base:\n")

    prompt = tmp_train_prompt + '\n'.join(
        chosen_knowledge) + '\n\n' + train_prompt + '\n\n' + input_text.strip()

    if not prompt.endswith(':'):
        prompt += "\nAnswer:"  # XXX:最初的实验加了，但有的提示词最后一个不叫Answer，换了名字。这种会出现问题
        # 为了不干扰旧的，先这样

    return prompt


def _get_optimal_knowledge(knowledge_memory: KnowledgeMemory | MultiKnowledgeMemory, context: str,
                           mode: str) -> Knowledge:
    if mode == 'train':
        this_knowledge: Knowledge = random.choice(knowledge_memory[context])  # 随机，似乎不适合greedy
    else:  # test或eval
        if isinstance(knowledge_memory, KnowledgeMemory):
            this_knowledge = knowledge_memory[context][0]
        else:  # MultiKnowledgeMemory
            # this_knowledge = knowledge_memory.get_first_value(key=context)[0]
            knowledges = knowledge_memory.get_first_value(key=context)
            this_knowledge = knowledges[0] if knowledges else None  # fixme: 这里是个临时修改

    return this_knowledge


def llm_inference_category(args,
                           knowledge_base: KnowledgeBase,
                           llm: LLM,
                           train_prompt: str | Callable[[str], str],
                           input_text: str,
                           mode: str = "train",
                           **kwargs) -> str:
    assert mode in ["train", "eval"], "mode must be in ['train', 'eval']"

    if isinstance(train_prompt, str):
        if mode == 'train':
            prompt = train_prompt + '\n\n' + input_text.strip()
            if not prompt.endswith(':'):
                prompt += "\nAnswer:"
        else:
            prompt = _tmp_adjust(args, knowledge_base, train_prompt, input_text, **kwargs)
    elif isinstance(train_prompt, Callable):
        prompt = train_prompt(input_text)
    else:
        raise TypeError("train_prompt must be a string or a callable function")

    input_length = len(prompt.split('\n'))

    absent_set = set()
    knowledge_memory = knowledge_base.get_knowledge_memory() if mode == "train" \
        else knowledge_base.get_inference_knowledge_memory()

    try_cnt = 0
    max_tries = 50  # 这里只是替换步数，非重新尝试，可以开大一点。比如一个10步的推理本身就需要10个max_tries
    while True:
        try_cnt += 1
        print("prompt:", prompt)

        prompt = add_space(prompt)
        response = llm.generate_single(input_text=prompt, model=args.llm_model, **kwargs)
        response = response.replace("\n\n", "\n")
        whole_text = prompt + " " + response

        prefix_response = "\n".join(whole_text.split('\n')[:input_length - 1])
        pending_lines: list[str] = whole_text.split('\n')[input_length - 1:]  # 所有除去prompt的句子。每轮current_line不清零，所以不影响位置

        if not pending_lines:  # 针对输出仅一行
            return response

        replace_line = None
        sign = False
        for cur_line in pending_lines:
            if args.pred_trigger.lower() in cur_line.lower():
                break

            current_knowledge = extract_knowledge_texts(cur_line)
            if not current_knowledge:
                prefix_response += f'\n{cur_line}'
                continue

            context = cur_line[:cur_line.lower().index(current_knowledge[0].lower())]  # 第一个knowledge前面的文字被认为是场景
            # fixme: context这个命名也可以改

            key_context = context
            if key_context not in knowledge_memory or not knowledge_memory[key_context]:
                absent_set.add(key_context)
                prefix_response += f'\n{cur_line}'
                continue

            if random.random() > args.force_check_rate:
                prefix_response += f'\n{cur_line}'
                continue

            # =============
            this_knowledge = _get_optimal_knowledge(knowledge_memory=knowledge_memory,
                                                    context=key_context,
                                                    mode=mode)

            if this_knowledge is None:  # fixme: 不该存在，后面修改
                prefix_response += f'\n{cur_line}'
                continue

            if len(current_knowledge) > 1:
                warnings.warn("It's better to have only one knowledge in line: " + cur_line)
            # otherwise, you should design a more specific replacement strategy

            current_knowledge = current_knowledge[0]
            replace_line = context + current_knowledge
            last_line = replace_line
            replace_line = replace_line.replace(current_knowledge, this_knowledge.content)
            if replace_line != last_line:  # 替换前后一样的话要继续看下一行。否则就可以重新去生成response了
                print(f"进行一次有效替换：{last_line} → {replace_line}")
            else:
                prefix_response += f'\n{replace_line}'
                continue
            # =============

            sign = True
            break

        if not sign or try_cnt > max_tries:  # XXX: 第二个判断位置也不好
            out = "\n".join(pending_lines)
            print("response: ", out)
            return out

        if sign:
            prompt = f'{prefix_response}\n{replace_line}'
            print('有替换！')
        else:
            print('无替换！')
