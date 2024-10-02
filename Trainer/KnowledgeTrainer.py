class Trainer:
    def __init__(self, args, train_dataset: DatasetLoader, valid_dataset: DatasetLoader, test_dataset: DatasetLoader,
                 llm: LLM, rule_base: RuleBase = RuleBase()):
        self.args = args
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.llm = llm
        self.rule_base = rule_base

        self.lock = threading.Lock()

    def cold_start(self):
        """
        """
        dataset = zero_shot_CoT(self.args, self.llm, self.train_dataset)

        for data in dataset:
            knowledges = []
            for r in data.rationale:
                knowledges += r.extract_knowledge()  # 这儿也没有根据prediction和label的一致性选择正确的rule
                self.rule_base._add_knowledge(knowledges, data.question)

                self.rule_base.backward_buffer.append(
                    {'knowledges': knowledges,
                     'question': data.question,
                     'rationale': r.rationale, #回头根据rationale反演下rule的位置叭，这样有利于代码结构的统一，虽然引入了少量时间开销
                     'result': 'correct',
                     'score': 1})

        self.rule_base.build_conceptual_memory()
        self.rule_base.save(f"./data/{self.args.dataset}/rule_base_cold")
        logger.info("完成cold start")

