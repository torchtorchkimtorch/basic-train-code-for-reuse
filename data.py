from datasets import load_dataset

class SFTDataset:
    def __init__(self, tokenizer, reasoning_args, reasoning, data_path, data_path_type):
        self.tokenizer = tokenizer
        self.reasoning_args = reasoning_args
        self.reasoning = reasoning
        self.data_path = data_path
        self.data_path_type = data_path_type
        self.dataset = None
        self.chat = []
        self.whole_chat = []

    @staticmethod
    def sample_to_chat(sys, inp, out):
        if sys != "":
            chat = [
                {"role": "system", "content": sys},
                {"role": "user", "content": inp}
            ]
            whole_chat = [
                {"role": "system", "content": sys},
                {"role": "user", "content": inp},
                {"role": "assistant", "content": out}
            ]
        else:
            chat = [
                {"role": "user", "content": inp}
            ]
            whole_chat = [
                {"role": "user", "content": inp},
                {"role": "assistant", "content": out}
            ]
        return chat, whole_chat

    def preprocess(self):
        if self.data_path_type == "json":
            self.dataset = load_dataset("json", data_files=self.data_path)
        elif self.data_path_type == "hf":
            self.dataset = load_dataset(self.data_path)
        else:
            raise ValueError("Not supported data path type")

        def add_labels(example):
            chat, whole_chat = self.sample_to_chat(
                example.get("system_prompt", ""),
                example["user"],
                example["assistant"]
            )
            if self.reasoning_args == "it":
                tokenized_chat = self.tokenizer.apply_chat_template(
                        chat,
                        tokenize=True,
                        add_generation_prompt=True,
                )
            else:
                tokenized_chat = self.tokenizer.apply_chat_template(
                        chat,
                        tokenize=True,
                        add_generation_prompt=True,
                        **{self.reasoning_args: self.reasoning}
                )
            tokenized_whole_chat = self.tokenizer.apply_chat_template(
                    whole_chat,
                    tokenize=True,
                    add_generation_prompt=False,
            )
            label_len = len(tokenized_whole_chat) - len(tokenized_chat)
            labels = [-100] * (len(tokenized_whole_chat) - label_len)
            labels += tokenized_whole_chat[-label_len:]
            data = {
                "input_ids": tokenized_whole_chat,
                "labels": labels,
                "attention_mask": [1] * len(tokenized_whole_chat)
            }
            return data
        
        self.dataset = self.dataset.map(
            add_labels, 
            num_proc=4,
            remove_columns=["system_prompt", "user", "assistant"]
            )
        return self.dataset

