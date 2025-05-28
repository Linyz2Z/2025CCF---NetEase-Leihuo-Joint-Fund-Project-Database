import os
import torch
from torch.utils.data import Dataset
import pandas as pd

model_list = [
    "qwen25_72b_instruct",
    "gpt_4o_mini_cot",
    "ministral_8b_instruct_2410",
    "deepseek_chat",
    "glm_4_plus",
    "llama31_8b_instruct",
    "qwen25_32b_int4",
    "gpt_4o",
    "glm_4_air",
    "gpt_4o_mini",
    "qwen25_math_7b_instruct",
    "llama31_70b_instruct",
    "mistral_7b_instruct_v02",
    "mixtral_8x7b_instruct",
    "glm_4_flash",
    "qwq_32b_preview",
    "gemini15_flash",
    "deepseek_coder",
    "qwen25_7b_instruct",
    "llama31_405b_instruct"
]

class RouterDataset(Dataset):
    def __init__(self, dataset_path, test_mode=False):
        df = pd.read_csv(dataset_path)

        self.questions = df['question'].tolist()
        self.test_mode = test_mode
        if not self.test_mode:
            self.scores = torch.tensor(df[model_list].values.tolist())
            self.cluster_ids = df['cluster_id'].tolist() if 'cluster_id' in df.columns else None
        self.tokenizer = None
    
    def __getitem__(self, index):
        question = self.questions[index]
        question_id = self.tokenizer(
            question,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        question_id['input_ids'] = question_id.input_ids.flatten()
        question_id['attention_mask'] = question_id.attention_mask.flatten()

        if not self.test_mode:
            scores = self.scores[index]
            cluster_id = self.cluster_ids[index] if self.cluster_ids is not None else 0
            return question_id, scores, cluster_id
        else:
            return question_id
    
    def __len__(self):
        return len(self.questions)
    
    def register_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

