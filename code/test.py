import yaml
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from transformers import AutoTokenizer, AutoModel

from router_dataset import RouterDataset
from router_model import RouterModule

def evaluate(models, datasets, tokenizer, batch_size, device): 
    # hard voting 
    for model in models:
        if isinstance(model, RouterModule):
            model.eval()  

    dataset_scores = np.zeros(len(datasets))
    with torch.no_grad():
        for index, dataset in enumerate(datasets):
            dataset_path = os.path.join("val_data", f"{dataset}.csv")
            test_dataset = RouterDataset(dataset_path)
            test_dataset.register_tokenizer(tokenizer)
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            true_scores = []
            for batch in data_loader:
                _, scores, _ = batch
                true_scores.append(scores)
            true_scores = torch.cat(true_scores)

            pred_scores = torch.zeros(len(test_dataset), 20)
            for idx, model in enumerate(models):
                router_model = model.to(device)
                preds = []
                for batch in data_loader:
                    inputs, _1, _2 = batch
                    inputs = inputs.to(device)
                    x, _ = router_model.forward(**inputs)
                    pred = F.softmax(x, dim=1).cpu()
                    preds.append(pred)
                preds = torch.cat(preds)
                preds = torch.argmax(preds, dim=1)
                preds = F.one_hot(preds, num_classes=20)
                pred_scores += preds

            max_index = torch.argmax(pred_scores, dim=1)
            score = true_scores.gather(dim=1, index=max_index.unsqueeze(1)).squeeze(1)
            score = score.sum().item()
            avg_score = score / len(test_dataset)
            dataset_scores[index] = avg_score
            print(f'{dataset} score: {avg_score}')
        

def test(models, datasets, tokenizer, batch_size, device): 
    llms = [
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

    # hard voting 
    for model in models:
        if isinstance(model, RouterModule):
            model.eval()  

    with torch.no_grad():
        for index, dataset in enumerate(datasets):
            dataset_path = os.path.join('test_data', f'{dataset}_test_pred.csv')
            test_dataset = RouterDataset(dataset_path, test_mode=True)
            test_dataset.register_tokenizer(tokenizer)
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            pred_scores = torch.zeros(len(test_dataset), 20)
            for idx, model in enumerate(models):
                router_model = model.to(device)
                preds = []
                for batch in data_loader:
                    inputs = batch
                    inputs = inputs.to(device)
                    x, _ = router_model.forward(**inputs)
                    pred = F.softmax(x, dim=1).cpu()
                    preds.append(pred)
                preds = torch.cat(preds)
                preds = torch.argmax(preds, dim=1)
                preds = F.one_hot(preds, num_classes=20)
                pred_scores += preds

            max_index = torch.argmax(pred_scores, dim=1)
            pred_results = [llms[i.item()] for i in max_index]
            test_df = pd.read_csv(dataset_path)
            test_df['pred'] = pred_results
            result_path = f"test_results/{dataset}_test_pred.csv"
            test_df.to_csv(result_path, index=False)
            print(f"Complete {dataset} testing.")

if __name__ == '__main__':
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    os.chdir(script_directory)

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_datasets', nargs='+', default=['aclue', 'arc_c', 'cmmlu', 'hotpot_qa', 'math', 'mmlu', 'squad'])
    parser.add_argument('--model_file', type=str, default='models.yaml')
    parser.add_argument('--final_test', action='store_true')

    parser.set_defaults(final_test=False)
    args = parser.parse_args()

    # Step 1: 加载预训练好的模型
    models = []
    weights = []
    with open(args.model_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    for info in tqdm(data['models'], desc="Load models..."):
        if info['type'] == 'RouterModule':
            encoder_model = AutoModel.from_pretrained('microsoft/mdeberta-v3-base')
            model = RouterModule(encoder_model, hidden_state_dim=768, node_size=20, similarity_function="cos")
            state_dict = torch.load(f"../model/{info['name']}/best_model.pth")
            model.load_state_dict(state_dict)
            models.append(model)
        
        elif info['type'] == 'StatisticsModel':
            models.append('StatisticsModel')
        
        weights.append(float(info['weight']))
    
    tokenizer = AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base', truncation_side='left', padding=True)
    
    # Step 2: 开始验证集测试
    if not args.final_test:
        print("Test on validation dataset:")
        evaluate(models, args.test_datasets, tokenizer, args.batch_size, args.device)

    # Step 3: 对测试集中数据进行预测
    if args.final_test:
        print("Start testing...")
        test(models, args.test_datasets, tokenizer, args.batch_size, args.device)




