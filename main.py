from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig, AutoModel
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed

def compare_layer(param_1, param_2, name, use_cuda):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    param_1, param_2 = param_1.to(device), param_2.to(device)

    param_1_flat = param_1.view(-1).detach()
    param_2_flat = param_2.view(-1).detach()

    if use_cuda:
        cos_sim = torch.nn.functional.cosine_similarity(param_1_flat.unsqueeze(0), param_2_flat.unsqueeze(0)).item()
    else:
        param_1_flat = param_1_flat.cpu().numpy().reshape(1, -1)
        param_2_flat = param_2_flat.cpu().numpy().reshape(1, -1)
        cos_sim = cosine_similarity(param_1_flat, param_2_flat)[0][0]

    similarity_ratio = torch.sum(param_1_flat == param_2_flat).cpu().numpy() / param_1_flat.numel() * 100

    return name, similarity_ratio, cos_sim

def compare_model_layers(model_1, model_2, use_cuda=False):
    layer_similarity_ratio = {}
    cosine_similarities = {}

    with ThreadPoolExecutor() as executor:
        futures = []
        for (name_1, param_1), (name_2, param_2) in zip(model_1.named_parameters(), model_2.named_parameters()):
            if name_1 == name_2:
                futures.append(executor.submit(compare_layer, param_1, param_2, name_1, use_cuda))

        for future in as_completed(futures):
            name, similarity_ratio, cos_sim = future.result()
            name = name.replace('model.', '').replace('layer.', '').replace('layers.', '').replace('.weight', '')
            if name[1] == '.':
                name = f'0{name}'
            layer_similarity_ratio[name] = similarity_ratio
            cosine_similarities[name] = cos_sim

    # Sort results by layer name to maintain order
    layer_similarity_ratio = dict(sorted(layer_similarity_ratio.items()))
    cosine_similarities = dict(sorted(cosine_similarities.items()))

    return layer_similarity_ratio, cosine_similarities

def create_table(similarity_results, cosine_results):
    table = pd.DataFrame({
        'Layer': list(similarity_results.keys()),
        'Similarity': list(similarity_results.values()),
        'Cosine Similarity': list(cosine_results.values())
    })
    return table

def main():
    parser = argparse.ArgumentParser(description="Compare model layers including lm_head with optional CUDA acceleration")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to first model")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to second model")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file for results")
    parser.add_argument("--use_cuda", action='store_true', help="Use CUDA for acceleration if available")
    args = parser.parse_args()

    model_1 = AutoModelForCausalLM.from_pretrained(args.model1_path)
    model_2 = AutoModelForCausalLM.from_pretrained(args.model2_path)

    similarity_results, cosine_results = compare_model_layers(model_1, model_2, args.use_cuda)
    table = create_table(similarity_results, cosine_results)

    mean_row = pd.DataFrame({
        'Layer': ['Mean'],
        'Similarity': [np.mean(list(similarity_results.values()))],
        'Cosine Similarity': [np.mean(list(cosine_results.values()))]
    })

    table = pd.concat([table, mean_row]).reset_index(drop=True)
    print(table)
    table.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()

