"""
CLI: compute mean/std hidden-state embeddings over a dataset, for later standardization.

This produces a Torch-serialized dict mapping `(layer, horizon)` -> `(mean, std)` where:
- `layer` indexes into `outputs.hidden_states`
- `horizon` is the number of consecutive tokens averaged into an embedding

It is intended for use with `IncrementalInformationValueScorer(..., mean_std_embeddings_path=...)`.

Notes / suggested improvements:
- Replace `eval()` parsing with `ast.literal_eval()` for safety.
- The running variance update works but is a bit bespoke; consider using a standard Welford update for clarity.
- Computing hidden states for *all* tokens can be expensive; consider truncation / sampling or batching.
"""

import argparse
import torch
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute mean and standard deviation of embeddings for a given dataset')
    parser.add_argument('--dataset', type=str, help='Path to the dataset')
    parser.add_argument('--output', type=str, help='Path to save the output')
    parser.add_argument('--model_name_or_path', type=str, help='Model name or path')
    parser.add_argument('--layers', type=str, default=None, help='Layers to use')
    parser.add_argument('--forecast_horizons', type=str, default=None, help='Forecast horizons to use')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--debug_n', type=int, default=0, help='Number of documents to debug')
    args = parser.parse_args()

    # Suggested improvement: use `ast.literal_eval` instead of `eval` (safer for untrusted inputs).
    args.layers = eval(args.layers) if args.layers else None
    args.forecast_horizons = eval(args.forecast_horizons) if args.forecast_horizons else None

    # Load csv dataset of documents using pandas
    dataset = pd.read_csv(args.dataset)
    assert 'text' in dataset.columns, 'The dataset should contain a column named "text"'
    assert 'id' in dataset.columns, 'The dataset should contain a column named "id"'

    texts = dataset['text'].tolist()
    ids = dataset['id'].tolist()
    if args.debug_n:
        texts = texts[:args.debug_n]
        ids = ids[:args.debug_n]

    # Initialize the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.device)

    running_stats = {}  # count, mean, var

    # Compute embeddings
    # NOTE: variable name `id` shadows Python's built-in `id()`.
    # Suggested improvement: rename to `doc_id` and avoid wrapping `zip(...)` in `list(...)`.
    for id, text in list(zip(ids, texts)):
        logging.info(f'Processing document {id}')
        inputs = tokenizer(text, return_tensors='pt').to(args.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            for layer in args.layers:
                for horizon in args.forecast_horizons:
                    # Slide a window of length `horizon` over the token sequence and average hidden states
                    # within each window to get one embedding per position.
                    for i in range(outputs.hidden_states[layer].shape[1] - horizon):
                        embedding = outputs.hidden_states[layer][0, i:i+horizon].mean(dim=0)
                        if (layer, horizon) not in running_stats:
                            running_stats[(layer, horizon)] = (1, embedding, None)
                        else:
                            count, old_mean, old_var = running_stats[(layer, horizon)]
                            new_count = count + 1
                            new_mean = (1 / new_count) * (old_mean * count + embedding)

                            if new_count == 2:
                                new_var = torch.cat([old_mean.unsqueeze(0), embedding.unsqueeze(0)]).var(dim=0)
                            else:
                                new_var = (new_count - 2) / (new_count - 1) * old_var + (1 / new_count) * (embedding - old_mean).pow(2)

                            running_stats[(layer, horizon)] = (new_count, new_mean, new_var)

    embeddings = {}
    for (layer, horizon), (_, mean, var) in running_stats.items():
        embeddings[(layer, horizon)] = (mean.cpu(), var.sqrt().cpu())

    torch.save(embeddings, args.output)
    
