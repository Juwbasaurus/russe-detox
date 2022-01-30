import argparse
import csv

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
)
from tqdm import tqdm

from utils.data_utils import preprocess_source


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Specify a tuned model path.')
parser.add_argument(
    'model_path',
    type=str,
    help="Path to model and tokenizer files.",
)
args = parser.parse_args()
model_path = args.model_path
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)

inputs = []
preds = []
with open('data/orig/input/dev.tsv', 'r', encoding='utf8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in tqdm(reader):
        text = preprocess_source(row[0])
        input_ids = tokenizer.encode(text, return_tensors='pt').to(DEVICE)
        model_output = model.generate(
            input_ids,
            max_length=128,
            num_beams=3,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        preds.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in model_output])

with open('data/santana/submission.txt', 'w', encoding='utf8') as sub:
    for line in preds:
        try:
            line = line.split(' === ')[1]
        except IndexError:
            line = line
        sub.write(line + '\n')
