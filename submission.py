import csv

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)
from tqdm import tqdm

from utils.data_utils import preprocess_source


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tokenizer = GPT2TokenizerFast.from_pretrained('models/test')
model = GPT2LMHeadModel.from_pretrained('models/test').to(DEVICE)

inputs = []
preds = []
with open('data/orig/input/dev.tsv', 'r', encoding='utf8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        text = preprocess_source(row[0])
        input_ids = tokenizer.encode(text, return_tensors='pt').to(DEVICE)
        # model_output = model.generate(
        #     input_ids,
        #     do_sample=True,
        #     max_length=128,
        #     top_k=40,
        #     top_n=0.5,
        #     temperature=1.0,
        #     num_return_sequences=1,
        #     early_stopping=True,
        #     eos_token_id=tokenizer.eos_token_id,
        # )
        model_output = model.generate(
            input_ids,
            max_length=128,
            num_beams=2,
            # no_repeat_ngram_size=2,
            num_return_sequences=1,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        preds.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in model_output])

with open('data/santana/submission.txt', 'w', encoding='utf8') as sub:
    i = 0
    for line in preds:
        try:
            line = line.split(' === ')[1]
        except IndexError:
            line = line
            i += 1
        sub.write(line + '\n')
    print(i)
