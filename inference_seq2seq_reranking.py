import argparse
import csv

import numpy as np

import torch
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)


parser = argparse.ArgumentParser(description='Specify a tuned model path.')
parser.add_argument(
    'model_path',
    type=str,
    help="Path to model and tokenizer files.",
)
parser.add_argument(
    'rerank_model_path',
    type=str,
    help="Path to model and tokenizer files.",
)
args = parser.parse_args()
model_path = args.model_path
rerank = args.rerank_model_path

tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.eval()

retokenizer = GPT2TokenizerFast.from_pretrained(rerank,
                                          pad_token='<pad>',
                                          bos_token='<s>',
                                          eos_token='<s>',
                                          unk_token='<unk>',
                                          mask_token='<mask>',)
remodel = GPT2LMHeadModel.from_pretrained(rerank)
remodel.eval()

with open('data/orig/input/dev.tsv', 'r', encoding='utf8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    i = 0
    for row in reader:
        print(row[0])
        with torch.no_grad():
            input = tokenizer.encode(row[0], return_tensors='pt')
            model_output = model.generate(
                input,
                max_length=128,
                num_beams=10,
                no_repeat_ngram_size=2,
                num_return_sequences=10,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
            )
            min_ppl = 100_000_000_000
            min_idx = 0
            for idx, out in enumerate(model_output):
                ppl = np.exp(remodel(out, labels=out).loss)
                if ppl < min_ppl:
                    print(ppl)
                    min_ppl = ppl
                    min_idx = idx
            print(tokenizer.decode(model_output[min_idx], skip_special_tokens=True))
            print(min_idx)
            if i >= 10:
                break
            else:
                i += 1


        # for out in model_output:
        #     print(tokenizer.decode(out))
