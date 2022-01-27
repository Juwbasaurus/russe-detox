import csv

from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

tokenizer = GPT2TokenizerFast.from_pretrained('models/test')
model = GPT2LMHeadModel.from_pretrained('models/test')

with open('data/orig/input/dev.tsv', 'r', encoding='utf8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    next(reader)
    next(reader)
    next(reader)
    for row in reader:
        input = tokenizer.encode(row[0] + ' === ', return_tensors='pt')
        model_output = model.generate(
            input,
            do_sample=True,
            max_length=128,
            top_k=50,
            top_n=0.9,
            temperature=0.9,
            num_return_sequences=1,
            early_stopping=True,
        )
        for out in model_output:
            print(tokenizer.decode(out))

        model_output = model.generate(
            input,
            max_length=128,
            num_beams=5,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            early_stopping=True
        )
        for out in model_output:
            print(tokenizer.decode(out))
        break
