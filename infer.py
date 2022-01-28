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
    i = 0
    for row in reader:
        print(row[0])
        input = tokenizer.encode(row[0], return_tensors='pt')
        model_output = model.generate(
            input,
            do_sample=True,
            max_length=256,
            top_k=40,
            top_n=0.7,
            temperature=0.95,
            num_return_sequences=1,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        for out in model_output:
            print(tokenizer.decode(out))
        if i >= 10:
            break
        else:
            i += 1

        # input = tokenizer.encode(row[0] + '===', return_tensors='pt')
        # model_output = model.generate(
        #     input,
        #     max_length=128,
        #     num_beams=5,
        #     no_repeat_ngram_size=3,
        #     num_return_sequences=1,
        #     early_stopping=True,
        #     eos_token_id=tokenizer.eos_token_id,
        # )
        # for out in model_output:
        #     print(tokenizer.decode(out))
