import argparse
import csv

from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
)


parser = argparse.ArgumentParser(description='Specify a tuned model path.')
parser.add_argument(
    'model_path',
    type=str,
    help="Path to model and tokenizer files.",
)
args = parser.parse_args()
model_path = args.model_path

tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

with open('data/orig/input/dev.tsv', 'r', encoding='utf8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    i = 0
    for row in reader:
        print(row[0])
        input = tokenizer.encode(row[0], return_tensors='pt')
        model_output = model.generate(
            input,
            max_length=128,
            num_beams=5,
            no_repeat_ngram_size=2,
            num_return_sequences=1,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        for out in model_output:
            print(tokenizer.decode(out, skip_special_tokens=True))
        if i >= 10:
            break
        else:
            i += 1


        # for out in model_output:
        #     print(tokenizer.decode(out))
