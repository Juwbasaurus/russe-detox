from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)


def infer(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    text: str,
) -> str:
    input_ids = tokenizer.encode(text, return_tensors='pt')
    model_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=256,
        top_k=40,
        top_n=0.7,
        temperature=0.95,
        num_return_sequences=1,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(model_output)
