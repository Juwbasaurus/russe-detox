import re


def preprocess_source(text: str):
    text = text.strip().lower()
    text = re.sub(r'([^а-я\d])\1{2,}', '\g<1>', text)
    # text = re.sub(r'((.)\2{3,})', ' \g<2> ', text)
    return text

def preprocess_target(text: str):
    text = preprocess_source(text)
    return text