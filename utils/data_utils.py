import re

EMOJIS = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       "]+", flags=re.UNICODE)


def preprocess_source(text: str):
    text = text.strip().lower()
    text = EMOJIS.sub(r'', text)
    text = re.sub(r':\(|\)', '', text)
    text = re.sub(r'([^а-я\d])\1{2,}', '\g<1>', text)
    # text = re.sub(r'((.)\2{3,})', ' \g<2> ', text)
    return text

def preprocess_target(text: str):
    text = preprocess_source(text)
    return text