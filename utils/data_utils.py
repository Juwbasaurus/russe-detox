def preprocess_source(text: str):
    text = text.strip().lower()
    return text

def preprocess_target(text: str):
    text = preprocess_source(text)
    return text