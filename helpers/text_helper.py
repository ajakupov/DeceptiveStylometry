import string


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def preprocess_text(text):
    text = text.lower()
    text = remove_punctuation(text)

    return text