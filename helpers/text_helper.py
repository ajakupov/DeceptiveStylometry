import string
import nltk


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def preprocess_text(text):
    text = text.lower()
    text = remove_punctuation(text)
    return text


def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    return [token for token in tokens if any(c.isalpha() for c in token)]
