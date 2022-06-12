import nltk

from helpers.ott_helper import get_ott_negative_deceptive, get_ott_negative_truthful
from helpers.text_helper import preprocess_text, tokenize_text
from beans.StyleFeature import StyleFeature


class StyleDetector:
    def __init__(self, num_features=10):
        self.num_features = num_features
        self.deceptive_tokens = get_deceptive_tokens()
        self.truthful_tokens = get_truthful_tokens()
        self.deceptive_features = self.get_deceptive_feature_frequencies()
        self.truthful_features = self.get_truthful_feature_frequencies()

    def get_feature_mean(self):
        for feature in self.get_features():
            feuture_average = self.find_feature_by_name(feature, self.get_deceptive_feature_frequencies()) + \
                              self.find_feature_by_name(feature, self.get_truthful_feature_frequencies())

    def find_feature_by_name(self, target, style_features):
        for feature in style_features:
            if target == feature.feature:
                return feature

    def get_deceptive_feature_frequencies(self):
        feature_frequencies = []

        for feature in self.get_features():
            overall = len(self.deceptive_tokens)
            presence = self.deceptive_tokens.count(feature)
            feature_frequencies.append(StyleFeature(feature, presence/overall))

        return feature_frequencies

    def get_truthful_feature_frequencies(self):
        feature_frequencies = []

        for feature in self.get_features():
            overall = len(self.truthful_tokens)
            presence = self.truthful_tokens.count(feature)
            feature_frequencies.append(StyleFeature(feature, presence/overall))

        return feature_frequencies

    def get_corpus_tokens(self):
        whole_corpus = self.deceptive_tokens + self.truthful_tokens
        whole_corpus_freq_dist = list(nltk.FreqDist(whole_corpus).most_common(self.num_features))
        return whole_corpus_freq_dist

    def get_features(self):
        corpus_tokens = self.get_corpus_tokens()
        features = [word for word, freq in corpus_tokens]
        return features


def get_deceptive_tokens():
    text = ' '.join(get_ott_negative_deceptive()['text'].apply(lambda x: preprocess_text(x))[:300])
    return tokenize_text(text)


def get_truthful_tokens():
    text = ' '.join(get_ott_negative_truthful()['text'].apply(lambda x: preprocess_text(x))[:300])
    return tokenize_text(text)
