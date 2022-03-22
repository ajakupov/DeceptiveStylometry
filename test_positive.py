import nltk
import math
import pandas as pd

from helpers.ott_helper import get_ott_positive_deceptive, get_ott_positive_truthful
from helpers.text_helper import preprocess_text


if __name__ == '__main__':
    reviews_by_deception = {}

    reviews_by_deception['deceptive'] = ' '.join(
        get_ott_positive_deceptive()['text'].apply(lambda x: preprocess_text(x))[:300])
    reviews_by_deception['truthful'] = ' '.join(
        get_ott_positive_truthful()['text'].apply(lambda x: preprocess_text(x))[:300])
    deceptions = ['deceptive', 'truthful']
    reviews_by_deception_tokens = {}
    for deception in deceptions:
        tokens = nltk.word_tokenize(reviews_by_deception[deception])

        # Filter out punctuation
        reviews_by_deception_tokens[deception] = ([token for token in tokens if any(c.isalpha() for c in token)])

    for deception in deceptions:
        reviews_by_deception_tokens[deception] = (
            [token for token in reviews_by_deception_tokens[deception]])

    whole_corpus = []
    for deception in deceptions:
        whole_corpus += reviews_by_deception_tokens[deception]
    whole_corpus_freq_dist = list(nltk.FreqDist(whole_corpus).most_common(30))

    features = [word for word, freq in whole_corpus_freq_dist]
    feature_freqs = {}

    for deception in deceptions:
        # A dictionary for each candidate's features
        feature_freqs[deception] = {}

        # A helper value containing the number of tokens in the author's subcorpus
        overall = len(reviews_by_deception_tokens[deception])

        # Calculate each feature's presence in the subcorpus
        for feature in features:
            presence = reviews_by_deception_tokens[deception].count(feature)
            feature_freqs[deception][feature] = presence / overall

    corpus_features = {}

    # For each feature...
    for feature in features:
        # Create a sub-dictionary that will contain the feature's mean
        # and standard deviation
        corpus_features[feature] = {}

        # Calculate the mean of the frequencies expressed in the subcorpora
        feature_average = 0
        for deception in deceptions:
            feature_average += feature_freqs[deception][feature]
        feature_average /= len(deceptions)
        corpus_features[feature]["Mean"] = feature_average

        # Calculate the standard deviation using the basic formula for a sample
        feature_stdev = 0
        for deception in deceptions:
            diff = feature_freqs[deception][feature] - corpus_features[feature]["Mean"]
            feature_stdev += diff * diff
        feature_stdev /= (len(deceptions) - 1)
        feature_stdev = math.sqrt(feature_stdev)
        corpus_features[feature]["StdDev"] = feature_stdev

    feature_zscores = {}
    for deception in deceptions:
        feature_zscores[deception] = {}
        for feature in features:
            feature_val = feature_freqs[deception][feature]
            feature_mean = corpus_features[feature]["Mean"]
            feature_stdev = corpus_features[feature]["StdDev"]
            feature_zscores[deception][feature] = ((feature_val - feature_mean) / feature_stdev)

    deceptive_test = get_ott_positive_deceptive()['text'].apply(lambda x: preprocess_text(x))[300:]

    counter = 0
    for test in deceptive_test:
        testcase_tokens = nltk.word_tokenize(test)
        testcase_tokens = [token.lower() for token in testcase_tokens
                           if any(c.isalpha() for c in token)]

        overall = len(testcase_tokens)
        testcase_freqs = {}
        for feature in features:
            presence = testcase_tokens.count(feature)
            testcase_freqs[feature] = presence / overall

        # Calculate the test case's feature z-scores
        testcase_zscores = {}
        for feature in features:
            feature_val = testcase_freqs[feature]
            feature_mean = corpus_features[feature]["Mean"]
            feature_stdev = corpus_features[feature]["StdDev"]
            testcase_zscores[feature] = (feature_val - feature_mean) / feature_stdev
        label = ''
        score = 10000
        for deception in deceptions:
            delta = 0
            for feature in features:
                delta += math.fabs((testcase_zscores[feature] -
                                    feature_zscores[deception][feature]))
            delta /= len(features)
            #print("Delta score for candidate", deception, "is", delta)
            if delta < score:
                label = deception
                score = delta
        if label == "deceptive":
            counter += 1

    print("Deceptive test: {} %".format(counter))

    truthful_test = get_ott_positive_truthful()['text'].apply(lambda x: preprocess_text(x))[300:]

    counter = 0
    for test in truthful_test:
        testcase_tokens = nltk.word_tokenize(test)
        testcase_tokens = [token.lower() for token in testcase_tokens
                           if any(c.isalpha() for c in token)]

        overall = len(testcase_tokens)
        testcase_freqs = {}
        for feature in features:
            presence = testcase_tokens.count(feature)
            testcase_freqs[feature] = presence / overall

        # Calculate the test case's feature z-scores
        testcase_zscores = {}
        for feature in features:
            feature_val = testcase_freqs[feature]
            feature_mean = corpus_features[feature]["Mean"]
            feature_stdev = corpus_features[feature]["StdDev"]
            testcase_zscores[feature] = (feature_val - feature_mean) / feature_stdev
        label = ''
        score = 10000
        for deception in deceptions:
            delta = 0
            for feature in features:
                delta += math.fabs((testcase_zscores[feature] -
                                    feature_zscores[deception][feature]))
            delta /= len(features)
            # print("Delta score for candidate", deception, "is", delta)
            if delta < score:
                label = deception
                score = delta
        if label == "truthful":
            counter += 1

    print("Truthful test: {} %".format(counter))


