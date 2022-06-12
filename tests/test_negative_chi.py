import nltk
import math
import pandas as pd

from helpers.ott_helper import get_ott_negative_deceptive, get_ott_negative_truthful
from helpers.text_helper import preprocess_text

if __name__ == '__main__':
    reviews_by_deception = {'deceptive': ' '.join(
        get_ott_negative_deceptive()['text'].apply(lambda x: preprocess_text(x))[:300]), 'truthful': ' '.join(
        get_ott_negative_truthful()['text'].apply(lambda x: preprocess_text(x))[:300])}

    deceptions = ['deceptive', 'truthful']
    reviews_by_deception_tokens = {}
    for deception in deceptions:
        tokens = nltk.word_tokenize(reviews_by_deception[deception])

        # Filter out punctuation
        reviews_by_deception_tokens[deception] = ([token for token in tokens if any(c.isalpha() for c in token)])

    for deception in deceptions:
        reviews_by_deception_tokens[deception] = (
            [token.lower() for token in reviews_by_deception_tokens[deception]])

    deceptive_test = get_ott_negative_deceptive()['text'].apply(lambda x: preprocess_text(x))[300:]
    count = 0

    reviews_by_deception_tokens["test"] = [token.lower() for token in nltk.word_tokenize(" ".join(deceptive_test))]

    for deception in deceptions:

        # First, build a joint corpus and identify the 500 most frequent words in it
        joint_corpus = (reviews_by_deception_tokens[deception] +
                        reviews_by_deception_tokens["test"])
        joint_freq_dist = nltk.FreqDist(joint_corpus)
        most_common = list(joint_freq_dist.most_common(20))

        deception_share = (len(reviews_by_deception_tokens[deception]) / len(joint_corpus))

        chisquared = 0
        for word, joint_count in most_common:
            # How often do we really see this common word?
            deception_count = reviews_by_deception_tokens[deception].count(word)
            test_count = reviews_by_deception_tokens["test"].count(word)

            # How often should we see it?
            expected_deception_count = joint_count * deception_share
            expected_test_count = joint_count * (1 - deception_share)

            # Add the word's contribution to the chi-squared statistic
            chisquared += ((deception_count - expected_deception_count) *
                           (deception_count - expected_deception_count) /
                           expected_deception_count)

            chisquared += ((test_count - expected_deception_count) *
                           (test_count - expected_deception_count)
                           / expected_test_count)

        print("The Chi-squared statistic for candidate", deception, "is", chisquared)
