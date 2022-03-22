import nltk
import math
import pandas as pd

from helpers.ott_helper import get_ott_negative_deceptive, get_ott_negative_truthful
from helpers.text_helper import preprocess_text


if __name__ == '__main__':
    negative_deceptive = get_ott_negative_deceptive()['text'].apply(lambda x: preprocess_text(x))[:300]
    negative_truthful = get_ott_negative_truthful()['text'].apply(lambda x: preprocess_text(x))[:300]

    negative_deceptive_test = get_ott_negative_deceptive()['text'].apply(lambda x: preprocess_text(x))[300:]
    negative_truthful_test = get_ott_negative_truthful()['text'].apply(lambda x: preprocess_text(x))[300:]

    negative_deceptive_tokens = nltk.word_tokenize(' '.join(negative_deceptive))
    negative_truthful_tokens = nltk.word_tokenize(' '.join(negative_truthful))

    negative_deceptive_test_tokens = nltk.word_tokenize(' '.join(negative_deceptive_test))
    negative_truthful_test_tokens = nltk.word_tokenize(' '.join(negative_truthful_test))

    combined_negative_tokens = negative_deceptive_tokens + negative_truthful_tokens

    combined_frequencies = list(nltk.FreqDist(combined_negative_tokens).most_common(30))

    features = pd.DataFrame()
    features['feature'] = [feature for feature, frequency in combined_frequencies]
    features['deceptive_frequency'] = features['feature'].apply(
        lambda x: negative_deceptive_tokens.count(x)/len(negative_deceptive_tokens))
    features['truthful_frequency'] = features['feature'].apply(
        lambda x: negative_truthful_tokens.count(x)/len(negative_truthful_tokens))
    features['mean'] = (features['deceptive_frequency'] + features['truthful_frequency'])/2
    features['std'] = (features['deceptive_frequency'] - features['mean'])**2 + \
                      (features['truthful_frequency'] - features['mean'])**2
    features['std'] = features['std'].apply(lambda x: math.sqrt(x))

    features['deceptive_z_score'] = features['deceptive_frequency'] - features['mean']
    features['deceptive_z_score'] = features['deceptive_z_score'] / features['std']
    features['truthful_z_score'] = features['truthful_frequency'] - features['mean']
    features['truthful_z_score'] = features['truthful_z_score'] / features['std']

    features['deceptive_test_frequency'] = features['feature'].apply(
        lambda x: negative_deceptive_test_tokens.count(x) / len(negative_deceptive_test_tokens))
    features['deceptive_test_z_score'] = features['deceptive_test_frequency'] - features['mean']
    features['deceptive_test_z_score'] = features['deceptive_test_z_score'] / features['std']

    features['truthful_test_frequency'] = features['feature'].apply(
        lambda x: negative_truthful_test_tokens.count(x) / len(negative_truthful_test_tokens))
    features['truthful_test_z_score'] = features['truthful_test_frequency'] - features['mean']
    features['truthful_test_z_score'] = features['truthful_test_z_score'] / features['std']

    features.to_csv('local_datasets/features_negative.csv', index=False)

    print("Testing Deceptive Reviews...")
    delta = 0
    for _, row in features.iterrows():
        delta += math.fabs(row['deceptive_test_z_score'] - row['deceptive_z_score'])
    print("Delta score for deceptive is", delta/len(features['feature']))

    delta = 0
    for _, row in features.iterrows():
        delta += math.fabs(row['deceptive_test_z_score'] - row['truthful_z_score'])
    print("Delta score for truthful is", delta / len(features['feature']))

    print('#################')

    print("Testing Truthful Reviews...")
    delta = 0
    for _, row in features.iterrows():
        delta += math.fabs(row['truthful_test_z_score'] - row['deceptive_z_score'])
    print("Delta score for deceptive is", delta/len(features['feature']))

    delta = 0
    for _, row in features.iterrows():
        delta += math.fabs(row['truthful_test_z_score'] - row['truthful_z_score'])
    print("Delta score for truthful is", delta / len(features['feature']))
