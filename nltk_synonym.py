# %%
import io
from collections import Counter
import random
import sklearn
import itertools
from tqdm.notebook import tqdm
import functools
from sklearn.metrics import roc_auc_score



from nltk.corpus import wordnet as wn
import nltk
from util import load_easy_upmas, make_control_scrambled_aabb

# Download required NLTK data (run once)
nltk.download('wordnet')


# %%

easy_upmas = load_easy_upmas('data.txt')

# Now create a control set by scrambling the phrases
control = make_control_scrambled_aabb(easy_upmas)
    

# %%


def check_synonyms(word1, word2):
    """
    Check if two words are synonyms using WordNet.
    Returns True if words are synonyms, False otherwise.
    """
    # Get all synsets for both words
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    
    # If either word isn't found in WordNet, return False
    if not synsets1 or not synsets2:
        return False
    
    # Get all lemmas for each word
    lemmas1 = set(lemma.name() for synset in synsets1 for lemma in synset.lemmas())
    lemmas2 = set(lemma.name() for synset in synsets2 for lemma in synset.lemmas())
    
    # Check if there's any overlap in the lemmas
    # print(lemmas1, lemmas2)
    intersection = lemmas1.intersection(lemmas2)
    # if intersection: print(intersection)
    return bool(intersection)

# Test the function
print(check_synonyms('dog', 'cat'))  # False
print(check_synonyms('trade', 'switch'))  # True

# %%

def syn_score_1(ph1, ph2):
    a = check_synonyms(ph1[0], ph2[0]) + check_synonyms(ph1[1], ph2[1])
    b = check_synonyms(ph1[0], ph2[1]) + check_synonyms(ph1[1], ph2[0])
    return max(a, b)

def word_similarity(word1, word2, metric=wn.path_similarity, use_morphy=True):
    """
    Compute the similarity between two words using WordNet.
    Returns a value between 0 and 1.
    """

    if use_morphy:
        word1 = wn.morphy(word1) or word1
        word2 = wn.morphy(word2) or word2
    # Get all synsets for both words
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    
    # If either word isn't found in WordNet, return 0
    if not synsets1 or not synsets2:
        return 0
    
    # Get the maximum similarity between all pairs of synsets
    if metric in [wn.path_similarity, wn.wup_similarity]:
        max_similarity = max(metric(s1, s2) or 0 for s1, s2 in itertools.product(synsets1, synsets2))
    else:
        try:
            values = []
            for s1, s2 in itertools.product(synsets1, synsets2):
                if s1.pos() == s2.pos():
                    values.append(metric(s1, s2))
            max_similarity = max(values) if values else 0
        except:
            max_similarity = 0
    return max_similarity

def syn_score_2(ph1, ph2, metric=wn.path_similarity, verbose=False):
    """
    use itertools.product to get pairs of synsets
    """
    if isinstance(ph1, str):
        ph1 = ph1.split()
    if isinstance(ph2, str):
        ph2 = ph2.split()
    s00 = word_similarity(ph1[0], ph2[0], metric)
    s01 = word_similarity(ph1[0], ph2[1], metric)
    s10 = word_similarity(ph1[1], ph2[0], metric)
    s11 = word_similarity(ph1[1], ph2[1], metric)
    if verbose: print(f"{s00:.3f} {s01:.3f} {s10:.3f} {s11:.3f}")
    return max(s00 + s01, s10 + s11)

# %%
from nltk.corpus import wordnet_ic
nltk.download('wordnet_ic')
brown_ic = wordnet_ic.ic('ic-brown.dat')

def compare_metrics(metrics, verbose=False):
    for metric in metrics:
        syn_score = functools.partial(syn_score_2, metric=metric)
        y_true = [1] * len(easy_upmas) + [0] * len(control)
        metric_name = metric.__name__ if hasattr(metric, '__name__') else metric.func.__name__
        try:
            y_scores = [syn_score(*pair) for pair in tqdm(easy_upmas + control)]
        except Exception as e:
            y_scores = [0] * len(y_true)
            print(f"Error with {metric_name}")
            raise e
        auc = roc_auc_score(y_true, y_scores)
        print(f"AUC: {auc:.3f} for {metric_name}")

        if verbose:
            print(f"Easy UPMA avg: {sum(y_scores[:len(easy_upmas)]) / len(easy_upmas):.3f}")
            print(f"Control avg: {sum(y_scores[len(easy_upmas):]) / len(control):.3f}")

metrics = [
    wn.path_similarity,
    wn.lch_similarity,
    wn.wup_similarity,
    functools.partial(wn.jcn_similarity, ic=brown_ic),
    functools.partial(wn.res_similarity, ic=brown_ic),
    functools.partial(wn.lin_similarity, ic=brown_ic),
]

if __name__ == "__main__":
    compare_metrics(metrics, verbose=True)

# path similarity does best