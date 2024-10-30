# %%
import io
import random
from nltk.corpus import wordnet as wn
import nltk
import random


# %%

class Phrase:
    def __init__(self, words):
        self.words = words.split() if isinstance(words, str) else words
        self.word1, self.word2 = self.words

    def __str__(self):
        return f"{self.word1} {self.word2}"

    def __repr__(self):
        return f"Phrase({self.word1} {self.word2})"

    def __eq__(self, other):
        return (self.word1 == other.word1 and self.word2 == other.word2) or \
               (self.word1 == other.word2 and self.word2 == other.word1)
    
    def __lt__(self, other):
        return str(self) < str(other)

    def __getitem__(self, item):
        return self.words[item]

    def __eq__(self, other):
        return (self.word1 == other.word1 and self.word2 == other.word2)

    def __hash__(self):
        return hash(self.word1) + hash(self.word2)

def load_easy_upmas(source_path='data.txt'):
    with io.open(source_path, 'r', encoding='utf-8') as f:
        data = [l.split('//') for l in f.read().splitlines()]
    data = [[*map(str.strip, l)] for l in data]
    word_counts = []
    easy_upmas = []
    for l in data:
        this_word_count = tuple(len(phrase.split()) for phrase in l)
        if this_word_count == (2, 2):
            easy_upmas.append(tuple(Phrase(phrase.split()) for phrase in l))
        word_counts.append(this_word_count)

    print(f"Making {len(easy_upmas)} easy upma pairs from {len(data)} total upmas")
    return easy_upmas


def make_control_scrambled_aabb(easy_upmas):
    control = []
    for ph1, orig_ph2 in easy_upmas:
        while True:
            ph2 = random.choice(easy_upmas)[1]
            if ph2 != orig_ph2:
                break
        control.append((ph1, ph2))
    return control

def lemmas(word):
        result = set(lemma.name() for synset in wn.synsets(word) for lemma in synset.lemmas())
        return result

def make_control_syn(upmas):
    nltk.download('wordnet')
    control = []
    for ph1, ph2 in upmas:
        new_ph2 = []
        for i in range(2):
            w_lemmas = lemmas(ph1[i])
            if len(w_lemmas) < 2:
                print(f"Not enough lemmas, skipping {ph1} due to {ph1[i]}")
                break
            while True:
                new_w = random.choice(list(w_lemmas)).replace('_', ' ')
                if new_w != ph1[i]:
                    new_ph2.append(new_w)
                    break
        if len(new_ph2) < 2:
            continue
        new_ph2 = Phrase(new_ph2)
        control.append((ph1, new_ph2))
    return control


token_rates = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.600,
}


