# %%

from nltk.corpus import wordnet as wn
from util import Phrase
import os
import pandas as pd
import csv


wordnet_phrases = []
for lemma in wn.all_lemma_names():
    if len(lemma.split('_')) == 2:
        wordnet_phrases.append(Phrase(lemma.split('_')))

print(f"Added {len(wordnet_phrases)} phrases from WordNet")

# %%

import kagglehub

# Download latest version
path = kagglehub.dataset_download("therohk/urban-dictionary-words-dataset")

filename = "urbandict-word-defs.csv"
filepath = os.path.join(path, filename)

try:
    df = pd.read_csv(
        filepath,
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines='skip',  # Skips lines with too many fields
        encoding='utf-8'
    )
    print(f"Loaded {len(df)} rows from {filename}")
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")

# %%
two_words = df.word.str.replace('-', ' ').str.count(' ') == 1

ud_df = df[two_words & (df.up_votes > 100)]
print(f"Keeping {len(ud_df)} definitions with 2 words and >100 upvotes")
urban_dictionary_phrases = set(ud_df.word.str.replace('-', ' ').str.lower())
# filter to actually have 2 words
urban_dictionary_phrases = [Phrase(phrase.split()) for phrase in urban_dictionary_phrases if len(phrase.split()) == 2]
print(f"{len(urban_dictionary_phrases)} unique words in Urban Dictionary")

# %%        

# Overlap between WordNet and Urban Dictionary
overlap = set(wordnet_phrases) & set(urban_dictionary_phrases)
print(f"Overlap: {len(overlap)}")

union = set(wordnet_phrases) | set(urban_dictionary_phrases)
print(f"Union: {len(union)}")

# %%

# Load phrases from lots_of_phrases.log
with open("lots_of_phrases.log") as f:
    content = f.read()

gpt4o_mini_phrases = set(content.splitlines())
gpt4o_mini_phrases = [Phrase(phrase.split()) for phrase in gpt4o_mini_phrases if len(phrase.split()) == 2]
print(f"Loaded {len(gpt4o_mini_phrases)} phrases from lots_of_phrases.log")

triple_union = set(wordnet_phrases) | set(urban_dictionary_phrases) | set(gpt4o_mini_phrases)
print(f"Triple union: {len(triple_union)}")