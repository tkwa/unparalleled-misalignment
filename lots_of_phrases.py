# %%

"""
Strategy: generate lots of phrases, use nltk to get their synonyms,
and see if any combinations are synonyms of each other.

"""

import io
import random
from nltk.corpus import wordnet as wn
import key
import nltk
import random
from tqdm.notebook import tqdm
from util import Phrase, token_rates
from openai import OpenAI
from typing import Iterable
import string

import concurrent.futures
import re
import time

# %%

synsets_cache = {}

def is_synonym(word1: str, word2: str) -> bool:
    """
    Caches synsets of each word in `synsets_cache` and returns whether two words share a synset
    """
    if word1 not in synsets_cache:
        synsets_cache[word1] = set(wn.synsets(word1))
    if word2 not in synsets_cache:
        synsets_cache[word2] = set(wn.synsets(word2))
    synsets1 = synsets_cache[word1]
    synsets2 = synsets_cache[word2]
    return bool(synsets1 & synsets2)

is_synonym('dog', 'hound')

# %%

def is_parallel(ph1: Phrase, ph2: Phrase) -> bool:
    """
    Returns whether two phrases are parallel. Symmetric but not transitive.
    """
    return is_synonym(ph1.word1, ph2.word1) and is_synonym(ph1.word2, ph2.word2)

phrases_history = set()
# %%
client = OpenAI()

def query_model(prompt, model='gpt-4o-mini', verbose=False):
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    response = client.chat.completions.create(model=model, messages=messages)
    usage = response.usage.total_tokens
    token_rate = token_rates[model]
    if verbose: print(f"Used {usage} tokens, cost est. ${usage/1e6*token_rate:.6f}")
    return response


def process_content(content: str):
    results = []
    for line in content.splitlines():
        words = line.split()
        words = [word.lower() for word in line.split() if word.isalpha()]
        if len(words) == 2:
            results.append(Phrase(words))
    return results

generate_phrases_prompt = """What are {batch_size} two-word phrases starting with the letter {letter}? Include some slang, obscene, and vulgar phrases, names, and phrases containing uncommon words"""
idioms_prompt = """List {batch_size} two-word idioms starting with {letter} whose common meaning is different from the separate meanings of the words."""
obscene_prompt = """List {batch_size} two-word vulgar or obscene idioms, like "{obscene_example}"."""

def make_prompt(batch_size: int):
    prompt_type_choices = {'common': 0.4, 'idiom': 0.4, 'obscene': 0.2}
    prompt_type = random.choices(list(prompt_type_choices.keys()), weights=list(prompt_type_choices.values()))[0]
    match prompt_type:
        case 'common':
            starting_letter = random.choice(string.ascii_uppercase)
            prompt = generate_phrases_prompt.format(batch_size=batch_size, letter=starting_letter)
        case 'idiom':
            starting_letter = random.choice(string.ascii_uppercase)
            prompt = idioms_prompt.format(batch_size=batch_size, letter=starting_letter)
        case 'obscene':
            obscene_example = random.choice([
                'road head',
                'hand job',
                'drug dealer'
            ])
            prompt = obscene_prompt.format(obscene_example=obscene_example, batch_size=batch_size)
        case _:
            raise ValueError(f"Unknown prompt type {prompt_type}")

    return prompt

def generate_phrases_one_call(batch_size: int, model='gpt-4o-mini') -> list[Phrase]:
    prompt = make_prompt(batch_size)
    response = query_model(prompt, model=model)
    content = response.choices[0].message.content
    usage = response.usage.total_tokens
    phrases = process_content(content)
    return phrases, usage

import concurrent.futures
from tqdm import tqdm

def generate_phrases(n: int, batch_size: int = 100, model='gpt-4o-mini', log_path=None) -> list[Phrase]:
    phrases = set()
    total_usage = 0
    max_workers = 10  # Adjust the number of workers as needed

    def task():
        return generate_phrases_one_call(batch_size, model=model)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start with initial batch of calls
        num_initial_calls = n // batch_size
        futures = {executor.submit(task) for _ in range(num_initial_calls)}
        with tqdm(total=n) as t:
            while len(phrases) < n:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    futures.remove(fut)
                    new_phrases, usage = fut.result()
                    total_usage += usage
                    old_len = len(phrases)
                    phrases.update(new_phrases)
                    new_len = len(phrases)
                    t.update(new_len - old_len)
                    t.set_postfix({'phrases': len(phrases), 'toks': total_usage, '$': total_usage / 1e6 * token_rates[model]})
                    if log_path:
                        with open(log_path, 'a') as f:
                            for ph in new_phrases:
                                f.write(f"{ph}\n")

                    max_potential_phrases = len(phrases) + len(futures) * batch_size
                    if max_potential_phrases < n:
                        for _ in range((n - max_potential_phrases) // batch_size):
                            futures.add(executor.submit(task))
            # Cancel any remaining futures
            for fut in futures:
                fut.cancel()
    return list(phrases), total_usage

all_phrases, usage = generate_phrases(10000, 200, model='gpt-4o-mini', log_path='lots_of_phrases.log')
phrases_history.update(all_phrases)
# %%

def find_parallel_phrases(phrases: Iterable[Phrase]) -> list[set[Phrase]]:
    """
    Returns a list of tuples of parallel phrases of size at least 2
    This is O(n^2)
    For large n, would be better to keep a dict of synset pairs and check all synset pairs consistent with each phrase
    """
    result = set()
    for ph1 in tqdm(phrases):
        parallel = set()
        for ph2 in phrases:
            if is_parallel(ph1, ph2) and ph1.word1 != ph2.word1 and ph1.word2 != ph2.word2:
                parallel.add(ph2)
        if parallel:
            parallel.add(ph1)
            result.add(tuple(sorted(tuple(parallel))))

    print(f"Found {len(result)} sets of parallel phrases")
    return result

parallel_phrases = find_parallel_phrases(phrases_history)


# %%
parallel_phrases