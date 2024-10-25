import io
import random

# %%

def easy_upmas(source_path='data.txt'):
    with io.open('data.txt', 'r', encoding='utf-8') as f:
        data = [l.split('//') for l in f.read().splitlines()]
    data = [[*map(str.strip, l)] for l in data]
    word_counts = []
    easy_upmas = []
    for l in data:
        this_word_count = tuple(len(phrase.split()) for phrase in l)
        if this_word_count == (2, 2):
            easy_upmas.append(tuple(phrase.split() for phrase in l))
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