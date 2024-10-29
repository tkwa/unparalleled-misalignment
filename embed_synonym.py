# %%

from util import load_easy_upmas, make_control_scrambled_aabb, Phrase, make_control_syn
import key
from tqdm.notebook import tqdm
import itertools
from collections import Counter
import pandas as pd
import numpy as np
import re


easy_upmas = load_easy_upmas('data.txt')
control_aabb = make_control_scrambled_aabb(easy_upmas)
control_syn = make_control_syn(easy_upmas)


# %%

# get repeated phrases
phrases = itertools.chain.from_iterable(easy_upmas)
Counter(phrases).most_common(5)

# %%

from openai import OpenAI
client = OpenAI()

def get_embeddings(text, model="text-embedding-3-small"):
   return [np.array(e.embedding) for e in client.embeddings.create(input = text, model=model).data]

all_embeddings = dict()

all_upmas = easy_upmas + control_aabb + control_syn
all_phrases = set(ph for upma in all_upmas for ph in upma)

all_words = set(w for ph in all_phrases for w in ph)

all_embeddable = list(all_phrases | all_words)

batch_size = 256
model = "text-embedding-3-large"
usage = 0
for i in tqdm(range(0, len(all_embeddable), batch_size)):
    batch = all_embeddable[i:i+batch_size]
    batch_str = [str(s) for s in batch]
    embedding_response = client.embeddings.create(input = batch_str, model=model)
    embeddings = [np.array(e.embedding) for e in embedding_response.data]
    usage += embedding_response.usage.total_tokens
    all_embeddings.update({s: emb for s, emb in zip(batch, embeddings)})

token_rate = 0.130 if model == "text-embedding-3-large" else 0.020
print(f"Used {usage} tokens; cost est. ${usage/1e6*token_rate:.6f}")

# %%


def cosine_similarity(v1, v2):
    return np.einsum('... d, ... d -> ...', v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# make dataframe of upmas. type is 'upma' or 'control_...'
upma_df = pd.DataFrame(easy_upmas + control_aabb + control_syn, columns=['ph0', 'ph1'])
upma_df['type'] = ['upma']*len(easy_upmas) + ['control_aabb']*len(control_aabb) + ['control_syn']*len(control_syn)

all_upmas = easy_upmas + control_aabb + control_syn
ph0_ph1_cossim = [cosine_similarity(all_embeddings[ph0], all_embeddings[ph1]) for ph0, ph1 in all_upmas]
w00_w10_cossim = [cosine_similarity(all_embeddings[ph0[0]], all_embeddings[ph1[0]]) for ph0, ph1 in all_upmas]
w01_w11_cossim = [cosine_similarity(all_embeddings[ph0[1]], all_embeddings[ph1[1]]) for ph0, ph1 in all_upmas]
upma_df['ph0_ph1_cossim'] = ph0_ph1_cossim
upma_df['w00_w10_cossim'] = w00_w10_cossim
upma_df['w01_w11_cossim'] = w01_w11_cossim
upma_df['w0x_w1x_cossim'] = (np.array(w00_w10_cossim) + np.array(w01_w11_cossim))/2

upma_df

# %%

print("Similarity between phrases in each upma, by type:")
print(upma_df.groupby('type')['ph0_ph1_cossim'].mean())
print("Similarity between first words in each upma, by type:")
print(upma_df.groupby('type')['w00_w10_cossim'].mean())
print("Similarity between second words in each upma, by type:")
print(upma_df.groupby('type')['w01_w11_cossim'].mean())

# %%

# Make a scatterplot of ph0_ph1_cossim vs w0x_w1x_cossim, colored by type
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=upma_df, x='ph0_ph1_cossim', y='w0x_w1x_cossim', hue='type')

# We thought that phrase similarity would be lower for upmas than controls (if word sim is equal), but it's not.
# Embeddings cannot distinguish between upmas and controls. 

# %%

prompt = """A UPMA is a pair of two word phrases such that:
* The first words in each phrase share a meaning (potentially including slang or vulgar meanings, but opposites get 0 points) (0-5)
* The second words in each phrase share a meaning (0-5)
* The first phrase is a recognized phrase (collocation or idiom, jargon or niche uses okay) (0-5)
* The second phrase is a recognized phrase (0-5)
* The two whole phrases have dramatically different meanings (0-10)

For the following candidate UPMA, rate it on all these metrics, then add up the subscores to get a UPMA score out of 30.
On the last line, write the final score.

{p1} // {p2}
"""

token_rates = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.600,
}

upma1 = ("hot dog", Phrase("cold cat"))

prompt_for_upma1 = prompt.format(p1=upma1[0], p2=upma1[1])

prompt_fn = lambda upma: prompt.format(p1=upma[0], p2=upma[1])
def query_model(upma, model='gpt-4o-mini', prompt_fn=prompt_fn, verbose=False):
    prompt = prompt_fn(upma)
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    response = client.chat.completions.create(model=model, messages=messages)
    usage = response.usage.total_tokens
    token_rate = token_rates[model]
    if verbose: print(f"Used {usage} tokens, cost est. ${usage/1e6*token_rate:.6f}")
    return response
response = query_model(upma1)

# %%
def response_to_score(response):
    content = response.choices[0].message.content
    last_line = content.splitlines()[-1]
    return int(re.search(r"(\d+)", last_line).group(1))

content = response.choices[0].message.content
print(response_to_score(response))

# %%

syn_prompt = """
Do {w1} and {w2} share a meaning (potentially including slang or vulgar meanings, but opposites get 0 points)?
Rate the similiarity of their most similar meaning out of 10. Be concise. On the last line, write the score like:
Rating: X"""

good_phrase_prompt = """Is {p} a recognized phrase (proper names, jargon or niche uses okay)? Rate it out of 5, and be concise. On the last line, write the score like:
Rating: X"""

different_meaning_prompt = """Do {p1} and {p2} have dramatically different meanings? Rate it out of 10, and be concise. On the last line, write the score like:
Rating: X"""

def query_model_multiple(upma, model='gpt-4o-mini', verbose=False):
    responses = []
    for i, phrase in enumerate(upma):
        if i == 0:
            prompt = good_phrase_prompt.format(p=phrase)
        elif i == 1:
            prompt = syn_prompt.format(w1=phrase[0], w2=phrase[1])
        else:
            prompt = different_meaning_prompt.format(p1=upma[0], p2=upma[1])
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        response = client.chat.completions.create(model=model, messages=messages)
        responses.append(response)
        usage = response.usage.total_tokens
        token_rate = token_rates[model]
        if verbose: print(f"Used {usage} tokens, cost est. ${usage/1e6*token_rate:.6f}")
    return responses

# %%
import concurrent.futures
import re

def query_model_and_score(upma, model='gpt-4o-mini'):
    response = query_model(upma, model)
    content = response.choices[0].message.content
    score = response_to_score(response)
    return score, content, response.usage.total_tokens

def query_model_all_parallel(upmas, model='gpt-4o-mini', log_path=None):
    scores = [None] * len(upmas)
    contents = [None] * len(upmas)
    total_usage = 0
    token_rate = token_rates[model]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(query_model_and_score, upma, model): i for i, upma in enumerate(upmas)}

        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(upmas)):
            index = future_to_index[future]
            try:
                score, content, usage = future.result()
                scores[index] = score
                contents[index] = content
                total_usage += usage
                if log_path is not None:
                    with open(log_path, 'a') as f:
                        f.write(str(score) + '\n')
                        f.write(content + '\n\n')
            except Exception as exc:
                print(f'{upmas[index]} generated an exception: {exc}')

    print(f"Used {total_usage} tokens, cost est. ${total_usage/1e6*token_rate:.6f}")

    return scores, contents

scores, contents = query_model_all_parallel(all_upmas, model='gpt-4o-mini', log_path='responses.log')


# %%
upma_df['4o_mini'] = scores
upma_df['content'] = contents

# %%

print(upma_df.groupby('type')['gpt-4o-mini'].mean())


# Histogram of scores colored by type
sns.histplot(data=upma_df, x='gpt-4o-mini', hue='type', bins=10, multiple='layer')

# %%

# Get all rows where the score is at least 27
upma_df[upma_df['gpt-4o-mini'] >= 27]

# %%
from sklearn.metrics import roc_auc_score

y_true = [1] * len(easy_upmas) + [0] * len(control_aabb) + [0] * len(control_syn)
y_scores = [score if isinstance(score, int) else 0 for score in scores]
auc = roc_auc_score(y_true, y_scores)
print(f"AUC: {auc:.3f}")

# %%

c = upma_df[(upma_df.type == 'control_aabb') & (upma_df['gpt-4o-mini'] == 20)].content.values[0]
print(c)

# %%

c = upma_df[(upma_df.type == 'upma') & (upma_df['gpt-4o-mini'] <= 24)].content.values
for r in c:
    print(r)