# %%

from util import load_easy_upmas, make_control_scrambled_aabb, Phrase, make_control_syn, token_rates
import key
from tqdm.notebook import tqdm
import itertools
from collections import Counter
import pandas as pd
import numpy as np
import re
from ratelimit import limits, sleep_and_retry

# %%


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
upma_df['is_upma'] = upma_df['type'] == 'upma'

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

good_phrase_prompt = """Is {p} a recognized phrase (proper names, jargon or niche uses okay)? Rate it out of 10, and be concise. On the last line, write the score like:
Rating: X"""

different_meaning_prompt = """Do {p1} and {p2} have dramatically different meanings? Rate it out of 10, and be concise. On the last line, write the score like:
Rating: X"""

def query_model_multiple(upma, model='gpt-4o-mini', verbose=False):
    responses = {}
    responses['syn_0'] = query_model(upma, model, prompt_fn=lambda upma: syn_prompt.format(w1=upma[0][0], w2=upma[1][0]), verbose=verbose)
    responses['syn_1'] = query_model(upma, model, prompt_fn=lambda upma: syn_prompt.format(w1=upma[0][1], w2=upma[1][1]), verbose=verbose)
    responses['phrase_0'] = query_model(upma, model, prompt_fn=lambda upma: good_phrase_prompt.format(p=upma[0]), verbose=verbose)
    responses['phrase_1'] = query_model(upma, model, prompt_fn=lambda upma: good_phrase_prompt.format(p=upma[1]), verbose=verbose)
    responses['contrast'] = query_model(upma, model, prompt_fn=lambda upma: different_meaning_prompt.format(p1=upma[0], p2=upma[1]), verbose=verbose)
    return responses

# %%
import concurrent.futures
import re
import time

def query_model_and_score(upma, model='gpt-4o-mini'):
    responses = query_model_multiple(upma, model)
    contents = {key:response.choices[0].message.content for key, response in responses.items()}
    scores = {
        key:(response_to_score(response))
        for key, response in responses.items()
    }
    usage = sum(r.usage.total_tokens for r in responses.values())
    return scores, contents, usage

def query_model_all_parallel(upmas, model='gpt-4o-mini', log_path=None):
    scores = [None] * len(upmas)
    contents = [None] * len(upmas)
    total_usage = 0
    token_rate = token_rates[model]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(query_model_and_score, upma, model): i for i, upma in enumerate(upmas)}

        with tqdm(concurrent.futures.as_completed(future_to_index), total=len(upmas)) as t:
            for future in t:
                index = future_to_index[future]
                try:
                    score, content, usage = future.result()
                    scores[index] = score
                    contents[index] = content
                    total_usage += usage
                    if log_path is not None:
                        with open(log_path, 'a') as f:
                            f.write(str(score) + '\n')
                            for s in content.values():
                                f.write(s + '\n')
                            f.write('\n')
                    t.set_postfix({'phrases': len(phrases), 'toks': total_usage, '$': total_usage / 1e6 * token_rates[model]})

                except Exception as exc:
                    print(f'{upmas[index]} generated an exception: {exc}')

    print(f"Used {total_usage} tokens, cost est. ${total_usage/1e6*token_rate:.6f}")

    return scores, contents

scores, contents = query_model_all_parallel(all_upmas, model='gpt-4o-mini', log_path='responses.log')


# %%
scores_df = pd.DataFrame(scores)
scored_df = pd.concat([upma_df, scores_df], axis=1)
scored_df['content'] = contents
scored_df['sum_score'] = \
    scored_df.syn_0 + scored_df.syn_1 + scored_df.phrase_0 + scored_df.phrase_1 + scored_df.contrast

# write to file
scored_df.to_csv('scored_df.csv', index=False)

# %%

print(scored_df.groupby('type')['sum_score'].mean())


# Histogram of scores colored by type
sns.histplot(data=scored_df, x='sum_score', hue='type', bins=10, multiple='layer')
plt.title("UPMA scores by type, equal weights")
plt.show()
# %%

# %%

# Get all rows where the score is at least 27
scored_df[scored_df['sum_score'] >= 31]

# %%
from sklearn.metrics import roc_auc_score

y_true = scored_df['is_upma']
y_scores = [score if isinstance(score, int) else 0 for score in scored_df['sum_score']]
auc = roc_auc_score(y_true, y_scores)
print(f"AUC (unweighted): {auc:.3f}")

# %%

# Now train a linear classifier on subscores
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = scored_df[['syn_0', 'syn_1', 'phrase_0', 'phrase_1', 'contrast']]
y = scored_df['is_upma']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")

print(f"Weights: {clf.coef_}")



# %%

# Add a new column for weighted score, weighted by coefficients

scored_df['weighted_score'] = clf.predict_proba(X)[:, 1]

sns.histplot(data=scored_df, x='weighted_score', hue='type', bins=10, multiple='layer')
plt.title("UPMA scores by type, linear classifier weights")
plt.show()

auc_weighted = roc_auc_score(y, clf.predict_proba(X)[:, 1])
print(f"AUC (weighted): {auc_weighted:.3f}")

# %%

c = scored_df[(scored_df.is_upma == False) & (scored_df['weighted_score'] >= 0.8)]
c
# %%

for k, v in c.content[537].items():
    print(k, v)

# %%

# Get lowest score with is_upma == True, sorted by weighted_score
c = scored_df[scored_df.is_upma == True].sort_values('weighted_score')
c.head(20)

# %%
top20 = c.tail(20).iloc[::-1][['ph0', 'ph1', 'weighted_score', 'content']]
top20