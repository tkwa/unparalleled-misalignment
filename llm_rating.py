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
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


from openai import OpenAI
client = OpenAI()


# %%

easy_upmas = load_easy_upmas('data.txt')
control_aabb = make_control_scrambled_aabb(easy_upmas)
control_syn = make_control_syn(easy_upmas)

# make dataframe of upmas. type is 'upma' or 'control_...'
upma_df = pd.DataFrame(easy_upmas + control_aabb + control_syn, columns=['ph0', 'ph1'])
upma_df['type'] = ['upma']*len(easy_upmas) + ['control_aabb']*len(control_aabb) + ['control_syn']*len(control_syn)
upma_df['is_upma'] = upma_df['type'] == 'upma'

all_upmas = easy_upmas + control_aabb + control_syn

# %%

def query_model(upma, prompt_fn, model='gpt-4o-mini', verbose=False):
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

# %%
def response_to_score(response):
    content = response.choices[0].message.content
    last_line = content.splitlines()[-1]
    return int(re.search(r"(\d+)", last_line).group(1))

# %%

syn_prompt = """
Do "{w1}" and "{w2}" share a meaning (potentially including slang or vulgar meanings, but opposites get 0 points)?
Rate the similiarity of their most similar meaning out of 10. Be concise. On the last line, write the score like:
Rating: X"""

good_phrase_prompt = """Is "{p}" a recognized phrase (proper names, jargon or niche uses okay)? Rate it out of 10, and be concise. On the last line, write the score like:
Rating: X"""

different_meaning_prompt = """Do "{p1}" and "{p2}" have a similar meaning? Rate their similarity out of 10, and be concise. On the last line, write the score like:
Rating: X"""

def query_model_multiple(upma, model='gpt-4o-mini', verbose=False):
    responses = {}
    responses['syn_0'] = query_model(upma, prompt_fn=lambda upma: syn_prompt.format(w1=upma[0][0], w2=upma[1][0]), model=model, verbose=verbose)
    responses['syn_1'] = query_model(upma, prompt_fn=lambda upma: syn_prompt.format(w1=upma[0][1], w2=upma[1][1]), model=model, verbose=verbose)
    responses['phrase_0'] = query_model(upma, prompt_fn=lambda upma: good_phrase_prompt.format(p=upma[0]), model=model, verbose=verbose)
    responses['phrase_1'] = query_model(upma, prompt_fn=lambda upma: good_phrase_prompt.format(p=upma[1]), model=model, verbose=verbose)
    responses['contrast'] = query_model(upma, prompt_fn=lambda upma: different_meaning_prompt.format(p1=upma[0], p2=upma[1]), model=model, verbose=verbose)
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
                    t.set_postfix({'toks': total_usage, '$': total_usage / 1e6 * token_rates[model]})

                except Exception as exc:
                    print(f'{upmas[index]} generated an exception: {exc}')

    print(f"Used {total_usage} tokens, cost est. ${total_usage/1e6*token_rate:.6f}")

    return scores, contents

scores, contents = query_model_all_parallel(all_upmas, model='gpt-4o-mini', log_path='responses.log')


# %%
scores_df = pd.DataFrame(scores)
scored_df = pd.concat([upma_df, scores_df], axis=1)
scored_df['content'] = contents

# write to file
scored_df.to_csv('scored_df.csv', index=False)

# %%
from sklearn.metrics import roc_auc_score

y_true = scored_df['is_upma']
scored_df['sum_score'] = sum([scored_df[row_name] for row_name in
                              ['syn_0', 'syn_1', 'phrase_0', 'phrase_1', 'contrast']])
y_scores = [score if isinstance(score, int) else 0 for score in scored_df['sum_score']]
auc = roc_auc_score(y_true, y_scores)
print(f"AUC (unweighted): {auc:.3f}")

# %%

# Now train a linear classifier on subscores
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
# save classifier to file
with open('classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)


# %%

# Add a new column for weighted score, weighted by coefficients

scored_df['weighted_score'] = clf.predict_proba(X)[:, 1]

sns.histplot(data=scored_df, x='weighted_score', hue='type', bins=10, multiple='layer')
plt.title("UPMA scores by type, linear classifier weights")
plt.show()

auc_weighted = roc_auc_score(y, clf.predict_proba(X)[:, 1])
print(f"AUC (weighted): {auc_weighted:.3f}")

# %%

def evaluate_candidates(candidates, model='gpt-4o-mini'):
    candidates_df = pd.DataFrame(candidates, columns=['ph0', 'ph1'])
    scores, contents = query_model_all_parallel(candidates, model=model)
    scores_df = pd.DataFrame(scores)
    scored_df = pd.concat([candidates_df, scores_df], axis=1)
    scored_df['content'] = contents
    scored_df['weighted_score'] = clf.predict_proba(scored_df[['syn_0', 'syn_1', 'phrase_0', 'phrase_1', 'contrast']])[:, 1]
    return scored_df

parallel_phrases = pickle.load(open('parallel_phrases.pkl', 'rb'))

# only keep first match for each key for now
candidates = [(k, list(v)[0]) for k, v in parallel_phrases.items()]
candidate_df = evaluate_candidates(candidates[:1000])
candidate_df

# %%

with open('candidate_df.pkl', 'wb') as f:
    pickle.dump(candidate_df, f)

# %%

# Now we can sort by weighted score to get the best candidates
best_candidates = candidate_df.sort_values('weighted_score', ascending=False)
best_candidates.head(50)