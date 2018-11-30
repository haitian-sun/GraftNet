"""Script to compute relation embeddings for each relation in given list."""

relations_file = "freebase_2hops/relations"
embeddings_file = "glove"
output_file = "scratch/relation_emb.pkl"
dim = 300

import cPickle as pkl
import numpy as np
from tqdm import tqdm

word_to_relation = {}
relation_lens = {}
def _add_word(word, t, v):
    if word not in word_to_relation: word_to_relation[word] = []
    word_to_relation[word].append((v, t))
    if v not in relation_lens: relation_lens[v] = 0
    relation_lens[v] += t

with open(relations_file) as f:
    for ii, line in enumerate(f):
        relation = line.strip()
        domain, typ, prop = relation[4:-1].split(".")[-3:]
        for word in domain.split("_"):
            _add_word(word, 1, relation)
        for word in typ.split("_"):
            _add_word(word, 2, relation)
        for word in prop.split("_"):
            _add_word(word, 3, relation)

relation_emb = {r: np.zeros((dim,)) for r in relation_lens}
with open(embeddings_file) as f:
    for line in tqdm(f):
        word, vec = line.strip().split(None, 1)
        if word in word_to_relation:
            for rid, typ in word_to_relation[word]:
                relation_emb[rid] += typ * np.array(
                    [float(vv) for vv in vec.split()])

for relation in relation_emb:
    relation_emb[relation] = relation_emb[relation] / relation_lens[relation]

pkl.dump(relation_emb, open(output_file, "w"))
