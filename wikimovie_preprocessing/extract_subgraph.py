"""Script to extract subgraph for text given entities in the text.

A random walk is performed with the entities mentioned in the text as seeds
over the multigraph composed of both entities and relations in the KB as
nodes.

The extracted subgraph is stored as a list of tuples.
"""

import sys

data_path = sys.argv[1]
out_path = sys.argv[2]
split = sys.argv[3]

linked_text_json = out_path + "/wiki-entities_linked_qa_" + split + ".json"
subgraph_json = out_path + "/wiki-entities_subgraph_qa_" + split + ".json"

kb_file = out_path + "/processed_kb.pkl"
relation_file = out_path + "/relation_vocab.txt"

import json
import cPickle as pkl
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.preprocessing import normalize

MAX_ITER = 20
GAMMA = 0.5
INCLUDE = 0.005
MAX_ENT = 50
SELECT = "fixed"

def _create_multigraph():
    """Returns a dict mapping nodes to ids, a dict mapping relations
    to adjacency matrices and a sparse adjacency matrix
    of the multigraph."""
    kb_e, kb_r = pkl.load(open(kb_file, "rb"))
    print("%d nodes in multigraph" % len(kb_e))
    print("creating multigraph...")
    row_ones, col_ones = [], []
    for relation in kb_r.keys():
        kb_r["inv_" + relation] = kb_r[relation].transpose()
        row_idx, col_idx = kb_r[relation].nonzero()
        for ii in range(row_idx.shape[0]):
            head, tail = row_idx[ii], col_idx[ii]
            row_ones.append(head)
            col_ones.append(tail)
            row_ones.append(tail)
            col_ones.append(head)
    sp_mat = csr_matrix(
        (np.ones((len(row_ones),)), (np.array(row_ones), np.array(col_ones))),
        shape=(len(kb_e), len(kb_e)))
    print("done")
    return kb_e, kb_r, normalize(sp_mat, norm="l1", axis=1)

def _personalized_pagerank(seed, W, restart_prob):
    """Return the PPR vector for the given seed and restart prob.

    Args:
        seed: A sparse matrix of size E x 1.
        W: A sparse matrix of size E x E whose rows sum to one.
        restart_prob: A scalar in [0, 1].

    Returns:
        ppr: A vector of size E.
    """
    r = restart_prob * seed
    s = np.copy(r)
    for i in range(MAX_ITER):
        r_new = (1. - restart_prob) * (W.transpose().dot(r))
        s = s + r_new
        delta = abs(r_new.sum())
        if delta < 1e-5: break
        r = r_new
    return np.squeeze(s)

def _get_subgraph(entities, kb_r, multigraph_W):
    """Get subgraph describing a neighbourhood around given entities."""
    seed = np.zeros((multigraph_W.shape[0], 1))
    seed[entities] = 1. / len(set(entities))
    ppr = _personalized_pagerank(seed, multigraph_W, 0.8)
    if SELECT == "fixed":
        sorted_idx = np.argsort(ppr)[::-1]
        extracted_ents = sorted_idx[:MAX_ENT]
        # check if any ppr values are nearly zero
        zero_idx = np.where(ppr[extracted_ents] < 1e-6)[0]
        if zero_idx.shape[0] > 0:
            extracted_ents = extracted_ents[:zero_idx[0]]
    else:
        extracted_ents = np.where(ppr > INCLUDE)[0]
    extracted_tuples = []
    for relation in kb_r:
        if relation.startswith("inv"): continue
        submat = kb_r[relation][extracted_ents, :]
        submat = submat[:, extracted_ents]
        row_idx, col_idx = submat.nonzero()
        for ii in range(row_idx.shape[0]):
            extracted_tuples.append(
                (extracted_ents[row_idx[ii]], relation,
                 extracted_ents[col_idx[ii]]))
    return extracted_tuples, extracted_ents

def _convert_to_readable(tuples, inv_map, rel_map):
    readable_tuples = []
    for tup in tuples:
        readable_tuples.append([
            {"kb_id": int(tup[0]), "text": inv_map[tup[0]]},
            {"rel_id": rel_map[tup[1]], "text": tup[1]},
            {"kb_id": int(tup[2]), "text": inv_map[tup[2]]},
        ])
    return readable_tuples

def _readable_entities(entities, inv_map):
    readable_entities = []
    for ent in entities:
        readable_entities.append(
            {"text": inv_map[ent], "kb_id": ent})
    return readable_entities

def _get_answer_coverage(answers, entities):
    found, total = 0., 0
    all_entities = set(entities)
    for answer in answers:
        if answer["kb_id"] in all_entities: found += 1.
        total += 1
    return found / total

if __name__ == "__main__":
    kb_e, kb_r, multi_W = _create_multigraph()

    inv_map = {i: k for k, i in kb_e.iteritems()}

    relations = open(relation_file).read().splitlines()
    relations_map = {r: i for (i,r) in enumerate(relations)}

    answer_coverage, total = 0.0, 0
    with open(linked_text_json, "rb") as fin, open(subgraph_json, "wb") as fout:
        for line in tqdm(fin):
            data = json.loads(line)
            entities = [ent["kb_id"] for ent in data["entities"]]
            if not entities:
                extracted_tuples = []
                extracted_ents = []
            else:
                extracted_tuples, extracted_ents = _get_subgraph(
                    entities, kb_r, multi_W)
            data["subgraph"] = {}
            data["subgraph"]["tuples"] = _convert_to_readable(
                extracted_tuples, inv_map, relations_map)
            data["subgraph"]["entities"] = _readable_entities(
                extracted_ents, inv_map)
            if "answers" in data:
                if not data["answers"]: continue
                current_coverage = _get_answer_coverage(data["answers"], extracted_ents)
                #if current_coverage < 1.:
                #    import pdb
                #    pdb.set_trace()
                answer_coverage += current_coverage
            total += 1
            fout.write(json.dumps(data) + "\n")
    print("Answer coverage in retrieved subgraphs = %.3f" % (answer_coverage / total))
