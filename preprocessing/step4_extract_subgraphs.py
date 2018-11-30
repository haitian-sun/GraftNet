"""Script to run PPR on question subgraphs and retain top entities.

Usage:
    python step4_extract_subgraphs.py

`subgraph_dir` below contains question-wise subgraphs in files named
<questionId>.nxhd.

Specify the question json file and the list of seed entities for the
questions at the top of this script before running. The output is in the same
format as the input json but with an extra field `subgraph` which lists the
relations and entities associated with the question.
"""

question_json = "scratch/webqsp_processed.json"
output_json = "webqsp_subgraphs.json"

# fixed files
subgraph_dir = "freebase_2hops/stagg.neighborhoods/"
question_seeds = "scratch/stagg_linked_questions.txt"
question_emb = "scratch/webqsp_embeddings.pkl"
relation_emb = "scratch/relation_emb.pkl"

import os
import json
import random
import numpy as np
import cPickle as pkl
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

random.seed(0)

MAX_FACTS = 5000000
MAX_ITER = 20
RESTART = 0.8
MAX_ENT = 500
NOTFOUNDSCORE = 0.
EXPONENT = 2.
MAX_SEEDS = 1
DECOMPOSE_PPV = True
SEED_WEIGHTING = True
RELATION_WEIGHTING = True
FOLLOW_NONCVT = True
USEANSWER = False


def _filter_relation(relation):
    if relation == "<fb:common.topic.notable_types>": return False
    domain = relation[4:-1].split(".")[0]
    if domain == "type" or domain == "common": return True
    return False


def _read_facts(fact_file, relation_embeddings, question_embedding,
                seeds, qId):
    """Read all triples from the fact file and create a sparse adjacency
    matrix between the entities. Returns mapping of entities to their
    indices, a mapping of relations to the
    and the combined adjacency matrix."""
    seeds_found = set()
    with open(fact_file) as f:
        entity_map = {}
        relation_map = {}
        row_ones, col_ones = [], []
        num_entities = 0
        num_facts = 0
        for line in f:
            try:
                e1, rel, e2 = line.strip().split(None, 2)
            except ValueError:
                continue
            if _filter_relation(rel): continue
            if e1 not in entity_map:
                entity_map[e1] = num_entities
                num_entities += 1
            if e2 not in entity_map:
                entity_map[e2] = num_entities
                num_entities += 1
            if rel not in relation_map:
                relation_map[rel] = [[], []]
            if e1 in seeds: seeds_found.add(e1)
            if e2 in seeds: seeds_found.add(e2)
            row_ones.append(entity_map[e1])
            col_ones.append(entity_map[e2])
            row_ones.append(entity_map[e2])
            col_ones.append(entity_map[e1])
            relation_map[rel][0].append(entity_map[e1])
            relation_map[rel][1].append(entity_map[e2])
            num_facts += 1
            if num_facts == MAX_FACTS:
                break
    if not relation_map:
        return {}, {}, None
    for rel in relation_map:
        row_ones, col_ones = relation_map[rel]
        m = csr_matrix(
            (np.ones((len(row_ones),)), (np.array(row_ones), np.array(col_ones))),
            shape=(num_entities, num_entities))
        relation_map[rel] = normalize(m, norm="l1", axis=1)
        if RELATION_WEIGHTING:
            if rel not in relation_embeddings:
                score = NOTFOUNDSCORE
            else:
                score = np.dot(question_embedding, relation_embeddings[rel]) / (
                    np.linalg.norm(question_embedding) *
                    np.linalg.norm(relation_embeddings[rel]))
            relation_map[rel] = relation_map[rel] * np.power(score, EXPONENT)
    if DECOMPOSE_PPV:
        adj_mat = sum(relation_map.values()) / len(relation_map)
    else:
        adj_mat = csr_matrix(
            (np.ones((len(row_ones),)), (np.array(row_ones), np.array(col_ones))),
            shape=(num_entities, num_entities))
    return entity_map, relation_map, normalize(adj_mat, norm="l1", axis=1)

def _personalized_pagerank(seed, W):
    """Return the PPR vector for the given seed and adjacency matrix.

    Args:
        seed: A sparse matrix of size E x 1.
        W: A sparse matrix of size E x E whose rows sum to one.

    Returns:
        ppr: A vector of size E.
    """
    restart_prob = RESTART
    r = restart_prob * seed
    s_ovr = np.copy(r)
    for i in range(MAX_ITER):
        r_new = (1. - restart_prob) * (W.transpose().dot(r))
        s_ovr = s_ovr + r_new
        delta = abs(r_new.sum())
        if delta < 1e-5: break
        r = r_new
    return np.squeeze(s_ovr)

def _get_subgraph(entities, kb_r, multigraph_W):
    """Get subgraph describing a neighbourhood around given entities."""
    seed = np.zeros((multigraph_W.shape[0], 1))
    if not SEED_WEIGHTING:
        seed[entities] = 1. / len(set(entities))
    else:
        seed[entities] = np.expand_dims(np.arange(len(entities), 0, -1),
                                        axis=1)
        seed = seed / seed.sum()
    ppr = _personalized_pagerank(seed, multigraph_W)
    sorted_idx = np.argsort(ppr)[::-1]
    extracted_ents = sorted_idx[:MAX_ENT]
    extracted_scores = ppr[sorted_idx[:MAX_ENT]]
    # check if any ppr values are nearly zero
    zero_idx = np.where(ppr[extracted_ents] < 1e-6)[0]
    if zero_idx.shape[0] > 0:
        extracted_ents = extracted_ents[:zero_idx[0]]
    extracted_tuples = []
    ents_in_tups = set()
    for relation in kb_r:
        submat = kb_r[relation][extracted_ents, :]
        submat = submat[:, extracted_ents]
        row_idx, col_idx = submat.nonzero()
        for ii in range(row_idx.shape[0]):
            extracted_tuples.append(
                (extracted_ents[row_idx[ii]], relation,
                 extracted_ents[col_idx[ii]]))
            ents_in_tups.add((extracted_ents[row_idx[ii]],
                extracted_scores[row_idx[ii]]))
            ents_in_tups.add((extracted_ents[col_idx[ii]],
                extracted_scores[col_idx[ii]]))
    return extracted_tuples, list(ents_in_tups)

def _read_seeds():
    """Return map from question ids to seed entities."""
    seed_map = {}
    with open(question_seeds) as f:
        for line in f:
            qId, seeds = line.strip("\n").split("\t")
            #seed_map[qId] = [seeds.split(",", 1)[0].split("=")[0]]
            seed_map[qId] = [seed.split("=")[0]
                             for seed in seeds.split(",") if seed]
    return seed_map

def _convert_to_readable(tuples, inv_map):
    readable_tuples = []
    for tup in tuples:
        readable_tuples.append([
            {"kb_id": inv_map[tup[0]], "text": inv_map[tup[0]]},
            {"rel_id": tup[1], "text": tup[1]},
            {"kb_id": inv_map[tup[2]], "text": inv_map[tup[2]]},
        ])
    return readable_tuples

def _readable_entities(entities, inv_map):
    readable_entities = []
    try:
        for ent, sc in entities:
            readable_entities.append(
                {"text": inv_map[ent], "kb_id": inv_map[ent],
                    "pagerank_score": sc})
    except TypeError:
        for ent in entities:
            readable_entities.append(
                {"text": inv_map[ent], "kb_id": inv_map[ent]})
    return readable_entities

def _get_answer_coverage(answers, entities, inv_map):
    found, total = 0., 0
    all_entities = set([inv_map[ee] for ee, _ in entities])
    for answer in answers:
        if answer["freebaseId"] in all_entities: found += 1.
        total += 1
    return found / total

if __name__ == "__main__":

    questions = json.load(open(question_json))
    random.shuffle(questions)
    seed_map = _read_seeds()

    relation_embeddings = pkl.load(open(relation_emb))
    question_embeddings = pkl.load(open(question_emb))

    with open(output_json, "wb") as fo:
        answer_recall, total = 0.0, 0
        max_recall = 0.
        bad_questions = []
        num_empty = 0
        for question in tqdm(questions):
            question_embedding = question_embeddings[question["QuestionId"]]
            fact_file = os.path.join(
                subgraph_dir, question["QuestionId"] + ".nxhd")
            if not os.path.exists(fact_file):
                print("fact file not found for %s" % question["QuestionId"])
                entity_map, relation_map, adj_mat = {}, {}, None
            else:
                entity_map, relation_map, adj_mat = _read_facts(
                    fact_file, relation_embeddings, question_embedding,
                    seed_map[question["QuestionId"]], question["QuestionId"])
            inv_map = {i: k for k, i in entity_map.iteritems()}
            seed_entities = []
            ans_entities = []
            if question["QuestionId"] in seed_map:
                for ee in seed_map[question["QuestionId"]]:
                    if ee in entity_map:
                        seed_entities.append(entity_map[ee])
                for answer in question["Answers"]:
                    if answer["freebaseId"] in entity_map:
                        ans_entities.append(entity_map[answer["freebaseId"]])
            if not seed_entities:
                print("No seeds found for %s!" % question["QuestionId"])
                extracted_tuples, extracted_ents = [], []
            elif adj_mat is None:
                print("No facts for %s!" % question["QuestionId"])
                extracted_tuples, extracted_ents = [], []
            else:
                sd = seed_entities + ans_entities if USEANSWER else seed_entities
                extracted_tuples, extracted_ents = _get_subgraph(
                    sd, relation_map, adj_mat)
            if not extracted_tuples:
                num_empty += 1
            if not question["Answers"]:
                curr_recall = 0.
                cmax_recall = 0.
            else:
                curr_recall = _get_answer_coverage(question["Answers"],
                                                   extracted_ents, inv_map)
                cmax_recall = float(len([answer for answer in question["Answers"]
                    if answer["freebaseId"] in entity_map])) / len(question["Answers"])
            if curr_recall < 1.:
                bad_questions.append([question["QuestionId"], question["QuestionText"],
                    seed_map[question["QuestionId"]], curr_recall])

            answer_recall += curr_recall
            max_recall += cmax_recall
            total += 1
            data = {
                "question": question["QuestionText"],
                "entities": _readable_entities(seed_entities[:MAX_SEEDS], inv_map),
                "answers": [{"kb_id": answer["freebaseId"], "text": answer["text"]}
                            for answer in question["Answers"]],
                "id": question["QuestionId"],
                "subgraph": {
                    "entities": _readable_entities(extracted_ents, inv_map),
                    "tuples": _convert_to_readable(extracted_tuples, inv_map)
                }
            }
            fo.write(json.dumps(data) + "\n")
    print("%d questions with empty subgraphs." % num_empty)
    print("Answer recall = %.3f" % (answer_recall / total))
    print("Upper Bound = %.3f" % (max_recall / total))
    print("Example questions with low recall: ")
    print("\n".join(["%s\t%s\t%s\t%.2f" % (
        item[0], item[1], ",".join(ss for ss in item[2]), item[3])
        for item in bad_questions[:10]]))
