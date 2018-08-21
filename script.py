import sys
import json
import nltk
from tqdm import tqdm
from collections import Counter
from itertools import izip

def combine_dist(dist1, dist2, w1):
    ensemble_dist = dist2.copy()
    for gid, prob in dist1.items():
        if gid in ensemble_dist:
            ensemble_dist[gid] = (1 - w1) * ensemble_dist[gid] + w1 * prob
        else:
            ensemble_dist[gid] = prob
    return ensemble_dist

def get_one_f1(entities, dist, eps, answers):
    correct = 0.0
    total = 0.0
    best_entity = -1
    max_prob = 0.0
    preds = []
    for entity in entities:
        if dist[entity] > max_prob:
            max_prob = dist[entity]
            best_entity = entity
        if dist[entity] > eps:
            preds.append(entity)
    
    return cal_eval_metric(best_entity, preds, answers)

def cal_eval_metric(best_pred, preds, answers):
    correct, total = 0.0, 0.0
    for entity in preds:
        if entity in answers:
            correct += 1
        total += 1
    if len(answers) == 0:
        if total == 0:
            return 1.0, 1.0, 1.0, 1.0 # precision, recall, f1, hits
        else:
            return 0.0, 1.0, 0.0, 1.0 # precision, recall, f1, hits
    else:
        hits = float(best_pred in answers)
        if total == 0:
            return 1.0, 0.0, 0.0, hits # precision, recall, f1, hits
        else:
            precision, recall = correct / total, correct / len(answers)
            f1 = 2.0 / (1.0 / precision + 1.0 / recall) if precision != 0 and recall != 0 else 0.0
            return precision, recall, f1, hits

def compare_pr(kb_pred_file, doc_pred_file, hybrid_pred_file, w_kb, eps_doc, eps_kb, eps_ensemble, eps_hybrid, eps_ensemble_all):
    doc_only_recall, doc_only_precision, doc_only_f1, doc_only_hits = [], [], [], []
    kb_only_recall, kb_only_precision, kb_only_f1, kb_only_hits = [], [], [], []
    ensemble_recall, ensemble_precision, ensemble_f1, ensemble_hits = [], [], [], []
    hybrid_recall, hybrid_precision, hybrid_f1, hybrid_hits = [], [], [], []
    ensemble_all_recall, ensemble_all_precision, ensemble_all_f1, ensemble_all_hits = [], [], [], []

    total_not_answerable = 0.0
    with open(kb_pred_file) as f_kb, open(doc_pred_file) as f_doc, open(hybrid_pred_file) as f_hybrid:
        line_id = 0
        for line_kb, line_doc, line_hybrid in tqdm(zip(f_kb, f_doc, f_hybrid)):
            line_id += 1
            line_kb = json.loads(line_kb)
            line_doc = json.loads(line_doc)
            line_hybrid = json.loads(line_hybrid)
            assert line_kb['answers'] == line_doc['answers'] == line_hybrid['answers']
            answers = set([unicode(answer) for answer in line_kb['answers']])
            # total_not_answerable += (len(answers) == 0)
            # assert len(answers) > 0

            dist_kb = line_kb['dist']
            dist_doc = line_doc['dist']
            dist_hybrid = line_hybrid['dist']
            dist_ensemble = combine_dist(dist_kb, dist_doc, w_kb)
            dist_ensemble_all = combine_dist(dist_ensemble, dist_hybrid, w1=0.3)

            kb_entities = set(dist_kb.keys())
            doc_entities = set(dist_doc.keys())
            either_entities = kb_entities | doc_entities
            assert either_entities == set(dist_hybrid.keys())

            p, r, f1, hits = get_one_f1(doc_entities, dist_doc, eps_doc, answers)
            doc_only_precision.append(p)
            doc_only_recall.append(r)
            doc_only_f1.append(f1)
            doc_only_hits.append(hits)

            p, r, f1, hits = get_one_f1(kb_entities, dist_kb, eps_kb, answers)
            kb_only_precision.append(p)
            kb_only_recall.append(r)
            kb_only_f1.append(f1)
            kb_only_hits.append(hits)

            p, r, f1, hits = get_one_f1(either_entities, dist_ensemble, eps_ensemble, answers)
            ensemble_precision.append(p)
            ensemble_recall.append(r)
            ensemble_f1.append(f1)
            ensemble_hits.append(hits)

            p, r, f1, hits = get_one_f1(either_entities, dist_hybrid, eps_hybrid, answers)
            hybrid_precision.append(p)
            hybrid_recall.append(r)
            hybrid_f1.append(f1)
            hybrid_hits.append(hits)

            p, r, f1, hits = get_one_f1(either_entities, dist_ensemble_all, eps_ensemble_all, answers)
            ensemble_all_precision.append(p)
            ensemble_all_recall.append(r)
            ensemble_all_f1.append(f1)
            ensemble_all_hits.append(hits)


    print('text only setting:')
    print('hits: ', sum(doc_only_hits) / len(doc_only_hits))
    print('precision: ', sum(doc_only_precision) / len(doc_only_precision))
    print('recall: ', sum(doc_only_recall) / len(doc_only_recall))
    print('f1: ', sum(doc_only_f1) / len(doc_only_f1))
    print('\n')
    
    print('kb only setting:')
    print('hits: ', sum(kb_only_hits) / len(kb_only_hits))
    print('precision: ', sum(kb_only_precision) / len(kb_only_precision))
    print('recall: ', sum(kb_only_recall) / len(kb_only_recall))
    print('f1: ', sum(kb_only_f1) / len(kb_only_f1))
    print('\n')

    print('late fusion:')
    print('hits: ', sum(ensemble_hits) / len(ensemble_hits))
    print('precision: ', sum(ensemble_precision) / len(ensemble_precision))
    print('recall: ', sum(ensemble_recall) / len(ensemble_recall))
    print('f1: ', sum(ensemble_f1) / len(ensemble_f1))
    print('\n')

    print('early fusion:')
    print('hits: ', sum(hybrid_hits) / len(hybrid_hits))
    print('precision: ', sum(hybrid_precision) / len(hybrid_precision))
    print('recall: ', sum(hybrid_recall) / len(hybrid_recall))
    print('f1: ', sum(hybrid_f1) / len(hybrid_f1))
    print('\n')

    print('early & late fusion:')
    print('hits: ', sum(ensemble_all_hits) / len(ensemble_all_hits))
    print('precision: ', sum(ensemble_all_precision) / len(ensemble_all_precision))
    print('recall: ', sum(ensemble_all_recall) / len(ensemble_all_recall))
    print('f1: ', sum(ensemble_all_f1) / len(ensemble_all_f1))
    print('\n')


if __name__ == "__main__":
    dataset = sys.argv[1]
    pred_kb_file = sys.argv[2]
    pred_doc_file = sys.argv[3]
    pred_hybrid_file = sys.argv[4]
    if dataset == "wikimovie":
        w_kb = 0.9
        eps_doc, eps_kb, eps_ensemble, eps_hybrid, eps_ensemble_all = 0.5, 0.55, 0.6, 0.5, 0.55
    elif dataset == "webqsp":
        w_kb = 1.0
        eps_doc, eps_kb, eps_ensemble, eps_hybrid, eps_ensemble_all = 0.15, 0.2, 0.2, 0.2, 0.3
    else:
        assert False, "dataset not recognized"

    compare_pr(pred_kb_file, pred_doc_file, pred_hybrid_file, w_kb, eps_doc, eps_kb, eps_ensemble, eps_hybrid, eps_ensemble_all)
