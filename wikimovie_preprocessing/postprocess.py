"""Script to prepare the json files with retrieved documents.

The retrieval file with question Id to document Id mapping is read, along
with the document file with the json object for each document and the question
json files with the remaining question data. The documents are added to the 
question jsons and the output is saved. Answer recall among the retrieved docs
is also computed using an exact string match.
"""

import sys
data_path = sys.argv[1]

retrieval_file = data_path + "/wiki-entities_qa_retrieved_docs.txt"
question_file_prefix = data_path + "/wiki-entities_subgraph_qa_"
document_json = data_path + "/processed_wiki.json"
entity_index_json = data_path + "/inverted_entity_doc_index.json"

output_json_prefix = data_path + "/wiki-entities_docs_subgraphs_qa_"

import json
import re
import nltk
import numpy as np
from tqdm import tqdm

RANGES = [1, 5, 10, 20, 50, 100]

print("reading documents ...")
id_to_doc = {}
with open(document_json) as f:
    for line in tqdm(f):
        data = json.loads(line.strip())
        id_to_doc[data["documentId"]] = data

print("reading retrievals ...")
id_to_retrieved = {}
with open(retrieval_file) as f:
    for line in tqdm(f):
        try:
            qId, retr = line.strip().split("\t")
        except ValueError:
            print("WARNING: Did not find any passages for", line.strip())
            continue
        doc_scores = [dd.split("=") for dd in retr.split(",") if dd]
        id_to_retrieved[qId] = [(int(dd[0]), float(dd[1])) for dd in doc_scores]

print("reading entity index ...")
entity_index = json.load(open(entity_index_json))

def _compute_recall(question_obj):
    """Compute answer recall in retrieved documents."""
    answers = set([u" ".join(nltk.word_tokenize(answer["text"])).lower()
                             for answer in question_obj["answers"]])
    pattern = re.compile(u"|".join(re.escape(aa) for aa in answers),
                         flags=re.UNICODE | re.IGNORECASE)
    total = len(answers)
    recall = np.zeros((len(RANGES),))
    text_matches, text_docs = [], []
    all_entities_in_docs = set()
    for ii, doc_id in enumerate(question_obj["passages"]):
        text = (id_to_doc[doc_id["document_id"]]["document"]["text"] + " " +
                id_to_doc[doc_id["document_id"]]["title"]["text"])
        c_doc = id_to_doc[doc_id["document_id"]]
        all_entities_in_docs.update(
            [entity["kb_id"] for entity in c_doc["document"]["entities"]])
        all_entities_in_docs.update(
            [entity["kb_id"] for entity in c_doc["title"]["entities"]])
        matches = re.findall(pattern, text)
        for match in matches:
            if match in answers:
                text_matches.append((ii, match))
                text_docs.append(id_to_doc[doc_id["document_id"]])
                recall[[ix for ix, rr in enumerate(RANGES) if rr > ii]] += 1.
                answers.remove(match)
        if not answers: break
    answer_ids = set([answer["kb_id"] for answer in question_obj["answers"]])
    entity_matches = all_entities_in_docs.intersection(answer_ids)
    num_intersect = len(entity_matches)
    return recall / total, float(num_intersect) / total

for split in ["dev", "test", "train"]:
    print("processing %s ..." % split)
    filepath = question_file_prefix + split + ".json"
    output = output_json_prefix + split + ".json"
    recall, hits, total = np.zeros((len(RANGES),)), np.zeros((len(RANGES),)), 0
    e_recall = 0.
    num_passages = 0.
    with open(filepath) as f, open(output, "wb") as fo:
        for line in tqdm(f):
            data = json.loads(line.strip())
            if data["id"] in id_to_retrieved:
                passage_ids = set(id_to_retrieved[data["id"]])
            else:
                passage_ids = set()
            for entity in data["entities"]:
                e_id = str(entity["kb_id"])
                if e_id not in entity_index: continue
                if len(entity_index[e_id]) > 50: continue
                passage_ids.update([(item, -1) for item in entity_index[e_id]])
            data["passages"] = [
                {"document_id": item[0], "retrieval_score": item[1]}
                for item in passage_ids]
            num_passages += len(data["passages"])
            my_recall, my_e_recall = _compute_recall(data)
            e_recall += my_e_recall
            recall += my_recall
            hits[my_recall > 0.] += 1.
            total += 1
            fo.write(json.dumps(data) + "\n")
    print("recall@%r = %r" % (RANGES, recall / total))
    print("hits@%r = %r" % (RANGES, hits / total))
    print("entity based recall @%r = %r" % (100, e_recall / total))
    print("average number of passages = %.f" % (num_passages / total))
