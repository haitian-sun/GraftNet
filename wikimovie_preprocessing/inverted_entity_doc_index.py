# Script to create a map from entities to documents which contain them.

import sys
data_path = sys.argv[1]

processed_documents_file = data_path + "/processed_wiki.json"
output_file = data_path + "/inverted_entity_doc_index.json"

import json
from tqdm import tqdm

entity_index = {}

def _add_entity(entity, doc_id):
    e_id = entity["kb_id"]
    if e_id not in entity_index:
        entity_index[e_id] = set()
    entity_index[e_id].add(doc_id)

with open(processed_documents_file) as fd:
    for line in tqdm(fd):
        doc = json.loads(line.strip())
        doc_id = doc["documentId"]
        for entity in doc["document"]["entities"]:
            _add_entity(entity, doc_id)
        for entity in doc["title"]["entities"]:
            _add_entity(entity, doc_id)

for k, v in entity_index.iteritems():
    entity_index[k] = list(v)

json.dump(entity_index, open(output_file, "w"))
