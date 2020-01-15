"""Script to preprocess KB into a set of entities and adjacency matrices.

Output is a text file mapping entity names to their ids (vocab.txt) a relation
file mapping relation names to their ids and a
sparse numpy matrix of shape V x V x R of directed relations.
"""

import sys
import io
import operator
import numpy as np
import cPickle as pkl
from scipy.sparse import csr_matrix

data_path = sys.argv[1]
out_path = sys.argv[2]

input_file = data_path + "/knowledge_source/wiki_entities/wiki_entities_kb.txt"
output_file = out_path + "/processed_kb.pkl"
entity_vocab = out_path + "/entity_vocab.txt"
relation_vocab = out_path + "/relation_vocab.txt"

RELATIONS = {
    "directed_by": 0,
    "written_by": 1,
    "starred_actors": 2,
    "release_year": 3,
    "in_language": 4,
    "has_genre": 5,
    "has_imdb_rating": 6,
    "has_imdb_votes": 7,
    "has_tags": 8,
}

def read_line(line):
    if line == u"\n":
        return None
    tokens = line.strip().split()
    head = []
    found = False
    for ii, tt in enumerate(tokens[1:]): # ignore leading number
        if tt in RELATIONS:
            relation = tt
            found = True
            break
        head.append(tt)
    if not found: return None
    head = u" ".join(head)
    tails = []
    ctail = []
    for tt in tokens[ii+2:]:
        if tt.endswith(u","):
            ctail.append(tt[:-1])
            tails.append(u" ".join(ctail))
            ctail = []
        else:
            ctail.append(tt)
    tails.append(u" ".join(ctail))
    return [(head, relation, tail) for tail in tails]

if __name__ == "__main__":
    entity_map = {}
    edges = {r: [] for r in RELATIONS}
    with io.open(input_file) as f:
        for line in f:
            rels = read_line(line)
            if rels is None: continue
            for rel in rels:
                if rel[0] not in entity_map:
                    entity_map[rel[0]] = len(entity_map)
                if rel[2] not in entity_map:
                    entity_map[rel[2]] = len(entity_map)
                edges[rel[1]].append((entity_map[rel[0]],
                                      entity_map[rel[2]]))
    adjacency = {}
    for r in RELATIONS:
        if not edges[r]: continue
        adjacency[r] = csr_matrix((np.ones((len(edges[r]),)), zip(*edges[r])),
                                  shape=[len(entity_map), len(entity_map)])

    pkl.dump([entity_map, adjacency], open(output_file, "wb"))

    # save entity vocab
    sorted_entities = sorted(entity_map.items(), key=operator.itemgetter(1))
    f = io.open(entity_vocab, "w", encoding="utf-8")
    f.write(u"\n".join([item[0] for item in sorted_entities]))
    f.close()
    # save relation vocab
    relations = RELATIONS.keys()
    f = open(relation_vocab, "w")
    f.write("\n".join(relations))
    f.close()
