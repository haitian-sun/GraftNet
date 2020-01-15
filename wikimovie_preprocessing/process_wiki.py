"""Script to preprocess wiki.txt.

Extracts documents (sentences) with their titles, tokenizes, and links entities
from the KB.
"""

import sys

data_path = sys.argv[1]
out_path = sys.argv[2]

data_file = data_path + "/knowledge_source/wiki.txt"
kb_file = out_path + "/processed_kb.pkl"
output_file = out_path + "/processed_wiki.json"

import json
import io
import re
import nltk
import cPickle as pkl
from tqdm import tqdm

MIN_ENTITY_LEN = 3
ARTICLES = set([u"the", u"a", u"an"])

# read entity vocab
print("Reading entities and building regexes...")
entity_vocab, _ = pkl.load(open(kb_file, "rb"))
entity_regexes, entity_regexes_lower = [], []
entity_regexes_noarticles, entity_regexes_noarticles_lower = [], []
entity_regexes = {}
for entity, eid in tqdm(entity_vocab.iteritems()):
    if len(entity) < MIN_ENTITY_LEN: continue
    if re.match("\w", entity[-1]) is None:
        entity_clean = entity[:-1]
    else:
        entity_clean = entity
    entity_toks = nltk.word_tokenize(entity_clean)
    if len(entity_toks) > 1 and entity_toks[0].lower() in ARTICLES:
        entity_clean = u" ".join(tok.lower() for tok in entity_toks[1:])
    else:
        entity_clean = u" ".join(tok.lower() for tok in entity_toks)
    pattern = u"(?:(?<=\s)|(?<=^)){}(?:(?=\s)|(?=$))".format(re.escape(entity_clean))
    regex = re.compile(pattern, flags=re.UNICODE | re.IGNORECASE)
    if entity_clean not in entity_regexes:
        entity_regexes[entity_clean] = [regex, []]
    entity_regexes[entity_clean][1].append((entity, eid))
entity_keys = entity_regexes.keys()
entity_keys = sorted(entity_keys, key=lambda x: -len(x))

documentId = 0

def _link_entities(line):
    """Return list of matching entities in text."""
    matches = []
    def _add_entities(match, text, entities):
        start_pos = len(text[:m.start()].split())
        end_pos = start_pos + len(text[m.start():m.end()].split())
        for entity in entities:
            matches.append({"text": entity[0], "kb_id": entity[1],
                            "start": start_pos, "end": end_pos})
        text = text[:m.start()] + " ".join(
            "<ENT>" for _ in range(end_pos-start_pos)) + text[m.end():]
        return text
    for key in entity_keys:
        regex, entities = entity_regexes[key]
        m = re.search(regex, line)
        if m is not None: line = _add_entities(m, line, entities)
    return matches

def _preprocess_line(line):
    """Tokenize, link and lower-case."""
    text = u" ".join(nltk.word_tokenize(line))
    entities = _link_entities(text)
    return {"text": text.lower(), "entities": entities}

def process_document(line, title):
    """Process a single line from wiki.txt. `title` is the detected title
    for current article. If empty means a new article is beginning."""
    global documentId
    if title is None:
        title = _preprocess_line(line.strip().split(None, 1)[1])
        return None, title
    elif line == u"\n":
        return None, None
    else:
        doc = _preprocess_line(line.strip().split(None, 1)[1])
        doc_obj = {
            "documentId": documentId,
            "title": title,
            "document": doc,
        }
        documentId += 1
        return doc_obj, title

# process wiki docs
print("Processing wiki sentences...")
with io.open(data_file) as f, open(output_file, "wb") as fo:
    title = None
    for line in tqdm(f):
        document, title = process_document(line, title)
        if document is not None:
            fo.write(json.dumps(document) + "\n")
