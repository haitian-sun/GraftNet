"""Script to link entities in questions to entities in KB.

See link_entities_in_text for details.
"""

import sys

data_path = sys.argv[1]
out_path = sys.argv[2]
split = sys.argv[3]

input_file = data_path + "/questions/wiki_entities/wiki-entities_qa_" + split + ".txt"
linked_json = out_path + "/wiki-entities_linked_qa_" + split + ".json"

kb_file = out_path + "/processed_kb.pkl"
kb_entities = out_path + "/entity_vocab.txt"

import json
import io
import nltk
import cPickle as pkl
import re
from tqdm import tqdm

def read_kb_entities(kb_loc):
    """Read KB entities and create inverted index from lowercase to entity ids.

    Input:
        kb_file: pickled list whose first element is a dict mapping entity
            names to their ids.

    Output:
        name_map: Dict mapping entity identifier strings to entity ids.
            The identifier strings are formed by removing leading articles and
            converting to lowercase.
    """
    kb_ents, _ = pkl.load(open(kb_loc, "rb"))
    name_map = {}
    all_entities = []
    for entity, eid in kb_ents.iteritems():
        if len(entity) > 1 and re.match("\w", entity[-1]) is None:
            regex = re.compile(u"\\b{}\\b".format(re.escape(entity[:-1])), flags=re.UNICODE)
        else:
            regex = re.compile(u"\\b{}\\b".format(re.escape(entity)), flags=re.UNICODE)
        regex_ans = re.compile(u"{}".format(re.escape(entity)), flags=re.UNICODE)
        all_entities.append((regex, entity, eid, regex_ans))
    all_entities = sorted(all_entities, key=lambda x: -len(x[1]))
    return name_map, all_entities

if __name__ == "__main__":
    name_map, all_entities = read_kb_entities(kb_file)

    entities = io.open(kb_entities).read().splitlines()
    assert len(set(entities)) == len(entities)
    entity_dict = {e:i for i, e in enumerate(entities)}

    unmatched_answers = []
    multiplematch_questions = []
    num_linked = 0
    count_linked = 0.0
    total = 0
    with io.open(input_file) as fin, open(linked_json, "wb") as fout:
        for ii, line in tqdm(enumerate(fin)):
            _, qa = line.strip().split(" ", 1)
            text, answers = qa.split("\t")
            answer_obj = []
            matches = []
            for regex_q, ent, eid, regex_a in all_entities:
                if regex_a is None: continue
                # match answer
                m = regex_a.search(answers)
                if m is not None:
                    answers = answers[:m.start()] + u" ____ " + answers[m.end():]
                    answer_obj.append({"text": ent, "kb_id": eid})
                # match question
                m = regex_q.search(text)
                if m is not None:
                    text = text[:m.start()] + u" __{}__ ".format(eid) + text[m.end():]
                    matches.append((eid, ent, m.start()))
            if not answer_obj:
                print "Error in ", (text, answers)
                unmatched_answers.append((text, answers))
                continue
            # tokenize
            question_obj = {}
            question_obj["id"] = split + "_" + str(ii)
            question_obj["entities"] = []
            matches = sorted(matches, key=lambda x: x[2])
            tokenized_text = nltk.word_tokenize(text)
            for eid, ent, _ in matches:
                tokenized_entity = nltk.word_tokenize(ent)
                try:
                    idx = tokenized_text.index(u"__{}__".format(eid))
                except:
                    print(tokenized_text)
                    print(eid)
                    sys.exit()
                tokenized_text = (tokenized_text[:idx] +
                                  tokenized_entity + tokenized_text[idx+1:])
                question_obj["entities"].append({
                    "text": ent,
                    "start": idx,
                    "end": idx + len(tokenized_entity),
                    "kb_id": eid,
                })
            question_obj["question"] = u" ".join(tokenized_text)
            if question_obj["entities"]: linked = True
            else: linked = False
            curr_count_linked = len(question_obj["entities"])
            if curr_count_linked != 1:
                multiplematch_questions.append(question_obj)
            count_linked += curr_count_linked
            num_linked += int(linked)
            total += 1
            question_obj["answers"] = answer_obj
            fout.write(json.dumps(question_obj) + "\n")
    print("%d answers not matched" % len(unmatched_answers))
    print("%d linked to KB" % num_linked)
    print("%.2f links to KB per question" % (count_linked / total))
    print("%d bad questions" % len(multiplematch_questions))
