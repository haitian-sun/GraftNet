# Read questions, assign ids and store in text file for querying.
# Usage: python prepare_questions.py
# The output txt file is in the following format:
# questionId<TAB>question

import sys
data_path = sys.argv[1]

question_file_prefix = data_path + "/wiki-entities_subgraph_qa_"
splits = ["dev", "test", "train"]
output_txt = data_path + "/wiki-entities_lucene_qas.txt"

import os
import json
import io
from tqdm import tqdm
import string
from nltk.corpus import stopwords as SW

stopwords = set(SW.words("english")) | set(string.punctuation)

with io.open(output_txt, "w") as fot:
    for split in splits:
        print("processing %s ..." % split)
        with open(question_file_prefix + split + ".json") as fin:
            for ii, line in tqdm(enumerate(fin)):
                question = json.loads(line.strip())
                keywords = u" ".join([token for token in question["question"].split()
                                      if token not in stopwords])
                fot.write("%s\t%s\n" % (question["id"], keywords))
