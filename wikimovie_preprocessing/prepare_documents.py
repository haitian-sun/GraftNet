# Extract all documents into text file.
# Usage: python prepare_documents.py
# The output file has lines in the following format:
# docID<TAB>docTitle<TAB>docContents

import sys
data_path = sys.argv[1]

doc_file = data_path + "/processed_wiki.json"
output_file = data_path + "/wiki_for_lucene.txt"

import json
import io

with open(doc_file) as fi, io.open(output_file, "w") as fo:
    for line in fi:
        doc_obj = json.loads(line.strip())
        fo.write(
            u"%s\t%s\t%s\n" % (doc_obj["documentId"],
                               doc_obj["title"]["text"],
                               doc_obj["document"]["text"]))
