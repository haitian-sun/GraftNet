#!/bin/bash

set -e

DATA_PATH="movieqa/"
OUT_PATH="out/"

mkdir -p $OUT_PATH

echo "Processing the KB ..."
python process_kb.py $DATA_PATH $OUT_PATH

echo "Processing the corpus ..."
python process_wiki.py $DATA_PATH $OUT_PATH

for SPLIT in "dev" "test" "train"; do
    echo "Processing $SPLIT questions ..."
    python link_entities_in_qa.py $DATA_PATH $OUT_PATH $SPLIT
    python extract_subgraph.py $DATA_PATH $OUT_PATH $SPLIT
done

echo "Preparing wiki documents for lucene ..."
python prepare_documents.py $OUT_PATH
echo "Preparing questions for lucene ..."
python prepare_questions.py $OUT_PATH
echo "Creating inverted index for entities ..."
python inverted_entity_doc_index.py $OUT_PATH

echo "Running lucene retrieval ..."
NUMRET=50

CURRPATH=`pwd`
SRCPATH="ssquad"

cd $SRCPATH
javac -sourcepath .:src/ -cp "lib/*" src/edu/cmu/ml/ssquad/WikiMoviesDocumentRetrieval.java

java -cp "lib/*:src/" -Dssquad.topk=$NUMRET edu.cmu.ml.ssquad.WikiMoviesDocumentRetrieval $CURRPATH/$OUT_PATH/wiki-entities_lucene_qas.txt $CURRPATH/$OUT_PATH/wiki_for_lucene.txt $CURRPATH/$OUT_PATH/wiki-entities_qa_retrieved_docs.txt

echo "Postprocessing final question files ..."
cd $CURRPATH
python postprocess.py $OUT_PATH
