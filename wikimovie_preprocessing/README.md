## Preprocessing code for WikiMovie dataset

This folder contains the code for pre-processing of the WikiMovie
dataset for the EMNLP 2018 paper
[Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text](https://arxiv.org/abs/1809.00782).

First download the data from the following [link](http://www.thespermwhale.com/jaseweston/babi/movieqa.tar.gz)
and extract it.

Then to run all the steps in the pipeline run the script `run_pipeline.sh`.
Make sure to change the `DATA_PATH` variable to the data folder extracted above.
The output files will be stored in a directory named `out` by default.

The following output files will be needed to run the GraftNet model:
1. `wiki-entities_docs_subgraphs_qa_[train|dev|test].json` containing the questions in each split
with the associated fact sub-graphs and pointers to the retrieved passages in the file below.

2. `processed_wiki.json` a preprocessed version of the text corpus with entity links.

3. `entity_vocab.txt` and `relation_vocab.txt` containing a list of all the entities and relations
respectively.

The `ssquad/` sub-directory contains a Java pipeline for running lucene retrieval.

**NOTE**: Running the entire pipeline takes around 24 hours!
