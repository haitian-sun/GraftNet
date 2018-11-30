# Extractig Question Subgraphs

This folder contains the scripts for extracting the KB portion of question
subgraphs for the GRAFT-Net model.

To run the pipeline simply call `./run_pipeline` from a bash terminal. The script first downloads and unpacks the required files:
- WebQuestionsSP dataset
- Glove embeddings
- Entity links for the questions from [STAGG](https://raw.githubusercontent.com/scottyih/STAGG)
- Our preprocessed version of the Freebase data.

All the downloads plus the pipeline takes upwards of 1 hour to complete.

The preprocessed Freebase data includes a subset for each question with only the relations and entities reachable within 2 hops from the seed entities mentioned in the question (from the entity linked data). Further this subset is pre-processed to include only relations which are mentioned in the semantic parse of at least one of the questions.

After unpacking, the pre-processed data includes a `relations` file listing all the included relations, a `all_entities` file including all the entities in any subset, and a `stagg.neighborhoods` directory with a file for each question named `<questionId>.nxhd` which contains the subset of triples for that question.

Edge-weighted PPR is performed on the subset of each question starting from the seeds mentioned in that question to extract a subgraph with 500 most important entities. This PPR code is in `step4_extract_subgraphs.py`.
