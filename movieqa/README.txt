LICENSE
=======

This data is released under the Creative Commons Public License. A copy is included with the data.


INTRODUCTION
============

This document explains the contents of the MovieQA dataset prepared for "Key-Value Memory Networks for Directly Reading Documents" by Miller et al, 2016. There are two main components:

* "questions": The questions subdirectory contains all of the question-answer pairs on movies. It is split into train, dev, and test sets. The dev and test subdirectories also contain topic-level breakdowns of the questions for more targeted analysis of results. See the "Question Answering Dataset" section for more information.

* "knowledge_source": The files in the knowledge_source subdirectory contain data in various formats to use to answer questions. Details on the structures of all the included files and how they were prepared can be found below (see "Wikipedia Dataset", "Knowledge Base Dataset", "IE Dataset", and "Synthetic Dataset").

Note that in many of the subdirectories there is a distinction between "full" and "wiki_entities" data. The "full" dataset is taken from the set of all movies that have valid OMDb entries. From this set, the "wiki_entities" files are further filtered to be within the set of Wikipedia answerable entity questions and entity facts. For example, if the Wikipedia entry on Blade Runner *did not* mention Ridley Scott as the director, then the OMDb and MovieLens based QA pair (who directed blade runner, ridley scott) would be removed. Similarly, the KB triplet (blade runner, directed_by, ridley scott) would also be removed.

The "wiki_entities" data was used for the experiments of the paper mentioned above.


QUESTION ANSWERING DATASET
==========================

This section describes the dataset of question-answer pairs that make up the Facebook MovieQA benchmark. The files for the QA dataset can be found under movieqa/question/.

Generation:
* This data is generated from OMDb and MovieLens (for the topic tags).
* There are two sets of questions, "full" and "wiki_entities" as explained in `Introduction`.
* OMDb data was downloaded from http://beforethecode.com/projects/omdb/download.aspx on May 28th, 2015.

Question counts:
  full:
    * train - 196453
    * dev   - 10000
    * test  - 10000

  wiki_entities:
    * train - 96185
    * dev   - 10000
    * test  - 9952


Other notes:
* Questions about tags (movie_to_tags and tag_to_movie) with more than 50 answers are removed.


WIKIPEDIA DATASET
=================

This section describes how the dataset of Wikipedia articles was built as a knowledge source for the Facebook MovieQA benchmark. The file is movieqa/knowledge_source/wiki.txt.

The data contains text taken directly from Wikipedia articles in a dump dated 2015-08-05 (https://dumps.wikimedia.org/enwiki/20150805/ -- since expired). From the original dump, everything past the first section is thrown out (i.e. only the text before the contents box is kept). The full list of Wikipedia article URLs (for release in accordance with Wikipedia's requirements) is in movieqa/knowledge_source/wiki_urls.txt.

Algorithm for filtering to only relevant Wikipedia articles:

* For each Wikipedia article, if it has parentheses in its title, throw it out unless they contain “film”.
* Put the OMDb movie titles into a hash, and for each article, check if the title is in the hash (exact string match).
* Take that set of articles, and check in the body for at least one entity from the matched movie (actor, director, or writer) and throw out those that contain no relevant entities.

The script which extracts the summaries from full Wikipedia articles is covered under creative commons. It was adapted from here: http://medialab.di.unipi.it/wiki/Wikipedia_Extractor with the only change being that the script simply prints the text before the “Contents” box for each wikipedia article, instead of the entire article.


KNOWLEDGE BASE DATASET
======================

The movie knowledge base files are movieqa/knowledge_source/full/full_kb.txt and movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt

The knowledge base dataset is a file of (movie, relation, object) groups. It is meant to be as similar to a true knowledge base as possible. It has no 'and' in the answer list--multiple answers are only comma-separated. All lines are space separated as "movie title" "relation_type" "object".

The files were generated from OMDb and MovieLens data in the same way that the synthetic datasets were created, see below.


IE DATASET
==========

This section describes the simulated KB built with information extraction from the Wikipedia dataset detailed above. The file is movieqa/knowledge_source/wiki_ie.txt

The data is in KV-MemNN format, with the memory key tab-separated from the output value(s). Open-source software was used as part of a pipeline to generate this data:

* Stanford NLP Toolkit for coreference resolution.
* SENNA for semantic role labelling.

Coreference resolution helps resolve ambiguous references. Semantic role labelling then takes each sentence and attempts to transform it into a (subject, verb, object) triplet. The KV-MemNN format is obtained from combining subject + verb and tab separating the object. The flipped relation is also included in the KB as (object + REV_verb, subject). The "REV" tag is used to differentiate the target of the verb between the two forms.


SYNTHETIC DATASETS
==================

This section describes the synthetic documents generated to understand the difference between using a KB versus reading documents directly (Wikipedia). The files can be found in the movieqa/knowledge_source/full/synthetic/ and movieqa/knowledge_source/wiki_entities/synthetic directories.

Generation:
* This dataset was generated from 9 fields from OMDb and the MovieLens tags data.
* The files are comprised of a varying number of statements from available fields for each movie (up to 10, 9 OMDb + 1 MovieLens).
* Some info about how the data was made:
    * The script creates 5 quartiles for the field imdbVotes, and writes “popular”, “unpopular”, etc. based on quartile.
    * Splits imdbRating into >8, >6, >4, <4.
    * Tags (from MovieLens) for each movie are each unique and lowercase. If there are no tags, that statement is not printed.
* Each file is generated by following a specific template or a combination of different templates.
* template set: can be one of {'one', 'all'}.
    * 'one': uses one template per field. The template is in English. (Example: Who directed a movie is always presented as "DIRECTOR directed MOVIE_TITLE")
    * 'all': samples uniformly one template from a set for each field. (Example: Who directed a movie has a number of templates, like "DIRECTOR directed MOVIE_TITLE", "MOVIE_TITLE was directed by DIRECTOR", etc.)
    * conj=X: probability of conjunction is X. Conjunction works as follows:
      * Exception: statements about the plot or tags are never conjoined.
      * K = sample from binomial(X, num_templates_which_can_be_conjoined - 1)
        for i = 1, K do
          conjoin two statements (from the can-be-conjoined list)
        end
      * Bounds on K are [0, 7].
      * Bounds on the number of statements are [3, 10].
    * coref=X: probability of replacing a given MOVIE_TITLE with 'it'. Pseudocode:
        for each appearance of MOVIE_TITLE after the first (for a given movie)
          if sample from binomial(X) == 1 then
            replace MOVIE_TITLE with "it"
          else
            replace MOVIE_TITLE with the title of the movie
          end
        end

The files are as follows:
  * movie_statements_entities={full or wiki-entities}_{template_set}_{conjunction_prob}_{coref_prob}.txt
  * example: movie_statements_entities=full_templates=one_conj=0.7_coref=0.0.txt, uses one template per field, the number of conjunctions of statements for a given movie follows binomial(.7, 7), and there is no coref.
