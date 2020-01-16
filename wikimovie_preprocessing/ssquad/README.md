SSQUAD: A PSEUDODOCUMENT SEARCH UTILITY FOR QUESTION-ANSWERING
==============================================================
<krivard@cs.cmu.edu> November 2017

SSquad is only slightly more engineered than a shell script, with
individual Java main classes for each major dataset variant. This
gives us the flexibility to input wildly varying data formats, but it
does make it a bit awkward to learn. This document should help give
you a roadmap for the different structures at work.


CLASS LISTING
=============

[domain][variation][pagewise|sentencewise]Dataset - Main class for a particular dataset. *Dataset.process() encodes input format
                                                    handling. Common input formats use inheritance to share code.
QueryFolio - a thing that holds the pseudodocuments for a query
     Query - a QF for generating QUASAR context documents, including extracting candidate solutions
    Export - a QF for generating Solr input files used in the hand-labeling activity

        DocumentStore - a thing that manages the lucene store for one or more queries
[method]DocumentStore - a DS configured for variant storage
    DocumentStoreTron - a DS factory

ProcessHTML - a thing that parses an HTML file, extracts the plaintext, and optionally splits by sentences (via Stanford NLP)


