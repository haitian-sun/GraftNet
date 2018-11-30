"""Script to read WebQuestionsSP json file and extract required fields.

Output will be a json list of questions with the following fields:
    - QuestionId
    - QuestionText (processed)
    - QuestionKeywords (after removing wh-words, stopwords)
    - OracleEntities (from annotations)
    - Answers -- List of dicts with fields "freebaseID" and "text"
"""

import json
import nltk
import os
import io

from nltk.corpus import stopwords as SW

import sys

out_json = "scratch/webqsp_processed.json"
in_files = ["webqsp_train", "webqsp_test"]

stopwords = set(SW.words("english"))
stopwords.add("'s")

def extract_keywords(text):
    """Remove wh-words and stop words from text."""
    return u" ".join([token for token in nltk.word_tokenize(text)
        if token not in stopwords])

def get_answers(question):
    """extract unique answers from question parses."""
    answers = set()
    for parse in question["Parses"]:
        for answer in parse["Answers"]:
            answers.add((answer["AnswerArgument"],
                answer["EntityName"]))
    return answers

def get_entities(question):
    """extract oracle entities from question parses."""
    entities = set()
    for parse in question["Parses"]:
        if parse["TopicEntityMid"] is not None:
            entities.add((parse["TopicEntityMid"], parse["TopicEntityName"]))
    return entities

questions = []
for fil in in_files:
    data = json.load(open(fil))
    for question in data["Questions"]:
        q_obj = {
            "QuestionId": question["QuestionId"],
            "QuestionText": question["ProcessedQuestion"],
            "QuestionKeywords": extract_keywords(question["ProcessedQuestion"]),
            "OracleEntities": [
                {"freebaseId": "<fb:" + entity[0] + ">", "text": entity[1]}
                for entity in get_entities(question)
            ],
            "Answers": [
                {"freebaseId": "<fb:" + answer[0] + ">"
                 if answer[0].startswith("m.") or answer[0].startswith("g.") else answer[0],
                 "text": answer[1]}
                for answer in get_answers(question)
            ]
        }
        questions.append(q_obj)

json.dump(questions, open(out_json, "w"))
