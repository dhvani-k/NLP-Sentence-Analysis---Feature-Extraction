# your imports go here
from collections import Counter

import nltk
from nltk.corpus import brown
import math
import numpy as np
import spacy
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import stanza
import spacy_stanza
import torch
from spacy.matcher import DependencyMatcher
from spacy import displacy
import random


# global variables (e.g., nlp modules, sentencetransformer models, dependencymatcher objects) go here
from transformers import RobertaTokenizer, RobertaModel

stanza.download("en")
nlp = spacy_stanza.load_pipeline("en")

matcher = DependencyMatcher(nlp.vocab)

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)


# adding globals here will mean that when we import e.g., extract_indirect_object it will still work ;)
PO = [  # looks for direct objects followed by indirect objects
    {"RIGHT_ID": "direct_obj",
     "RIGHT_ATTRS": {
         "DEP": "obj"  # dobj in spacy
     }},
    {
        "LEFT_ID": "direct_obj",
        "LEFT_ATTRS": {
            "DEP": "obj"  # dobj in spacy
        },
        "RIGHT_ID": "indirect_obj",
        "REL_OP": "$++",  # to the right and sibling
        "RIGHT_ATTRS": {
            "DEP": "obl",  # dative in spacy
        }
    }
]

DO = [
    {"RIGHT_ID": "indirect_obj",
     "RIGHT_ATTRS": {
         "DEP": "iobj"
     }},
    {
        "LEFT_ID": "indirect_obj",
        "LEFT_ATTRS": {
            "DEP": "iobj"
        },
        "RIGHT_ID": "direct_obj",
        "REL_OP": "$++",
        "RIGHT_ATTRS": {
            "DEP": "obj",
        }
    }
]

matcher.add("PO", [PO])
matcher.add("DO", [DO])


def get_sentence_structure(sentence):
    doc = nlp(sentence)
    matches = matcher(doc)

    sentence_structure = None
    for match_id, match in matches:
        if match_id == nlp.vocab.strings["DO"]:
            sentence_structure = "DO"
        elif match_id == nlp.vocab.strings["PO"]:
            sentence_structure = "PO"
    return sentence_structure


def extract_direct_object(sentence):
    extracted_direct_object_string = ''

    if get_sentence_structure(sentence) == 'DO':
        doc = nlp(sentence)
        matches = matcher(doc)

        match_id, token_ids = matches[0]
        token_obj = doc[token_ids[1]]
        extracted_direct_object = [x.text for x in token_obj.subtree if x.dep_ == 'obj']
        extracted_direct_object_string = ' '.join(extracted_direct_object)

    elif get_sentence_structure(sentence) == 'PO':
        doc = nlp(sentence)
        matches = matcher(doc)

        match_id, token_ids = matches[0]
        prep_token = doc[token_ids[1]]
        po_token = prep_token.head
        po_children = [child for child in po_token.children if child.dep_ in ['obj', 'iobj']]
        if len(po_children) > 0:
            extracted_direct_object = po_children[0].text
            extracted_direct_object_string = extracted_direct_object

    return extracted_direct_object_string


def extract_indirect_object(sentence):
    extracted_indirect_object_string = ''

    if get_sentence_structure(sentence) == 'DO':
        doc = nlp(sentence)
        matches = matcher(doc)

        match_id, token_ids = matches[0]
        token_obj = doc[token_ids[0]]
        extracted_indirect_object = [x.text for x in token_obj.subtree if x.dep_ in ['iobj']]
        if extracted_indirect_object:
            extracted_indirect_object_string = ' '.join(extracted_indirect_object)

    elif get_sentence_structure(sentence) == 'PO':
        doc = nlp(sentence)
        matches = matcher(doc)

        match_id, token_ids = matches[0]
        prep_token = doc[token_ids[0]]
        po_token = prep_token.head
        po_children = [child for child in po_token.children if child.dep_ in ['obj', 'obl']]
        if len(po_children) > 1:
            extracted_indirect_object = [x.text for x in po_children if x.dep_ == 'obl']
            if extracted_indirect_object:
                extracted_indirect_object_string = extracted_indirect_object[0]
        elif len(po_children) == 1:
            # check if the direct object appears before the indirect object
            direct_obj_token = po_children[0]
            if direct_obj_token.i < po_token.i:
                extracted_indirect_object = [x.text for x in po_children if x.dep_ == 'obl']
                if extracted_indirect_object:
                    extracted_indirect_object_string = extracted_indirect_object[0]

    return extracted_indirect_object_string


def extract_feature_1(noun_phrase, sentence):
    feature_1 = None
    words = noun_phrase.split()
    feature_1 = len(words)
    assert type(feature_1) is int
    return feature_1


def extract_feature_2(noun_phrase, sentence):
    feature_2 = ''
    tokens = nlp(noun_phrase)
    pos_tags = ' '.join([token.pos_ for token in tokens])
    # Replace anything that appears less than 5 times with UNK
    pos_tags = ' '.join([tag if count >= 5 else 'UNK' for tag, count in Counter(pos_tags.split()).items()])
    feature_2 = pos_tags
    assert type(feature_2) is str
    return feature_2


def extract_feature_3(noun_phrase, sentence):
    nlp_engine = spacy.load("en_core_web_sm")

    doc = nlp_engine(sentence)
    noun_phrase_vec = nlp_engine(noun_phrase).vector
    similarities = []

    for token in doc:
        if token.pos_ == "NOUN":
            similarity = dot(token.vector, noun_phrase_vec)/(norm(token.vector)*norm(noun_phrase_vec))
            similarities.append(similarity)

    if len(similarities) < 2:
        return 0.0

    max_similarity = max(similarities)
    max_similarity = float(max_similarity)
    assert type(max_similarity) is float
    return max_similarity



def extract_sentence_embedding(sentence):
    sentence_embedding = None
    tokens = roberta_tokenizer.encode(sentence, add_special_tokens=True)
    token_ids = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        outputs = roberta_model(token_ids)

    last_hidden_state = outputs[0]

    sentence_embedding = torch.mean(last_hidden_state, dim=1)

    sentence_embedding = sentence_embedding.numpy()
    return sentence_embedding
    assert type(sentence_embedding) is np.array
    return sentence_embedding


def alter_sentence(sentence):
    altered_sentence = sentence
    # add anything to change the string here
    words = sentence.split()

    # Randomly shuffle the words
    random.shuffle(words)

    # Join the shuffled words back into a sentence
    altered_sentence = ' '.join(words)

    return altered_sentence

