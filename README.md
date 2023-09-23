# NLP: Sentence Analysis & Feature Extraction

This repository contains code for sentence analysis and feature extraction using natural language processing techniques. The code is written in Python and makes use of various libraries and models.

## Prerequisites
Before running the code, make sure you have the following dependencies installed:

* nltk
* numpy
* spacy
* stanza
* torch
* pandas
* spacy-stanza
* transformers

You can install these dependencies using pip:
```bash
pip install nltk numpy spacy stanza torch pandas spacy-stanza transformers
```

Additionally, you need to download the English language model for spacy:
```bash
python -m spacy download en
```

## Usage
The code provides several functions for sentence analysis and feature extraction:

- get_sentence_structure(sentence): This function takes a sentence as input and returns its structure, either "DO" (direct object) or "PO" (prepositional object).

- extract_direct_object(sentence): This function extracts the direct object from a sentence. It takes a sentence as input and returns the extracted direct object as a string.

- extract_indirect_object(sentence): This function extracts the indirect object from a sentence. It takes a sentence as input and returns the extracted indirect object as a string.

- extract_feature_1(noun_phrase, sentence): This function extracts feature 1 from a noun phrase and a sentence. It takes a noun phrase and a sentence as input and returns feature 1 as an integer.

- extract_feature_2(noun_phrase, sentence): This function extracts feature 2 from a noun phrase and a sentence. It takes a noun phrase and a sentence as input and returns feature 2 as a string.

- extract_feature_3(noun_phrase, sentence): This function extracts feature 3 from a noun phrase and a sentence. It takes a noun phrase and a sentence as input and returns feature 3 as a float.

- extract_sentence_embedding(sentence): This function extracts the sentence embedding using the Roberta model. It takes a sentence as input and returns the sentence embedding as a numpy array.

- alter_sentence(sentence): This function alters a sentence by shuffling its words randomly. It takes a sentence as input and returns the altered sentence as a string.

You can use these functions in your own code by importing them.

## Limitations
The code has the following limitations:

It can only handle sentences that are in the English language.
It cannot handle sentences that contain complex syntactic structures, such as relative clauses or embedded clauses.

##Future work
Future work on the code could include the following:

Expanding the code to handle other languages.
Improving the code's ability to handle complex syntactic structures.
Adding additional features to the code, such as the ability to extract the semantic role of a noun phrase in a sentence.
