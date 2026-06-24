# spacy_example.py

'''
spaCy is a fast, production-grade Python library for NLP. It gives you ready-made pipelines for:
● Tokenization
● Part-of-speech tagging
● Lemmatization
● Named entity recognition (NER)
● Dependency parsing
● (With the right model) word vectors / similarity

It hides a lot of the plumbing so you can get meaningful structure out of raw text with just a few lines.
'''
import spacy

# Load the small English model
# Need to install it manually:
# python -m spacy download en_core_web_sm (Small)
# python -m spacy download en_core_web_md (Medium)
# python -m spacy download en_core_web_lg (Large)
nlp = spacy.load("en_core_web_md")

text = "Apple is looking at buying U.K. startup for $1 billion. I love processing text with spaCy."

doc = nlp(text)  # run the pipeline

# Tokens, lemmas, POS
print("Token | Lemma | POS")
for token in doc:
    print(f"{token.text:8} | {token.lemma_:8} | {token.pos_}")

# Named Entities
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")

# Simple similarity (requires vectors; small model has limited vectors but still works)
# Instead, need to have installed a model with vectors.
# Use en_core_web_md or en_core_web_lg — both include proper word vectors.
sent1 = nlp("I enjoy natural language processing.")
sent2 = nlp("I like working with text.")
print("\nSimilarity between sentences:", round(sent1.similarity(sent2), 3))
