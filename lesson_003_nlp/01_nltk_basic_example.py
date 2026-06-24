import nltk

# Download the new tokenizer models
'''
NLTK’s tokenizers rely on pre‐trained models stored under the “punkt_tab” resource.
● This line checks your local nltk_data folder; if “punkt_tab” isn’t present, it downloads
and unzips the necessary files.
● You only need to run this once per environment.

punkt_tab is a table resource that contains language-specific abbreviation tables used by the Punkt tokenizer.
It is not the tokenizer itself.

It includes things like:
Lists of abbreviations
Language‑specific punctuation rules
Additional metadata

It is optional and only needed for certain languages (non-English) or advanced customization.
'''
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

text = "Natural Language Processing with Python is fun!"
tokens = word_tokenize(text)
print(tokens)

