import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

# Downloads (run once)

# The Punkt tokenizer models.
nltk.download('punkt', quiet=True)
# The WordNet lexical database — a huge dictionary of English words, meanings, synonyms, antonyms, and semantic relations.
nltk.download('wordnet', quiet=True)
# Open Multilingual WordNet — translations and multilingual extensions for WordNet.
nltk.download('omw-1.4', quiet=True)
# A Part‑of‑Speech (POS) tagging model based on the averaged perceptron algorithm.
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

'''
Lemmatization reduces words to their dictionary (base) form, called a lemma,
using vocabulary and part-of-speech information.
Example:
● "running", "ran", "runs" → "run"
● "better" → "good" (context-aware)

It's smarter than simple stemming, which just chops endings heuristically (e.g.,
"running" → "run", but "better" might become "bett").
'''
# Helpers to map NLTK POS tags to WordNet's format
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN  # fallback

text = "The striped bats are hanging on their feet and they are better than before."

tokens = word_tokenize(text)
print("Tokens:", tokens)

# pos_tag is one of the most useful tools in NLTK — it assigns a Part‑of‑Speech (POS) label to each token in your text.
# In simple terms, it tells you the grammatical role of every word.
pos_tags = pos_tag(tokens)
print("Pos Tags:", pos_tags)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Token   | Stemmed | Lemmatized")
for token, pos in pos_tags:
    stem = stemmer.stem(token)
    lemma = lemmatizer.lemmatize(token, get_wordnet_pos(pos))
    print(f"{token:7} | {stem:7} | {lemma}")