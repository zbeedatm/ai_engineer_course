import nltk
from nltk.corpus import wordnet
from gensim.downloader import load
from scipy.spatial.distance import cosine

'''
Synonym Ranking via WordNet + GloVe
● Uses NLTK’s WordNet to fetch candidate synonyms
● Ranks them by GloVe cosine similarity to the original word
'''
# Download WordNet (run once)
nltk.download('wordnet')

# Load pre-trained GloVe embeddings
model = load("glove-wiki-gigaword-50")

# usage
def top_synonyms(word, n=5):
    # gather lemmas of all synsets
    lemmas = set(
        t.name().replace('_', ' ')
        for syn in wordnet.synsets(word)
        for t in syn.lemmas()
        if t.name().lower() != word.lower()
    )

    # filter to those in the model
    candidates = [w for w in lemmas if w in model]

    # score by similarity
    scored = sorted(
        [(w, 1 - cosine(model[word], model[w])) for w in candidates],
        key=lambda x: -x[1]
    )

    return scored[:n]

print(top_synonyms('happy'))  # => top_synonyms("happy")
