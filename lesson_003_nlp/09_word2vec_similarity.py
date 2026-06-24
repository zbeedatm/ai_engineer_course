from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Downloads (run once)
nltk.download('punkt', quiet=True)

sentences = [
    "I love natural language processing",
    "Language models capture semantic meaning",
    "Word embeddings help with similarity tasks",
    "Deep learning creates vector representations of words",
    "We can find similar words using Word2Vec"
]

# Tokenize sentences
tokenized = [word_tokenize(s.lower()) for s in sentences]

# Train Word2Vec model
model = Word2Vec(
    tokenized,
    vector_size=50,
    window=2,
    min_count=1,
    sg=1  # Skip-gram
)

# Query similar words
print("Words most similar to 'language':", model.wv.most_similar("language", topn=3))
print("Words most similar to 'deep':", model.wv.most_similar("deep", topn=3))
