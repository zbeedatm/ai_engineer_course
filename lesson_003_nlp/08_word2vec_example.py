import gensim.downloader as api

# this is a one‑time download + cache
'''
Loads a pre-trained GloVe embedding model (“glove-wiki-gigaword-50”).
● “50” means each word is represented as a 50-dimensional vector.
● The first time you run this it downloads the model; afterward it’s cached
locally.
● These vectors were trained on Wikipedia + Gigaword corpus to capture
word meaning by context.
● print("king ~ queen:", model.similarity("king", "queen"))
'''
model = api.load("glove-wiki-gigaword-50")


'''
Word2Vec with Gensim ::

Computes the cosine similarity between the vectors for "king" and "queen".
● Cosine similarity is a number between -1 and 1 that measures how close the
two vectors point in direction; higher means more semantically similar.
● So this prints something like:
king ~ queen: 0.73 (actual value may vary slightly)
● Interpretation: “king” and “queen” are related in meaning, so their vectors are
close.
'''

print("king ~ queen:", model.similarity("king", "queen"))
print("paris + germany - france →",
      [w for w,_ in model.most_similar(
          positive=["paris","germany"],
          negative=["france"],
          topn=100)])