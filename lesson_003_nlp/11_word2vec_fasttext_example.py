# fast_text_example.py

from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)

'''
Imagine words are toys
You have two toy boxes: one called Word2Vec and one called FastText.
Word2Vec box:
● It only knows whole toys it was shown before.
● If you give it a new toy that looks almost the same but is missing a piece (“playng” instead of “playing”), it says: “I don’t know that
one.” (That’s why it failed.)
FastText box:
● It doesn’t just remember whole toys. It also knows the little pieces the toys are made of (like LEGO bricks).
● So if you give it “playng” (missing an “i”), it sees the same pieces as “playing” and says: “Oh, that’s close!” — that’s why similarity
is high (0.62).


What about “playing” and “soccer”?
When you ask both boxes: “Are ‘playing’ and ‘soccer’ similar?”
● They are kind of related (both are about games/sports), but in this tiny toy example they weren’t used together enough, so the
boxes don’t think they’re very close.
● The negative numbers (like -0.16 or -0.41) just mean “not similar” in their internal math — they’re not best friends. If the number
were big and positive, that would mean “these words are close buddies.”

Summary:
● Word2Vec: knows exact words only. “playng” is unknown.
● FastText: builds words from pieces, so even a slightly broken word (“playng”) still makes sense.
● “playing” vs “soccer” isn’t very close in this small example, so similarity is low (even negative).
'''

# Tiny corpus
sentences = [
    "I like playing soccer.",
    "He enjoys playing football.",
    "She loves sports and games.",
]

tokenized = [word_tokenize(s.lower()) for s in sentences]

# Train Word2Vec and FastText on same data
wv = Word2Vec(tokenized, vector_size=20, window=2, min_count=1, sg=1)
ft = FastText(tokenized, vector_size=20, window=2, min_count=1, sg=1)

print("Word2Vec similarity between 'playing' and 'soccer':")
print(wv.wv.similarity('playing', 'soccer'))

print("FastText similarity between 'playing' and 'soccer':")
print(ft.wv.similarity('playing', 'soccer'))

# Misspelled word example
misspelled = 'playng'
print(f"\nWord2Vec sees '{misspelled}' as:")
print(misspelled in wv.wv.key_to_index)

print(f"FastText sees '{misspelled}' as:")
print(misspelled in ft.wv.key_to_index)

print(f"Word2Vec similarity ('playng' vs 'playing'):")
try:
    print(wv.wv.similarity(misspelled, 'playing'))
except KeyError:
    print("Word2Vec: 'playng' is out of vocabulary (no vector)")

print(f"FastText knows 'playng' by subwords and similarity to 'playing':")
print(ft.wv.similarity(misspelled, 'playing'))
