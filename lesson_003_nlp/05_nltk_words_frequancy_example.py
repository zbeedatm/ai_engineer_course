from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

nltk.download('punkt', quiet=True)

'''
Word Frequency / Bag of Words (BoW)

● Tokenization splits into words and punctuation.
● Lowercasing normalizes.
● Bag of Words representation: text → frequency vector.
● Foundation for simple classification (e.g., spam vs. not-spam) or similarity.
'''
text = "Python is great, Python is simple, NLP with Python is powerful."
tokens = [t.lower() for t in word_tokenize(text)]
frequency = Counter(tokens) # Bag of Words

print("Original tokens:", tokens)
print("Frequencies:", frequency.most_common(10))