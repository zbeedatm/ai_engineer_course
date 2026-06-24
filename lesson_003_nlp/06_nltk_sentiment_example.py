from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)

positive = {"good", "great", "happy", "fun", "love", "powerful"}
negative = {"bad", "sad", "hate", "terrible", "hard"}

def sentiment(text):
    tokens = [t.lower() for t in word_tokenize(text)]
    pos = sum(1 for t in tokens if t in positive)
    neg = sum(1 for t in tokens if t in negative)
    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    else:
        return "Neutral"

print(sentiment("I love using Python for NLP. It's fun and powerful!"))
print(sentiment("This is a terrible example, I hate bugs."))