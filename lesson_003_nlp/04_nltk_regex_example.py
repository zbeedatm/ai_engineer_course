from nltk.tokenize import RegexpTokenizer

'''
● By default word_tokenize splits on punctuation, but often you want to treat things like
“aren’t” or em‐dashes more gracefully.
● We use a regex \w+ to pull out only letters and digits.
● Contractions become split (“aren” + “t”), which you may or may not prefer.
'''
# r'\w+' matches alphanumeric words only (drops punctuations entirely)
tokenizer = RegexpTokenizer(r'\w+')

sentence = "Children aren't playing soccer - they've gone inside!"
tokens = tokenizer.tokenize(sentence)
print(tokens)
