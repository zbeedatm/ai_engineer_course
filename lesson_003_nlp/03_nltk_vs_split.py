from nltk.tokenize import word_tokenize
import nltk

'''
This downloads the Punkt sentence tokenizer, which is the standard tokenizer used for:
Splitting text into sentences
Splitting sentences into words

It includes:
Pre‑trained models
Tokenization rules
Language data

This is the one almost all tutorials use.
quiet=True simply hides the download messages.


Resource	What it is	                When to use
-----------------------------------------------------------------------------
punkt	    The actual tokenizer model	Always when tokenizing text
punkt_tab	Extra abbreviation tables	Only for advanced or multilingual use (non‑English languages that require extra abbreviation rules)
'''
nltk.download('punkt', quiet=True)


'''
split() - keeps punctuation attached and mishandles contractions;
word_tokenize - separates them intelligently.

Punctuation refers to marks like:
. period / , comma / ? question mark / ! exclamation mark / : colon / ; semicolon / - dash/hyphen / " " quotation marks / ' apostrophe / () parentheses

These marks help:
Separate sentences
Show pauses
Indicate questions
Clarify structure
Prevent confusion
'''

text = "Don't stop believing! It's amazing."

print("split():", text.split())
print("word_tokenize():", word_tokenize(text))