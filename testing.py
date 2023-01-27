import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

text = """Professor Cuthbert Calculus is an intelligent professor but he has difficulty hearing and goes off on tangents during lectures. His exams are difficult and do not match with lecture."""

#tokenize places each word in an array
wordArray = nltk.word_tokenize(text)
#print(wordArray)

#Stemming produces the root word via utilizing the stem of the word
stemArray = []
stemmer = PorterStemmer()
for word in wordArray:
	stemWord = stemmer.stem(word)
	stemArray.append(stemWord)

#print(stemArray)

#Lemmetization produces the root word using the context of the word
lemmatizeArray = []
lemmatizer = WordNetLemmatizer()
for word in wordArray:
	lemmatizeWord = lemmatizer.lemmatize(word)
	lemmatizeArray.append(lemmatizeWord)

#print(lemmatizeArray)

#Parts of Speech example
POSArray = []
for word in wordArray:
	POSArray.append(nltk.pos_tag([word]))

print(POSArray)
