import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer, sent_tokenize

example = """Professor Jones is very nice. He does a great job at lecturing and his exams are fair. 
One thing that I wish he would do is pause more often because I can't write my notes thoroughly during lecture."""

tokens = TweetTokenizer()
tokens_sentences = [tokens.tokenize(t) for t in nltk.sent_tokenize(example)]
joinedSentences = []

for sentences in tokens_sentences:
	sentences = " ".join(sentences)
	joinedSentences.append(sentences)

sia = SentimentIntensityAnalyzer()
positiveList = []
negativeList = []
for sentence in joinedSentences:
	data = sia.polarity_scores(sentence)
	if data["pos"] > 0.3:
		positiveList.append(sentence)
	else:
		negativeList.append(sentence)
	
print(positiveList)
print(negativeList)
