import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer, sent_tokenize
positiveList = []
negativeList = []
def sentimentalAnalysis(example):
	tokens = TweetTokenizer()
	tokens_sentences = [tokens.tokenize(t) for t in nltk.sent_tokenize(example)]
	joinedSentences = []

	for sentences in tokens_sentences:
		sentences = " ".join(sentences)
		joinedSentences.append(sentences)

	sia = SentimentIntensityAnalyzer()
	for sentence in joinedSentences:
		data = sia.polarity_scores(sentence)
		print(sentence, " Polarity Score: ", data)
	if data["pos"] > 0.3:
		positiveList.append(sentence)
	else:
		negativeList.append(sentence)
	


lines = []

with open("negativeExamples.txt") as NumFile:

	[lines.append(line.strip("\n")) for line in NumFile.readlines()]

for sentences in lines:
	sentimentalAnalysis(sentences)

print("Positive Lines: ", positiveList)
print("Negative Lines: ", negativeList)
