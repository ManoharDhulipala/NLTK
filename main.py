import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer, sent_tokenize

example = ["Professor Jones is very nice. He does a great job at lecturing and his exams are fair. One thing that I wish he would do is pause more often because I can't write my notes thoroughly during lecture.", "I had the privilege of taking a course with Professor Jones, and it was an amazing experience. They were incredibly knowledgeable and made the material easy to understand. Professor Jones made the classroom environment engaging and fun, and I always looked forward to their lectures. They were also very approachable and always willing to help students who had questions or needed extra support. I highly recommend taking a class with Professor Jones to anyone looking for an exceptional learning experience."]

tokens = TweetTokenizer()
positiveList = []
negativeList = []

for reviews in example:
	tokens_sentences = [tokens.tokenize(t) for t in nltk.sent_tokenize(reviews)]

	joinedSentences = []
	for sentences in tokens_sentences:
	    sentences = " ".join(sentences)
	    joinedSentences.append(sentences)

	sia = SentimentIntensityAnalyzer()
	for sentence in joinedSentences:
		data = sia.polarity_scores(sentence)
		print(sentence, " ", data)
	if data["pos"] > 0.2:
		positiveList.append(sentence)
	else:
		negativeList.append(sentence)

print("Postive Statements: ", positiveList)
print("Negative Statements: ", negativeList)
