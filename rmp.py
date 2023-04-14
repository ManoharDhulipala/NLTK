from textblob import Word
import sys
import matplotlib
#matplotlib.use('Agg')
matplotlib.rc('font',size=8)
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import seaborn as sns
import matplotlib.pylab as pl
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score 
from textblob import TextBlob
df = pandas.read_csv("RateMyProfData.csv")
from nltk.corpus import stopwords
from nltk.corpus import wordnet

def get_word_similarity(word1, word2):
    synset1 = wordnet.synsets(word1)
    synset2 = wordnet.synsets(word2)
    if synset1 and synset2:
        wup_similarity = synset1[0].wup_similarity(synset2[0])
        if wup_similarity:
            return wup_similarity
    return 0


i = 0
correct = 0
falseNegative = 0
falsePositive = 0
Positive = 0
Negative = 0
generatedReview = []
ActualReview = []
wordList = []
for reviews in df.Review:
	actualReview = df.Rating[i]
	reviews = TextBlob(reviews)
	tokens=set(reviews.words)
	# stopwords
	stop=set(stopwords.words("english"))
	tokenized = tokens-stop
	tokenized  = ', '.join(tokenized)
	#print(tokenized)
	# Removing stop words using set difference operation
	tokenized = TextBlob(tokenized)
	words = tokenized.words
	for word in words:
		wordList.append(word)
	if (tokenized.sentiment.polarity > 0):
		generatedReview.append("Positive")
	else:
		generatedReview.append("Negative")
	if (generatedReview[-1] ==  actualReview):
		correct = correct + 1
	if (generatedReview[-1] == "Negative" and actualReview == "Positive"):
		falseNegative = falseNegative+1
	if (generatedReview[-1] == "Positive" and actualReview == "Negative"):
		falsePositive = falsePositive+1
	if (generatedReview[-1] == "Positive" and actualReview == "Positive"):
		Positive = Positive+1
	if (generatedReview[-1] == "Negative" and actualReview == "Negative"):
		Negative = Negative+1
		
			
		
	print("Review: ", generatedReview[-1], " Actual Review: ", actualReview)
	ActualReview.append(actualReview)
	i = i + 1

positive = 0
negative = 0
print([Positive, falsePositive])
print([falseNegative, Negative])
print("Accuracy: ", correct/(i+1))
print("Precision: ",Positive/(Positive+falsePositive)) 


y = np.array([positive, negative])
mylabels = ["Positive Reviews", "Negative Reviews"]

plt.pie(y, labels = mylabels)
plt.show()
