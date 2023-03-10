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
i = 0
correct = 0
generatedReview = []
for reviews in df.Review:
	actualReview = df.Rating[i]
	reviews = TextBlob(reviews)
	tokens=set(reviews.words)
	# stopwords
	stop=set(stopwords.words("english"))
	tokenized = tokens-stop
	tokenized  = ', '.join(tokenized)
	# Removing stop words using set difference operation
	tokenized = TextBlob(tokenized)
	if (tokenized.sentiment.polarity > 0):
		generatedReview.append("Positive")
	else:
		generatedReview.append("Negative")
	if (generatedReview[-1] ==  actualReview):
		correct = correct + 1
	print("Review: ", generatedReview[-1], " Actual Review: ", actualReview)

	i = i + 1
print("Accuracy: ", correct/(i+1))
