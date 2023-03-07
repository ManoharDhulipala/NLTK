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
for reviews in df.Review:
	reviews = TextBlob(reviews)
	print(reviews)
	print(reviews.sentiment)
