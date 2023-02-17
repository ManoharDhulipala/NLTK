from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

clarityRatings = [0, 1, 8, 74, 138]
organizationRatings = [1, 4, 16, 58, 162]

x = np.array(clarityRatings)
y = np.array(organizationRatings)

x = np.reshape(x, (0,1,8,74,138))
y = np.reshape(y, (1,4,16,58,162))


xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.2, random_state = 42)

clf = MultinomialNB()
clf.fit(xTrain, yTrain)

y_pred = clf.predict(xTest)

mae = mean_absolute_error(yTest,y_pred)
print("Mean absolute error: ", mae)
