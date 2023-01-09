#import the libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
import pickle

claimants = pd.read_csv("15.claimants.csv")
claimants.drop(["CASENUM"],inplace=True,axis = 1)
claimants = claimants.dropna()

X = claimants.iloc[:,[1,2,3,4,5]]
Y = claimants.iloc[:,0]
model = LogisticRegression()
model.fit(X,Y)

# save the model to disk
dump(model, open('15.Logistic_Model.sav', 'wb'))

# load the model from disk
loaded_model = load(open('15.Logistic_Model.sav', 'rb'))
result = loaded_model.score(X, Y)
print(result)
