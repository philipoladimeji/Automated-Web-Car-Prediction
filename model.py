import pandas as pd
import dill as pickle
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
from datetime import date
import datetime as DT
import io
from scipy import stats
from sklearn.metrics import accuracy_score
import xgboost
import csv as csv
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score,KFold

from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('carprices.csv')


X = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))