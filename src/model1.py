import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle

DATA_DIR = '../results/'
TEST_FILENAME = DATA_DIR + 'cases_train_processed.csv'

train = pd.read_csv(TEST_FILENAME, parse_dates=['date_confirmation'])

train_data = pd.concat([train, pd.get_dummies(train['sex'], prefix='sex')], axis=1)
train_data['confirmation_day'] = (train_data.date_confirmation - train_data.date_confirmation.min()).dt.days

X = train_data[
    ['age', 'sex_unknown', 'sex_female', 'sex_male', 'latitude', 'longitude', 'confirmation_day', 'Lat', 'Long_',
     'Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio']]

y = train_data['outcome']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

'''
# Uncomment to build the accuracy regression

for n in range (100, 1500, 100):
    clf = GradientBoostingClassifier(n_estimators=n)
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train), clf.score(X_valid, y_valid), n)
'''

clf = GradientBoostingClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train), clf.score(X_valid, y_valid))
pickle.dump(clf, open("../models/model_gbc.pkl", "wb"))