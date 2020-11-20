import ppscore as pps
import pandas as pd 
import numpy as np 
import pickle
import random
import os

from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

data = pd.read_csv(os.path.join('dataset','milestone1.csv'))
print(data.describe())
data = data.drop('Last_Update',axis=1)
print(data.isnull().sum())
data = data.drop('sex',axis=1)

data['date_confirmation'] = pd.to_datetime(data['date_confirmation'], errors='coerce')
data = data.fillna(method='ffill')
data.date_confirmation = data.date_confirmation.apply(lambda x: int(x.strftime('%d%m%Y')))

pps_df = pps.predictors(data, y="outcome",cross_validation=10,random_seed=123)
pps_df = pps_df.sort_values(by="model_score",axis=0,ascending=False)
pps_df = pd.DataFrame(pps_df)
cols = pps_df[pps_df['ppscore']>0.1]['x'].values

outcome = data.outcome
data= data[cols]
data['outcome'] = outcome
data = data.drop('date_confirmation',axis=1)

le = preprocessing.LabelEncoder()
le = le.fit(data['outcome'])
data.outcome = le.transform(data.outcome.values)
# data = pd.get_dummies(data)

for col in data.columns:
	if data[col].dtype == object:
		print(col)
		le = preprocessing.LabelEncoder()
		le = le.fit(data[col])
		data[col] = le.transform(data[col].values)
		data = data.drop(col,axis=1) 

index_train = random.sample(range(len(data)),int(len(data)*.8))
train = data.iloc[index_train]
test = data.drop(index_train,axis=0)
print(len(train))
print(len(test))
y_train = train['outcome']
y_test = test['outcome']
print(y_test.value_counts())
print(y_train.value_counts())
X_train = train.drop('outcome',axis=1)
X_test = test.drop('outcome',axis=1)
# y = data.outcome
# X = data.drop('outcome', axis = 1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print('\t############XGBoost: n_estimators=10\t############')
model = XGBClassifier(learning_rate=0.001,
       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
       n_estimators=10, n_jobs=-1)
model.fit(X_train, y_train)
# model_pkl = open(os.path.join('models','xgb10_classifier.pkl'), 'wb')
# pickle.dump(model, model_pkl)
# model = pickle.load(open(os.path.join('models','xgb10_classifier.pkl'), 'rb'))
with open(os.path.join('models','xgb10_classifier.pkl'), 'wb') as f:
       pickle.dump(model, f)

with open(os.path.join('models','xgb10_classifier.pkl'), 'rb') as f:
       model = pickle.load(f)
       

predictions_train = model.predict(X_train)
print("Accuracy Train {0:.2f}%".format(100*accuracy_score(predictions_train, y_train)))
print(classification_report(y_train, predictions_train))

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print("Accuracy Test {0:.2f}%".format(100*accuracy_score(predictions, y_test)))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

print('\t############XGBoost: n_estimators=20\t############')
model = XGBClassifier(learning_rate=0.001,
       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
       n_estimators=20, n_jobs=-1)
model.fit(X_train, y_train)
# model_pkl = open(os.path.join('models','xgb20_classifier.pkl'), 'wb')
# pickle.dump(model, model_pkl)
# model = pickle.load(open(os.path.join('models','xgb20_classifier.pkl'), 'rb'))
with open(os.path.join('models','xgb20_classifier.pkl'), 'wb') as f:
       pickle.dump(model, f)

with open(os.path.join('models','xgb20_classifier.pkl'), 'rb') as f:
       model = pickle.load(f)

predictions_train = model.predict(X_train)
print("Accuracy Train {0:.2f}%".format(100*accuracy_score(predictions_train, y_train)))
print(classification_report(y_train, predictions_train))

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print("Accuracy Test {0:.2f}%".format(100*accuracy_score(predictions, y_test)))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

print('\t############XGBoost: n_estimators=100\t############')
model = XGBClassifier(learning_rate=0.001,
       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)
# model_pkl = open(os.path.join('models','xgb100_classifier.pkl'), 'wb')
with open(os.path.join('models','xgb100_classifier.pkl'), 'wb') as f:
       pickle.dump(model, f)

with open(os.path.join('models','xgb100_classifier.pkl'), 'rb') as f:
       model = pickle.load(f)
# model = pickle.load(open)
predictions_train = model.predict(X_train)
print("Accuracy Train {0:.2f}%".format(100*accuracy_score(predictions_train, y_train)))
print(classification_report(y_train, predictions_train))

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print("Accuracy Test {0:.2f}%".format(100*accuracy_score(predictions, y_test)))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


print('\t############XGBoost: max_depth=5\t############')
model = XGBClassifier(learning_rate=0.001,
       max_delta_step=0, max_depth=5, min_child_weight=1, missing=None,
       n_estimators=20, n_jobs=-1)
model.fit(X_train, y_train)
# model_pkl = open(os.path.join('models','xgb5_classifier.pkl'), 'wb')
# pickle.dump(model, model_pkl)
# model = pickle.load(open(os.path.join('models','xgb5_classifier.pkl'), 'rb'))
with open(os.path.join('models','xgb5_classifier.pkl'), 'wb') as f:
       pickle.dump(model, f)

with open(os.path.join('models','xgb5_classifier.pkl'), 'rb') as f:
       model = pickle.load(f)

predictions_train = model.predict(X_train)
print("Accuracy Train {0:.2f}%".format(100*accuracy_score(predictions_train, y_train)))
print(classification_report(y_train, predictions_train))

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print("Accuracy Test {0:.2f}%".format(100*accuracy_score(predictions, y_test)))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


print('\t############Random Forest: n_estimators = 10\t############')
model = RandomForestClassifier(n_estimators=10, random_state=0)
model.fit(X_train, y_train)
# model_pkl = open(os.path.join('models','randomforest10_classifier.pkl'), 'wb')
# pickle.dump(model, model_pkl)
# model = pickle.load(open(os.path.join('models','randomforest10_classifier.pkl'), 'rb'))
with open(os.path.join('models','randomforest10_classifier.pkl'), 'wb') as f:
       pickle.dump(model, f)

with open(os.path.join('models','randomforest10_classifier.pkl'), 'rb') as f:
       model = pickle.load(f)

predictions_train = model.predict(X_train)
print("Accuracy Train {0:.2f}%".format(100*accuracy_score(predictions_train, y_train)))
print(classification_report(y_train, predictions_train))

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print("Accuracy Test {0:.2f}%".format(100*accuracy_score(predictions, y_test)))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

print('\t############Random Forest: n_estimators = 20\t############')
model = RandomForestClassifier(n_estimators=20, random_state=0)
model.fit(X_train, y_train)
# model_pkl = open(os.path.join('models','randomforest20_classifier.pkl'), 'wb')
# pickle.dump(model, model_pkl)
# model = pickle.load(open(os.path.join('models','randomforest20_classifier.pkl'), 'rb'))
with open(os.path.join('models','randomforest20_classifier.pkl'), 'wb') as f:
       pickle.dump(model, f)

with open(os.path.join('models','randomforest20_classifier.pkl'), 'rb') as f:
       model = pickle.load(f)

predictions_train = model.predict(X_train)
print("Accuracy Train {0:.2f}%".format(100*accuracy_score(predictions_train, y_train)))
print(classification_report(y_train, predictions_train))

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print("Accuracy Test {0:.2f}%".format(100*accuracy_score(predictions, y_test)))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

print('\t############Random Forest: n_estimators = 100\t############')
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
# model_pkl = open(os.path.join('models','randomforest100_classifier.pkl'), 'wb')
# pickle.dump(model, model_pkl)
# model = pickle.load(open(os.path.join('models','randomforest100_classifier.pkl'), 'rb'))
with open(os.path.join('models','randomforest100_classifier.pkl'), 'wb') as f:
       pickle.dump(model, f)

with open(os.path.join('models','randomforest100_classifier.pkl'), 'rb') as f:
       model = pickle.load(f)

predictions_train = model.predict(X_train)
print("Accuracy Train {0:.2f}%".format(100*accuracy_score(predictions_train, y_train)))
print(classification_report(y_train, predictions_train))

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print("Accuracy Test {0:.2f}%".format(100*accuracy_score(predictions, y_test)))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


print('\t############Random Forest: max_depth = 5\t############')
model = RandomForestClassifier(n_estimators=20,max_depth=5,random_state=0)
model.fit(X_train, y_train)
# model_pkl = open(os.path.join('models','randomforest5_classifier.pkl'), 'wb')
# pickle.dump(model, model_pkl)
# model = pickle.load(open(os.path.join('models','randomforest5_classifier.pkl'), 'rb'))
with open(os.path.join('models','randomforest5_classifier.pkl'), 'wb') as f:
       pickle.dump(model, f)

with open(os.path.join('models','randomforest5_classifier.pkl'), 'rb') as f:
       model = pickle.load(f)

predictions_train = model.predict(X_train)
print("Accuracy Train {0:.2f}%".format(100*accuracy_score(predictions_train, y_train)))
print(classification_report(y_train, predictions_train))

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print("Accuracy Test {0:.2f}%".format(100*accuracy_score(predictions, y_test)))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


print('\t############KNN: n_neighbors = 4\t############')
model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train, y_train)
# model_pkl = open(os.path.join('models','knn4_classifier.pkl'), 'wb')
# pickle.dump(model, model_pkl)
# model = pickle.load(open(os.path.join('models','knn4_classifier.pkl'), 'rb'))
with open(os.path.join('models','knn4_classifier.pkl'), 'wb') as f:
       pickle.dump(model, f)

with open(os.path.join('models','knn4_classifier.pkl'), 'rb') as f:
       model = pickle.load(f)

predictions_train = model.predict(X_train)
print("Accuracy Train {0:.2f}%".format(100*accuracy_score(predictions_train, y_train)))
print(classification_report(y_train, predictions_train))

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print("Accuracy Test {0:.2f}%".format(100*accuracy_score(predictions, y_test)))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

print('\t############KNN: n_neighbors = 8\t############')
model = KNeighborsClassifier(n_neighbors=8)
model.fit(X_train, y_train)
# model_pkl = open(os.path.join('models','knn8_classifier.pkl'), 'wb')
# pickle.dump(model, model_pkl)
# model = pickle.load(open(os.path.join('models','knn8_classifier.pkl'), 'rb'))
with open(os.path.join('models','knn8_classifier.pkl'), 'wb') as f:
       pickle.dump(model, f)

with open(os.path.join('models','knn8_classifier.pkl'), 'rb') as f:
       model = pickle.load(f)

predictions_train = model.predict(X_train)
print("Accuracy Train {0:.2f}%".format(100*accuracy_score(predictions_train, y_train)))
print(classification_report(y_train, predictions_train))

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print("Accuracy Test {0:.2f}%".format(100*accuracy_score(predictions, y_test)))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


print('\t############KNN: n_neighbors = 20\t############')
model = KNeighborsClassifier(n_neighbors=20)
model.fit(X_train, y_train)
# model_pkl = open(os.path.join('models','knn20_classifier.pkl'), 'wb')
# pickle.dump(model, model_pkl)
# model = pickle.load(open(os.path.join('models','knn20_classifier.pkl'), 'rb'))
with open(os.path.join('models','knn20_classifier.pkl'), 'wb') as f:
       pickle.dump(model, f)

with open(os.path.join('models','knn20_classifier.pkl'), 'rb') as f:
       model = pickle.load(f)

predictions_train = model.predict(X_train)
print("Accuracy Train {0:.2f}%".format(100*accuracy_score(predictions_train, y_train)))
print(classification_report(y_train, predictions_train))

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print("Accuracy Test {0:.2f}%".format(100*accuracy_score(predictions, y_test)))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


print('\t############KNN: weights = distance\t############')
model = KNeighborsClassifier(weights='distance')
model.fit(X_train, y_train)
# model_pkl = open(os.path.join('models','knn_distance_classifier.pkl'), 'wb')
# pickle.dump(model, model_pkl)
# model = pickle.load(open(os.path.join('models','knn_distance_classifier.pkl'), 'rb'))
with open(os.path.join('models','knn_distance_classifier.pkl'), 'wb') as f:
       pickle.dump(model, f)

with open(os.path.join('models','knn_distance_classifier.pkl'), 'rb') as f:
       model = pickle.load(f)

predictions_train = model.predict(X_train)
print("Accuracy Train {0:.2f}%".format(100*accuracy_score(predictions_train, y_train)))
print(classification_report(y_train, predictions_train))

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print("Accuracy Test {0:.2f}%".format(100*accuracy_score(predictions, y_test)))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))







