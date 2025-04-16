import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import joblib

dataset = pd.read_csv('/data/dataset.csv')   #import dataset

X = dataset.drop('Class', axis=1)   #splitting into features and label
y = dataset['Class']

# Splitting del dataset in train_set e test_set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# SVM fitting with train set
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X=X_train, y=y_train)

# Testing
y_pred = classifier.predict(X_test)

# Cofusion matrix
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Print
print(y_pred)
print(cm)
print(cr)

# Exprot model
filename = 'model.sav'
joblib.dump(classifier, filename)
print('Model exported!')

