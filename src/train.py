import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

import os 


# def train_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = DecisionTreeClassifier()
#     model.fit(X_train, y_train)
#     joblib.dump(model, "model.pkl")
#     return model

X =np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

y=np.array([0, 0, 0, 0, 1, 1, 1, 1])    

model = DecisionTreeClassifier()
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("Model trained and saved as model/model.pkl")