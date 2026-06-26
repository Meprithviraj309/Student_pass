import joblib
import numpy as np


model = joblib.load("model/model.pkl")
hours=[[5]]

prediction = model.predict(hours)

if prediction[0] == 1:
    print("PASS")
else:
    print("FAIL")
