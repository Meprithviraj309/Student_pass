import joblib
from flask import Flask




app = Flask(__name__)
model = joblib.load("model/model.pkl")

@app.route("/")

def home():
    prediction = model.predict([[5]])
    if prediction[0] == 1:
        return "PASS"
    else:
        return "FAIL"


if __name__ == "__main__":
    app.run(debug=True)



