import flask
from flask import Flask

app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])
def home():
    return "Started App"


@app.route('/train',methods = ['GET','POST'])
def training():
    return "Started Training"


@app.route('/predict',methods = ['GET','POST'])
def predict():
    return "Started Predicting"


if __name__=="__main__":
    app.run(debug=True)