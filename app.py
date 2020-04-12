from flask import Flask, render_template, redirect, request
from sklearn.externals import joblib

# __name__ == __main__
app = Flask(__name__)

model=joblib.load("model.pkl")

@app.route('/')
def hello():
	return render_template("index.html")

@app.route('/', methods = ['POST'])
def submit_data():
	if request.method == 'POST':
		hrs = float(request.form['hours'])
		marks = str(model.predict([[hrs]])[0][0])
	return render_template("index.html",ur_marks=marks)

if __name__ == '__main__':
	# app.debug = True
	app.run(debug = True)
