from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model.h5')
scaler = joblib.load('scaler.h5')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    catar = request.args.get('catar')
    cut = request.args.get("cut")
    depth = request.args.get('depth')
    table = request.args.get("table")
    size = request.args.get("size")


    price = model.predict(scaler.transform([[catar, cut, depth, table, size ]]))[0]
    return render_template("index.html", price = price)



if __name__ == "__main__":
    app.run()
