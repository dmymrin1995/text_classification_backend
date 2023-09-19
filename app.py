from flask import Flask, jsonify
from get_predict import ModelPredict

app = Flask(__name__)
model = ModelPredict()

model.load_model()


@app.route('/<string:review>')
def wait_string(review):
    rev = {review}
    rev_list =[]
    for s in rev:
        rev_list.append(s)
    prediction = model.get_predict(rev_list)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run()
