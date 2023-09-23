from flask import Flask, jsonify, request
from get_predict import ModelPredict

app = Flask(__name__)
model = ModelPredict()


@app.route('/review_predict/<string:review>', methods = ['GET', 'POST'])
def wait_string(review):
    if request.method == 'POST':
        rev = {review}
        rev_list =[]
        for s in rev:
            rev_list.append(s)
        prediction = model.get_predict(rev_list)
        return jsonify(prediction.tolist())


if __name__ == '__main__':
    model.load_model()
    app.run()
