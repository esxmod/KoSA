from flask import Flask, render_template, request, send_from_directory

from models.kobert import TFKoBertModel
from modules.tokenizer import Tokenizer


app = Flask(__name__,
            static_folder='build/static',
            template_folder='build')

model = TFKoBertModel('kykim/bert-kor-base', 128, 2)
model.load_weights('saved_models/best_weights.h5')

tokenizer = Tokenizer('kykim/bert-kor-base', 128)

memo = {}


def predict_sentence(sentence):
    result = tokenizer.tokenize(sentence, True)
    result = model.predict(result)
    result = result.tolist()[0]
    return result


@app.route('/api/predict', methods=['POST'])
def predict():
    params = request.get_json()
    query = params['query']

    if query not in memo:
        memo[query] = predict_sentence(query)

    return {
        'result': memo[query]
    }


@app.route("/manifest.json")
def manifest():
    return send_from_directory('build', 'manifest.json')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('build', 'favicon.ico')


@app.route('/')
@app.route('/<path:path>')
def index(path=None):
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
