from functools import wraps
from flask import Flask, render_template, request, send_from_directory
from examples import SentimentAnalysis, Summarization


app = Flask(__name__,
            static_folder='build/static',
            template_folder='build')

sa = SentimentAnalysis()
summ = Summarization()


def get_query(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        params = request.get_json()
        query = params['query']
        return func(query=query, *args, **kwargs)

    return wrapper


@app.route('/api/predict/sa', methods=['POST'])
@get_query
def predict_sa(query):
    return {
        'result': sa.predict(query)
    }


@app.route('/api/predict/sum', methods=['POST'])
@get_query
def predict_summ(query):
    return {
        'result': summ.predict(query)
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
