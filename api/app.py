from flask import Flask, request, jsonify
from journal import fun
from flask_cors import CORS 
app = Flask(__name__)
CORS(app)
@app.route('/analyze', methods=['POST'])
def analyze():
    journal_text = request.form.get('journalText')
    positive_words, negative_words, focused_words,op= fun(journal_text)
    response = {
        'positive_words': positive_words,
        'negative_words': negative_words,
        'focused_words': focused_words,
        'overall_polarity': op
    }
    return jsonify(response)
