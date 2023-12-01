from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from saige import SAIGE

bot = SAIGE()

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.post('/saige')
def get_answer():
    text = request.get_json().get("message")
    response = bot.query(query=text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == '__main__':
   app.run(host='0.0.0.0', port="5000", debug=False, threaded=False)
