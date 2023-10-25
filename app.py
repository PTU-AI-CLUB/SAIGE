from flask import Flask, jsonify, request, render_template, jsonify
from flask_cors import CORS
from saige import SAIGE

bot = SAIGE()

app = Flask(__name__)
CORS(app)

@app.post('/saige')
def get_answer():
    text = request.get_json().get("message")
    response = bot.query(query=text)
    # response = "Hi"
    message = {"answer" : response}
    return jsonify(message)

if __name__ == '__main__':
   app.run(host='LOCALHOST', port="PORT", debug=False, threaded=False)