from flask import Flask, request, jsonify
from saige import SAIGE

bot = SAIGE()

app = Flask(__name__)

@app.route("/qa", methods=["POST"])
def qa():
  question = request.json["question"]
  answer = bot.query(query=question)

  return jsonify({"answer": answer})

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
