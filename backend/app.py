from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/", methods = ["GET"])
def home():
    return "Ok"


@app.route("/message", methods = ["POST"])
def message():
    data = request.get_json()
    user_text = data.get("text", "")
    
    response_text = f"Flask received : {user_text}"

    return jsonify({"text": response_text})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)