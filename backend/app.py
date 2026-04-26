from flask import Flask, request, jsonify
from flask_cors import CORS
from salary_model import predictor

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        print("DATA RECEIVED:", data)  # debug

        salary = predictor.predict(data)

        return jsonify({
            "salary": round(float(salary), 2)
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)