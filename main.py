from flask import Flask, request, jsonify
from gpu_agent import process_query  # make sure this function exists in gpu_agent.py

app = Flask(__name__)

@app.route("/run", methods=["POST"])
def run():
    data = request.get_json() or {}
    query = data.get("query", "")
    result = process_query(query)
    return jsonify(result)
