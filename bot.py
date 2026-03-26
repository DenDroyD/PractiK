import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "OK"

@app.route('/webhook', methods=['POST'])
def webhook():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    app.run(host="0.0.0.0", port=port)