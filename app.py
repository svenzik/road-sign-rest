#!flask/bin/python
from flask import Flask, jsonify

app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'name': u'First'
    },
    {
        'id': 2,
        'name': u'Second'
    }
]

@app.route('/')
def index():
    #return "Hello, World!"
    return jsonify({'tasks': tasks})

if __name__ == '__main__':
    app.run(debug=True)

