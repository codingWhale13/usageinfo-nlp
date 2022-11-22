from flask import Flask

app = Flask(__name__)

@app.route('/usecase_replacement/<string:usecase>/')
def getReplacement(usecase):
    replacements = [
        "test",
        "test2"
    ]
    
    return {usecase: replacements}

if __name__ == '__main__':
    app.run(debug=True)