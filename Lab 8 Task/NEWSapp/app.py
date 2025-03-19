from flask import Flask , render_template
import requests

app = Flask(__name__)

NEWS_API_KEY = 'feedb05cb63443b1b87ffdcd604ac2ba'

    
Artical_API = 'https://newsapi.org/v2/everything?q=tesla&from=2025-02-19&sortBy=publishedAt&apiKey='

@app.route('/', methods =['GET'])

def News():
    parameter = f"{Artical_API}{NEWS_API_KEY}"
    response  = requests.get(parameter)
    data      = response.json()

    return render_template("index.html", articles=data.get("articles", []))


if __name__ == "__main__":
    app.run(debug=False)
