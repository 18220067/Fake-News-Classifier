from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
from newspaper import Article


app = FastAPI()
 
@app.get('/')
def home():
    return {'text' : 'Welcome to News Classifier'}

class request_body(BaseModel):
    link : str 

@app.post('/predict')
def predict(data: request_body):
    clf = joblib.load('./app/model/model_fakenewsclassifier.pkl')
    article=Article(data.link)
    article.download()
    article.parse()
    to_predict = article.title + ' ' + article.text
    to_predict = to_predict.lower()
    prediction = clf.predict([to_predict])
    if(prediction == 0):
        res = 'Fake'
    elif(prediction == 1):
        res = 'Real'
    return {'This News is {}'.format(res) }


# if __name__ == '__main__':

#     uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)