from fastapi import FastAPI
import joblib
model = joblib.load('iris_model')

app = FastAPI()

@app.get("/")
def predict_iris(sl:float,sw:float,pl:float,pw:float):
    result = model.predict([[sl,sw,pl,pw]])
    return {'Predicted class is':int(result[0])}