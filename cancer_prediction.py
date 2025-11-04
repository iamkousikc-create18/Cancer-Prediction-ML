import uvicorn
from fastapi import FastAPI
import pickle
app=FastAPI()
classifier=pickle.load(open("cancer.pkl","rb"))

@app.get('/')
def index():
    return {"Deployment":"Welcome To My Cancer Prediction Machine Learning Project"}
@app.post('/predict')
def predict(BMI:float,Smoking:int,GeneticRisk:int,PhysicalActivity:float,AlcoholIntake:float,CancerHistory:int):
    prediction=classifier.predict([[BMI,Smoking,GeneticRisk,PhysicalActivity,AlcoholIntake,CancerHistory]])
    if(prediction[0]==0):
        prediction="No Cancer"
    else:
        prediction="Cancer"
    return{
        "prediction":prediction
    }
if __name__=="__main__":
    uvicorn.run(app,host='127.0.0.1',port=5000)