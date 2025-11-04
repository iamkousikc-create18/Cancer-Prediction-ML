🧬 Cancer Prediction using Machine Learning & FastAPI

📌 Project Overview

This project predicts whether a person is likely to develop cancer based on medical and lifestyle factors such as age, smoking habits, alcohol intake, physical activity, and genetic history. Multiple machine learning models were trained and compared. The best model was saved using Pickle and deployed using FastAPI for real-time predictions. The dataset was also normalized using Min-Max Scaler to improve model performance.


---

🛠 Technologies Used

Language: Python
Deployment Framework: FastAPI, Uvicorn
Libraries Used:

Data Handling – pandas, numpy

Visualization – matplotlib, seaborn

Machine Learning – scikit-learn

Normalization – MinMaxScaler

Model Saving – pickle



---

🤖 Machine Learning Models Applied

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

SVC (Support Vector Classifier)

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes

Bagging Classifier

Voting Classifier

Stacking Classifier


The best model was selected based on accuracy, confusion matrix, precision, recall, and F1-score, and saved as cancer.pkl.


---

📊 Workflow

1. Data Loading & Cleaning

Imported dataset using pandas

Removed missing values and duplicates

Converted categorical values into numerical format



2. Feature Scaling / Normalization

Applied Min-Max Scaler to scale numerical features between 0 and 1


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X)


3. Exploratory Data Analysis (EDA)

Correlation heatmap, histograms, box plots, pair plots



4. Model Training & Evaluation

Trained multiple classification algorithms

Evaluated using accuracy, classification report, confusion matrix



5. Model Saving

import pickle
pickle.dump(best_model, open('cancer.pkl', 'wb'))


6. FastAPI Deployment

Loaded model using pickle

Created routes / (welcome) and /predict for cancer prediction

Input: age, smoking, alcohol intake, physical activity, genetic history, etc.

Output: "Cancer" or "No Cancer"



7. Run the API

uvicorn cancer:app --reload

Test at:
✅ Swagger UI – http://127.0.0.1:8000/docs
✅ Redoc – http://127.0.0.1:8000/redoc




---

📁 Project Structure

├── cancer.py               # FastAPI deployment file  
├── cancer.pkl              # Saved ML model  
├── cancer_prediction.ipynb # Model building & analysis notebook  
├── dataset.csv  
├── requirements.txt  
├── README.md


---

🚀 Future Enhancements

Add Streamlit / HTML frontend

Deploy on Render / AWS / Railway

Add database for storing patient history

Use GridSearchCV/RandomizedSearchCV for tuning



---

✅ Conclusion

This project shows a complete pipeline — from data preprocessing (Min-Max Scaling), model training, evaluation, saving with pickle, to FastAPI deployment. It enables fast and accurate cancer prediction, making it useful for healthcare applications.
