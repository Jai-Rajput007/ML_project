from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
application = Flask(__name__)

app = application

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Ensure all fields are captured and handle missing values
        gender = request.form.get('gender', 'unknown')
        race_ethnicity = request.form.get('race_ethnicity', 'unknown')
        parental_level_of_education = request.form.get('parental_level_of_education', 'unknown')
        lunch = request.form.get('lunch', 'unknown')
        test_preparation_course = request.form.get('test_preparation_course', 'unknown')
        reading_score = request.form.get('reading_score', 0)
        writing_score = request.form.get('writing_score', 0)
        average = request.form.get('average', 0)

        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=int(reading_score),
            writing_score=int(writing_score),
            average=int(average),
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    application.run(host="0.0.0.0")
