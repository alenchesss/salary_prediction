import os, pickle, numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE = os.getcwd()

# Load model
with open(os.path.join(BASE, 'model/salary_model.pkl'), 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
log_scale = model_data['log_scale']

with open(os.path.join(BASE, 'model/label_encoders.pkl'), 'rb') as f:
    encoders = pickle.load(f)

# Build category -> job titles mapping
import pandas as pd
df = pd.read_csv(os.path.join(BASE, 'data/processed/clean_jobs.csv'))
category_jobs = df.groupby('job_category')['job_title'].unique().apply(sorted).to_dict()
category_jobs = {k: list(v) for k, v in category_jobs.items()}

print("Server ready!")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        title_enc = encoders['title'].transform([data['job_title']])[0]
        country_enc = encoders['country'].transform([data['country']])[0]
        level_enc = encoders['level'].transform([data['experience_level']])[0]
        category_enc = encoders['category'].transform([data['category']])[0]
        forecast = float(data.get('forecast_years', 0))

        X = np.array([[title_enc, country_enc, level_enc, category_enc, 2024]])
        pred = model.predict(X)[0]

        if log_scale:
            base_salary = float(np.expm1(pred))
        else:
            base_salary = float(pred)

        predicted = round(base_salary * (1 + 0.03 * forecast), 2)
        return jsonify({'salary': predicted})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/options', methods=['GET'])
def options():
    return jsonify({
        'categories': sorted(list(category_jobs.keys())),
        'category_jobs': category_jobs,
        'countries': list(encoders['country'].classes_),
        'experience_levels': list(encoders['level'].classes_)
    })

if __name__ == '__main__':
    app.run(debug=False, port=5001)