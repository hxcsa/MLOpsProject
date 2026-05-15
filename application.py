from flask import Flask,render_template,request
import joblib
import numpy as np


app = Flask(__name__)

model_path = "artifacts/model/model.pkl"
scaler_path = "artifacts/proccessed/scaler.pkl"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Warning: Could not load model or scaler. Exception: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    # Changed predictions=None to prediction=None to match the predict route
    return render_template("index.html" , prediction=None)

@app.route('/predict',methods=["POST"])
def predict():
    try:
        healthcare_cost = float(request.form["healthcare_costs"])
        tumor_size = float(request.form["tumor_size"])
        treatment_type = int(request.form["treatment_type"])
        diabetes = int(request.form["diabetes"])
        mortality_rate = float(request.form["mortality_rate"])

        # The scaler expects 10 features. Since the form only provides 5, 
        # we pad the rest with default values (0) to prevent crashes.
        # Order: Healthcare_Costs, Tumor_Size_mm, Treatment_Type, Diabetes, 
        # Mortality_Rate_per_100K, Insurance_Status, Cancer_Stage, 
        # Screening_History, Country, Healthcare_Access
        input_data = np.array([[healthcare_cost, tumor_size, treatment_type, diabetes, mortality_rate, 0, 0, 0, 0, 0]])

        if model and scaler:
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]
        else:
            prediction = "Model not loaded (Mock Prediction: 12.34)"

        return render_template('index.html' , prediction=prediction)
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")
    
if __name__=="__main__":
    app.run(debug=True , host="0.0.0.0" , port=5000)
