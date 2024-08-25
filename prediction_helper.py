import pandas as pd
from joblib import load


model_rest = load("Artifacts\\model_rest.joblib")
model_young = load("Artifacts\\model_young.joblib")

scaler_rest = load("Artifacts\\scaler_rest.joblib")
scaler_young = load("Artifacts\\scaler_young.joblib")

def calculated_normalised_risk_score(medical_history):
    risk_scores ={
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    ## Split the medical history into lower str
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)

    max_score = 14
    min_score = 0

    normalised_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalised_risk_score


def preprocess_input(input_dict):
    expected_columns = ['age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
       'genetical_risk', 'normalised_risk_score', 'gender_Male',
       'region_Northwest', 'region_Southeast', 'region_Southwest',
       'marital_status_Unmarried', 'bmi_category_Obesity',
       'bmi_category_Overweight', 'bmi_category_Underweight',
       'smoking_status_Occasional', 'smoking_status_Regular',
       'employment_status_Salaried', 'employment_status_Self-Employed']
    
    insurance_plan_encoding = {'Bronze':1, 'Silver':2, 'Gold':3}
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        if key == 'Region' and value == 'Northwest':
            df['region_Northwest'] = 1
        if key == 'Region' and value == 'Southeast':
            df['region_Southeast'] = 1
        if key == 'Region' and value == 'Southwest':
            df['region_Southwest'] = 1
        if key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        if key == 'BMI Category' and value == 'Obesity':
            df['bmi_category_Obesity'] = 1
        if key == 'BMI Category' and value == 'Overweight':
            df['bmi_category_Overweight'] = 1
        if key == 'BMI Category' and value == 'Underweight':
            df['bmi_category_Underweight'] = 1
        if key == 'Smoking Status' and value == 'Occasional':
            df['smoking_status_Occasional'] = 1
        if key == 'Smoking Status' and value == 'Regular':
            df['smoking_status_Regular'] = 1
        if key == 'Employment Status' and value == 'Salaried':
            df['employment_status_Salaried'] = 1
        if key == 'Employment  Status' and value == 'Self-Employed':
            df['employment_status_Self-Employed'] = 1
        if key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value,1)
        if key == 'Age':
            df['age'] = value
        if key == 'Number of Dependants':
            df['number_of_dependants'] = value
        if key == 'Income in Lakhs':
            df['income_lakhs'] = value
        if key == 'Genetical Risk':
            df['genetical_risk'] = value

    df['normalised_risk_score'] = calculated_normalised_risk_score(input_dict['Medical History'])
    df = handle_scaling(input_dict['Age'],df)
    return df

def handle_scaling(age, df):
    if age<=25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']
    
    df['income_level'] = None

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis ='columns', inplace = True)

    return df


def predict(input_dict):
    input_df = preprocess_input(input_dict)
    
    if input_dict['Age']<=25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction)