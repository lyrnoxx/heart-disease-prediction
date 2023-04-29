import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained models
models = [joblib.load(f'{model}_classifier.joblib') for model in ['knn', 'svc', 'dt', 'rf']]

# Load and preprocess dataset
dataset = pd.read_csv('processed_dataset.csv').drop('target', axis=1)
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol','thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

# Define function to get user input and make prediction
def predict_heart_disease(user_input):
    # Set default values for remaining features
    default_values = {column: 0.0 for column in dataset.columns if column not in user_input}
    user_input.update(default_values)

    # Scale the user input
    input_data = pd.DataFrame(user_input, index=[0])
    input_data[columns_to_scale] = standardScaler.transform(input_data[columns_to_scale])

    # Make predictions using each model
    predictions = [model.predict(input_data)[0] for model in models]

    # Return the predictions
    return predictions

# Set up the Streamlit app
st.set_page_config(page_title="Heart Disease Prediction App")
st.title('Heart Disease Prediction')
st.write('This app predicts whether or not you have heart disease based on your personal information.')
st.write('Please fill out the form below to get started.')

# Get user input
user_input = {}
num_columns = min(10, len(dataset.columns))
for i in range(num_columns):
    column_name = dataset.columns[i]
    column_values = dataset[column_name].unique()
    column_values.sort()
    user_input[column_name] = st.selectbox(f'{column_name}:', column_values)

if st.button('Predict'):
    predictions = predict_heart_disease(user_input)
    results = ['Positive' if pred == 1 else 'Negative' for pred in predictions]
    st.write('Results:')
    st.write('KNN: ' + results[0])
    st.write('SVC: ' + results[1])
    st.write('DT: ' + results[2])
    st.write('RF: ' + results[3])

    # Display a chart of the predicted outcomes
    chart_data = pd.DataFrame({'Model': ['KNN', 'SVC', 'DT', 'RF'], 'Outcome': results})
    st.write('Chart:')
    st.bar_chart(chart_data.set_index('Model'))

    # Display a summary of the input values
    st.write('Input Values:')
    for key, value in user_input.items():
        st.write(f'{key}: {value}')

