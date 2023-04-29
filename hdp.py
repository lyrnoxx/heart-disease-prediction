import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
dataset = pd.read_csv('dataset.csv')

# Remove unecessary columns
#dataset.drop(['Unnamed: 0'], axis=1, inplace=True)

# Preprocess dataset
dataset = pd.get_dummies(dataset, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

scaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

# Split dataset into train and test sets
y = dataset['target']
X = dataset.drop(['target'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Define functions for model evaluation and prediction
def evaluate_model(model):
    return model.score(X_test, y_test)

def predict_disease(model, input_data):
    return model.predict(input_data)

# Create and evaluate models
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(evaluate_model(knn_classifier))

svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel=kernels[i])
    svc_classifier.fit(X_train, y_train)
    svc_scores.append(evaluate_model(svc_classifier))

dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features=i, random_state=0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(evaluate_model(dt_classifier))

rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators=i, random_state=0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(evaluate_model(rf_classifier))

# Create Streamlit app
st.set_page_config(page_title="Heart Disease Prediction", page_icon=":heartbeat:")

st.title("Heart Disease Prediction")

# Sidebar
st.sidebar.header("User Input Features")

def get_user_input():
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    age = st.sidebar.slider("Age", 0, 100, 50)
    cp = st.sidebar.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"))
    trestbps = st.sidebar.slider("Resting Blood Pressure", 0, 200, 100)
    chol = st.sidebar.slider("Cholesterol", 0, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/ddl", ("Yes", "No"))
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", ("Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"))
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 0, 300, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ("Yes", "No"))
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 10.0, 5.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", ("Upsloping", "Flat", "Downsloping"))
    ca = st.sidebar.selectbox("Number of Major Vessels Colored by Flourosopy", ("0", "1", "2", "3", "4"))
    thal = st.sidebar.selectbox("Thalassemia", ("Normal", "Fixed Defect", "Reversable Defect"))
    
    sex_male = 1 if sex == "Male" else 0
    sex_female = 1 if sex == "Female" else 0
    
    cp_typical_angina = 1 if cp == "Typical Angina" else 0
    cp_atypical_angina = 1 if cp == "Atypical Angina" else 0
    cp_non_anginal_pain = 1 if cp == "Non-Anginal Pain" else 0
    cp_asymptomatic = 1 if cp == "Asymptomatic" else 0
    
    fbs_yes = 1 if fbs == "Yes" else 0
    fbs_no = 1 if fbs == "No" else 0
    
    restecg_normal = 1 if restecg == "Normal" else 0
    restecg_st = 1 if restecg == "ST-T Wave Abnormality" else 0
    restecg_lv = 1 if restecg == "Left Ventricular" else 0
    exang_yes = 1 if exang == "Yes" else 0
    exang_no = 1 if exang == "No" else 0

    slope_upsloping = 1 if slope == "Upsloping" else 0
    slope_flat = 1 if slope == "Flat" else 0
    slope_downsloping = 1 if slope == "Downsloping" else 0

    ca_0 = 1 if ca == "0" else 0
    ca_1 = 1 if ca == "1" else 0
    ca_2 = 1 if ca == "2" else 0
    ca_3 = 1 if ca == "3" else 0

    thal_normal = 1 if thal == "Normal" else 0
    thal_fixed = 1 if thal == "Fixed Defect" else 0
    thal_reversable = 1 if thal == "Reversable Defect" else 0

    user_input = {
        "sex_male": sex_male,
        "sex_female": sex_female,
        "age": age,
        "cp_typical_angina": cp_typical_angina,
        "cp_atypical_angina": cp_atypical_angina,
        "cp_non_anginal_pain": cp_non_anginal_pain,
        "cp_asymptomatic": cp_asymptomatic,
        "trestbps": trestbps,
        "chol": chol,
        "fbs_yes": fbs_yes,
        "fbs_no": fbs_no,
        "restecg_normal": restecg_normal,
        "restecg_st": restecg_st,
        "restecg_lv": restecg_lv,
        "thalach": thalach,
        "exang_yes": exang_yes,
        "exang_no": exang_no,
        "oldpeak": oldpeak,
        "slope_upsloping": slope_upsloping,
        "slope_flat": slope_flat,
        "slope_downsloping": slope_downsloping,
        "ca_0": ca_0,
        "ca_1": ca_1,
        "ca_2": ca_2,
        "ca_3": ca_3,
        "ca_4": 0,
        "thal_normal": thal_normal,
        "thal_fixed": thal_fixed,
        "thal_reversable": thal_reversable,
        "thal_0": 0,
        "thal_1": 0,
        "thal_2": 0
    }

    input_df = pd.DataFrame(user_input, index=[0])
    input_df = pd.get_dummies(input_df, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])

    # Apply preprocessing to the user input
    input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])
    '''
    # Create model and predict output
    model_choice = st.sidebar.selectbox("Choose a machine learning model to predict heart disease", ("K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest"))
    if model_choice == "K-Nearest Neighbors":
        k = st.sidebar.slider("Number of Neighbors", 1, 20, 5)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        prediction = predict_disease(model, input_df)
    elif model_choice == "Support Vector Machine":
        kernel = st.sidebar.selectbox("Kernel", ("Linear", "Poly", "RBF", "Sigmoid"))
        model = SVC(kernel=kernel)
        model.fit(X_train, y_train)
        prediction = predict_disease(model, input_df)
    elif model_choice == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 1, len(X.columns), len(X.columns))
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(X_train, y_train)
        prediction = predict_disease(model, input_df)
    else:
        n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
        max_depth = st.sidebar.slider("Max Depth", 1, len(X.columns), len(X.columns))
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        prediction = predict_disease(model, input_df)
    if prediction == 0:
        st.write("The patient does not have heart disease.")
    else:
        st.write("The patient has heart disease.")
        
    user_input_df = pd.DataFrame({'sex': [sex], 'age': [age], 'cp_typical_angina': [cp_typical_angina],'cp_atypical_angina': [cp_atypical_angina], 'cp_non_anginal_pain': [cp_non_anginal_pain],'cp_asymptomatic': [cp_asymptomatic], 'trestbps': [trestbps], 'chol': [chol],'fbs_gt_120': [fbs_gt_120], 'restecg_normal': [restecg_normal],'restecg_st_t_wave_abnormality': [restecg_st_t_wave_abnormality],'restecg_left_ventricular_hypertrophy': [restecg_left_ventricular_hypertrophy],'thalach': [thalach], 'exang_yes': [exang_yes], 'oldpeak': [oldpeak],'slope_upsloping': [slope_upsloping], 'slope_flat': [slope_flat],'slope_downsloping': [slope_downsloping], 'ca_0': [ca_0], 'ca_1': [ca_1],'ca_2': [ca_2], 'ca_3': [ca_3], 'thal_fixed_defect': [thal_fixed_defect],'thal_normal': [thal_normal], 'thal_reversable_defect': [thal_reversable_defect]})
    model_choice = st.sidebar.selectbox("Choose a model", ("K-Nearest Neighbors", "Support Vector Machine","Decision Tree", "Random Forest"))
    if model_choice == "K-Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=11)
        model.fit(X_train, y_train)
        accuracy = evaluate_model(model)
        st.write("Accuracy:", accuracy)
        prediction = predict_disease(model, user_input_df)
        st.write("Prediction:", prediction[0])
    elif model_choice == "Support Vector Machine":
        kernel_choice = st.sidebar.selectbox("Choose a kernel", ("Linear", "Polynomial", "RBF", "Sigmoid"))
        kernel_dict = {"Linear": "linear", "Polynomial": "poly", "RBF": "rbf", "Sigmoid": "sigmoid"}
        kernel = kernel_dict[kernel_choice]
        model = SVC(kernel=kernel)
        model.fit(X_train, y_train)
        accuracy = evaluate_model(model)
        st.write("Accuracy:", accuracy)
        prediction = predict_disease(model, user_input_df)
        st.write("Prediction:", prediction[0])
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(max_features=13, random_state=0)
        model.fit(X_train, y_train)
        accuracy = evaluate_model(model)
        st.write("Accuracy:", accuracy)
        prediction = predict_disease(model, user_input_df)
        st.write("Prediction:", prediction[0])
    else:
        num_estimators = st.sidebar.selectbox("Choose the number of trees", [10, 100, 200, 500, 1000])
        model = RandomForestClassifier(n_estimators=num_estimators, random_state=0)
        model.fit(X_train, y_train)
        accuracy = evaluate_model(model)
        st.write("Accuracy:", accuracy)
        prediction = predict_disease(model, user_input_df)
        st.write("Prediction:", prediction[0])
        st.subheader("Model Evaluation")'''

model_names = ['K-Nearest Neighbors', 'Support Vector Machine', 'Decision Tree', 'Random Forest']
model_scores = [knn_scores, svc_scores, dt_scores, rf_scores]

for i in range(len(model_names)):
    st.write(model_names[i])
    st.line_chart(model_scores[i])

st.subheader("Make a Prediction")

user_input = get_user_input()

user_input_df = pd.DataFrame(data=[user_input], columns=X.columns)

user_input_df = pd.get_dummies(user_input_df, columns=['cp', 'restecg', 'slope', 'thal'])
user_input_df = user_input_df.reindex(columns=X.columns, fill_value=0)

user_input_df[columns_to_scale] = scaler.transform(user_input_df[columns_to_scale])

best_model_index = np.argmax([max(knn_scores), max(svc_scores), max(dt_scores), max(rf_scores)])
best_model_name = model_names[best_model_index]

if best_model_index == 0:
    best_model = KNeighborsClassifier(n_neighbors=knn_scores.index(max(knn_scores)) + 1)
elif best_model_index == 1:
    best_model = SVC(kernel=kernels[svc_scores.index(max(svc_scores))])
elif best_model_index == 2:
    best_model = DecisionTreeClassifier(max_features=dt_scores.index(max(dt_scores)) + 1, random_state=0)
else:
    best_model = RandomForestClassifier(n_estimators=estimators[rf_scores.index(max(rf_scores))], random_state=0)

best_model.fit(X_train, y_train)

prediction = predict_disease(best_model, user_input_df)

st.write("## Prediction")
if prediction[0] == 1:
    st.error("You have a high risk of having a heart disease!")
else:
    st.success("You have a low risk of having a heart disease.")

if st.button("Save Dataset"):
    dataset.to_csv("preprocessed_dataset.csv", index=False) # Save preprocessed dataset

if st.button("Save Model"):
    from joblib import dump
    dump(best_model, "heart_disease_prediction.joblib") # Save best model as a joblib file.
