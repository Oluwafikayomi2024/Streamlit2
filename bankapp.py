import streamlit as st
import pandas as pd
import pickle

columns_to_keep = [
    "country", 
    "location_type", 
    "cellphone_access", 
    "household_size", 
    "age_of_respondent", 
    "gender_of_respondent", 
    "relationship_with_head", 
    "marital_status", 
    "education_level", 
    "job_type",
    "bank_account"  # This is your target variable
]



@st.cache_resource
def load_data():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
       # Load the dataset
    df = pd.read_csv("Financial_inclusion_dataset.csv",  usecols = columns_to_keep)

    return model, df

# Execute the load_data function to load the model and data
model, df = load_data()

country_encoder = {
    
    'Kenya': 0,
    'Rwanda': 1,
    'Tanzania': 2,
    'Uganda': 3
    }

location_type_encoder = {
    'Rural': 0,
    'Urban': 1
}

cellphone_access_encoder = {
    'No': 0,
    'Yes': 1
}

gender_of_respondent_encoder = {
    'Female': 0,    
    'Male': 1
}

relationship_with_head_encoder = {
    'Child': 0,
    'Head of Household': 1,
    'Other non-relatives': 2,
    'Other relative': 3,
    'Parent': 4,    
    'Spouse': 5
}

marital_status_encoder = {
    'Divorced/Seperated': 0,
    'Dont know': 1,
    'Married/Living together': 2,
    'Single/Never Married': 3,
    'Widowed': 4
}
education_level_encoder = {
    'No formal education': 0,
    'Other/Dont know/RTA': 1,
    'Primary education': 2,
    'Secondary education': 3,
    'Tertiary education': 4,
    'Vocational/Specialised training': 5
}

job_type_encoder = {
    'Dont Know/Refuse to answer': 0,
    'Farming and Fishing': 1,
    'Formally employed Government': 2,
    'Formally employed Private': 3,
    'Government Dependent': 4,
    'Informally employed': 5,
    'No Income': 6,
    'Other Income': 7,
    'Remittance Dependent': 8,
    'Self employed': 9
}


st.title("Bank Account Prediction")
st.subheader("This app predicts whether a customer has a bank account")


#create two columns
col1, col2 = st.columns(2)

with col1:
##Display input features
    country = country_encoder[st.selectbox("Country", df['country'].unique())]
    location_type = location_type_encoder[st.selectbox("Location Type", df["location_type"]. unique())]
    cellphone_access = cellphone_access_encoder[st.selectbox("Cellphone Access", df["cellphone_access"]. unique())]
    household_size = st.number_input("Household Size", min_value=df["household_size"].min(), max_value=df["household_size"].max(),)
    age_of_respondent = st.number_input("Age of Respondent", min_value=df["age_of_respondent"].min(), max_value=df["age_of_respondent"].max())




with col2: 

    gender_of_respondent = gender_of_respondent_encoder[st.selectbox("Gender of Respondent", df['gender_of_respondent'].unique())]
    relationship_with_head = relationship_with_head_encoder[st.selectbox("Relationship with Head", df['relationship_with_head'].unique())]
    marital_status = marital_status_encoder[st.selectbox("Marital Status", df['marital_status'].unique())]
    education_level = education_level_encoder[st.selectbox("Education Level", df['education_level'].unique())]
    job_type = job_type_encoder[st.selectbox("Job Type", df['job_type'].unique())]


# Create a DataFrame with the input features
input_data = pd.DataFrame({
    "country": country,
    "location_type": location_type,
    "cellphone_access": cellphone_access,
    "household_size": household_size,
    "age_of_respondent": age_of_respondent,
    "gender_of_respondent": gender_of_respondent,
    "relationship_with_head": relationship_with_head,
    "marital_status": marital_status,
    "education_level": education_level,
    "job_type": job_type,
}, index=[0])


predict_button = st.button("Predict")
if predict_button:
    # Make the prediction
    # Ensure the input data is in the correct format for the model
    prediction = model.predict(input_data)
    prediction = model.predict(input_data)
    
    # Display the prediction result
    if prediction == "Yes":
        st.success("The customer is likely to have a bank account.")
    else:
        st.error("The customer is likely not to have a bank account.")
    
   