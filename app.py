import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os
import pandas as pd
import pickle
import base64

from sklearn.metrics.pairwise import cosine_similarity


def calculate_medicine_score(excellent_percentage, average_percentage, poor_percentage):
    weight_excellent = 6.0
    weight_average = 3
    weight_poor = 1

    # Using this formula to calculate medicine score
    score = (excellent_percentage / 100 * weight_excellent) + \
            (average_percentage / 100 * weight_average) + \
            (poor_percentage / 100 * weight_poor)

    return round(score, 2)


def recommend_medicines_by_symptoms(symptoms, tfidf_vectorizer, tfidf_matrix_uses, clean_df):

    symptom_str = ' '.join(symptoms)

    symptom_vector = tfidf_vectorizer.transform([symptom_str])

    sim_scores = cosine_similarity(tfidf_matrix_uses, symptom_vector)

    sim_scores = sim_scores.flatten()
    similar_indices = sim_scores.argsort()[::-1]  # Sort indices based on similarity score


    recommended_medicines = clean_df.iloc[similar_indices][:6]  # Select top 5 similar medicines

    recommended_medicines['Medicine Score'] = calculate_medicine_score(recommended_medicines['Excellent Review %'],
                                                                       recommended_medicines['Average Review %'],
                                                                       recommended_medicines['Poor Review %'])

    recommended_medicines = recommended_medicines.sort_values(by='Medicine Score', ascending=False)

    return recommended_medicines


with open("recommend.pkl", 'rb') as file:
    loaded_components = pickle.load(file)

session_state = st.session_state

if 'history' not in session_state:
    session_state.history = []


# Display logo
logo_url = "https://image.similarpng.com/very-thumbnail/2021/06/Medical-pharmacy-logo-design-template-on-transparent-background-PNG.png"  # Replace this with the URL of your logo image
st.markdown("<h1 style='text-align: center;'><img src='" + logo_url + "' style='width: 200px;'></h1>", unsafe_allow_html=True)

st.title("Medicine Recommendation System")


st.sidebar.title("Last Symptom and Recommendations")

if session_state.history:
    last_searched_symptom, recommended_medicines = session_state.history[-1]
    st.sidebar.subheader("Last Searched Symptom:")
    st.sidebar.write(last_searched_symptom)
    st.sidebar.subheader("Recommended Medicines:")
    for index, row in recommended_medicines.iterrows():
        st.sidebar.write(row['Medicine Name'])


num_symptoms = 1
selected_symptoms = []
for i in range(num_symptoms):
    selected_symptoms.append(st.multiselect(
        f'Select symptoms',
        ('Sneezing', 'Fever', 'Headache', 'Fatigue', 'Nausea', 'Vomiting', 'Diarrhea', 'Cough', 'Chest pain',
         'Joint pain', 'Abdominal pain', 'Appetite', 'Swelling', 'Itching', 'Sore throat', 'Eye pain', 'Infection',
         'Runny nose', 'Skin', 'Dandruff')
    ))

if st.button("Recommend"):
    last_symptom_recommendation = selected_symptoms
    selected_symptoms_flat = [symptom for sublist in selected_symptoms for symptom in sublist]


    if selected_symptoms_flat:

        recommended_medicines = recommend_medicines_by_symptoms(selected_symptoms_flat,
                                                                loaded_components['tfidf_vectorizer_uses'],
                                                                loaded_components['tfidf_matrix_uses'],
                                                                loaded_components['clean_df'])

        if not recommended_medicines.empty:

            session_state.history.append((selected_symptoms_flat, recommended_medicines))
            for i in range(0, len(recommended_medicines), 2):
                col1, col_space, col2 = st.columns([3, 0.2, 3])

                with col1:
                    if i < len(recommended_medicines):
                        medicine1 = recommended_medicines.iloc[i]
                        st.write(f"**{medicine1['Medicine Name']}**")
                        st.write(f"Manufacturer: {medicine1['Manufacturer']}")
                        st.write(f"Medicine Score: {medicine1['Medicine Score']}")
                        img_url1 = medicine1['Image URL']
                        if img_url1:
                            try:
                                response = requests.get(img_url1)
                                if response.status_code == 200:
                                    image = Image.open(BytesIO(response.content))
                                    image = image.resize((250, 250))
                                    st.image(image, caption=medicine1['Medicine Name'], width=200)
                                else:
                                    st.write("Image not available")
                            except Exception as e:
                                st.write(f"Error loading image: {e}")

                st.write(" ")
                st.write("\n")
                st.write("\n")
                st.write(" ")
                with col2:
                    if i + 1 < len(recommended_medicines):
                        medicine2 = recommended_medicines.iloc[i + 1]
                        st.write(f"**{medicine2['Medicine Name']}**")
                        st.write(f"Manufacturer: {medicine2['Manufacturer']}")
                        st.write(f"Medicine Score: {medicine2['Medicine Score']}")
                        img_url2 = medicine2['Image URL']
                        if img_url2:
                            try:
                                response = requests.get(img_url2)
                                if response.status_code == 200:
                                    image = Image.open(BytesIO(response.content))
                                    image = image.resize((250, 250))
                                    st.image(image, caption=medicine2['Medicine Name'], width=200)
                                else:
                                    st.write("Image not available")
                            except Exception as e:
                                st.write(f"Error loading image: {e}")

        else:
            st.write("No medicines found for the selected symptoms.")
    else:
        st.write("Please select at least one symptom.")


# Save Recommendation button
for i, (symptom, medicines) in enumerate(session_state.history):
    if st.sidebar.button(f"Save Recommendation {i + 1}"):
        file_name = f"recommendation_{i + 1}.txt"
        file_content = f"Symptom: {symptom}\n\nRecommended Medicines:\n"
        file_content += "\n".join(medicines['Medicine Name'])
        b64 = base64.b64encode(file_content.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/txt;base64,{b64}" download="{file_name}">Download {file_name}</a>'
        st.markdown(href, unsafe_allow_html=True)
