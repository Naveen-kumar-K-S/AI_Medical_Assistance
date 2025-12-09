import streamlit as st
from project_setup import ai_triage
from ai_ml_combined import text_to_symptom_list
from ml_model import predict_disease_from_symptoms, get_precautions_for_disease
from genai_explainer import generate_explanation

st.set_page_config(page_title="AI Medical Assistance", layout="centered")

st.title("🩺 AI Medical Assistance Prototype")
st.caption("Educational prototype – not a substitute for professional medical advice.")

user_input = st.text_input("Enter patient symptoms (sentence):", placeholder="e.g., I have chest pain and shortness of breath")

if st.button("Analyze") and user_input.strip():
    with st.spinner("Analyzing symptoms..."):
        triage = ai_triage(user_input)
        symptoms = text_to_symptom_list(user_input)

        if symptoms:
            disease = predict_disease_from_symptoms(symptoms)
            precautions = get_precautions_for_disease(disease)
            explanation = generate_explanation(disease, user_input)
        else:
            disease = "Not enough structured symptoms"
            precautions = []
            explanation = (
                "No clear disease could be predicted from this description. "
                "Please provide a few more specific symptoms and consult a doctor if you feel unwell."
            )

    st.markdown("### AI Triage")
    st.info(triage)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Structured Symptoms")
        if symptoms:
            st.write(symptoms)
        else:
            st.write("_No structured symptoms extracted._")

    with col2:
        st.markdown("### Predicted Disease")
        st.success(disease)

    st.markdown("### Precautions")
    if precautions:
        for p in precautions:
            st.markdown(f"- {p}")
    else:
        st.write("_No specific precautions available._")

    st.markdown("### GenAI Explanation")
    st.write(explanation)
