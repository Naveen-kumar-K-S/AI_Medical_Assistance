# ================================================
# COMBINED AI + ML + GenAI DEMO
# ================================================

from project_setup import ai_triage
from ml_model import (
    predict_disease_from_symptoms,
    get_precautions_for_disease,
    get_all_symptoms,
)
from genai_explainer import generate_explanation

# Canonical symptoms taken directly from the ML model
CANONICAL_SYMPTOMS = get_all_symptoms()

# Hand-made synonym mapping from free text → canonical symptom names
SYMPTOM_SYNONYMS = {
    "itching": ["itch", "itchy skin"],
    "skin_rash": ["rash", "rashes", "spots on skin"],
    "nodal_skin_eruptions": ["skin bumps", "small skin bumps"],
    "continuous_sneezing": ["sneezing", "sneeze", "sneezes a lot"],
    "runny_nose": ["running nose", "runny nose", "blocked nose", "stuffy nose"],
    "cough": ["coughing"],
    "chills": ["shivering", "feeling cold"],
    "stomach_pain": ["abdominal pain", "belly pain", "tummy ache", "stomach ache"],
    "chest_pain": ["chest tightness", "pain in chest", "pressure in chest"],
    "shortness_of_breath": [
        "breathless",
        "breathlessness",
        "short of breath",
        "difficulty breathing",
    ],
    "diarrhoea": ["diarrhea", "loose stool", "loose motions"],
    "joint_pain": ["knee pain", "elbow pain", "joint ache", "joint aches"],
    "headache": ["migraine", "head pain"],
    "fatigue": ["tired", "tiredness", "exhaustion", "low energy"],
    "nausea": ["feeling like vomiting", "sick to my stomach", "queasy"],
    # extend with more mappings over time
}

SYMPTOM_SYNONYMS = {
    "itching": ["itch", "itchy skin"],
    "skin_rash": ["rash", "rashes", "spots on skin"],
    "continuous_sneezing": ["sneezing", "sneeze", "sneezes a lot", "cold"],
    "runny_nose": ["running nose", "runny nose", "blocked nose", "stuffy nose", "cold"],
    "cough": ["coughing"],
    "chills": ["shivering", "feeling cold"],
}


def text_to_symptom_list(text: str):
    """
    Convert a free-text symptom sentence into a list of canonical symptom names.

    Uses:
    - direct matching on canonical names and their spaced versions
    - phrase-based synonyms defined in SYMPTOM_SYNONYMS
    """
    text = text.lower()
    found = set()

    # 1) Direct match on canonical names and their spaced versions
    for s in CANONICAL_SYMPTOMS:
        plain = s.replace("_", " ")
        if plain in text or s in text:
            found.add(s)

    # 2) Synonym-based phrase matching
    for canonical, phrases in SYMPTOM_SYNONYMS.items():
        for phrase in phrases:
            if phrase in text:
                found.add(canonical)

    return sorted(found)

def demo_ai_triage():
    print("=== AI MEDICAL ASSISTANCE : WEEK 1 DEMO ===")

    demo_queries = [
        "patient has cough and fever",
        "chest pain and breathless patient",
        "headache and fatigue",
        "no symptoms mentioned",
        "cough fever nausea diarrhea",
        "joint pain and chest pain",
    ]

    for q in demo_queries:
        result = ai_triage(q)
        print(f"Query : {q}")
        print(f"Result: {result}\n")

    import csv
    with open("data/ai_triage_outputs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "ai_triage"])
        for q in demo_queries:
            writer.writerow([q, ai_triage(q)])
    print("Saved AI triage outputs to data/ai_triage_outputs.csv")

def run_demo():
    demo_ai_triage()

    while True:
        query = input("Enter patient symptoms sentence (or 'q' to quit):\n> ")
        if query.lower().strip() in ("q", "quit", "exit"):
            break

        # 1) AI rule-based triage
        ai_result = ai_triage(query)

        # 2) Convert text → symptoms, then ML + precautions + GenAI
        symptoms = text_to_symptom_list(query)
        if symptoms:
            disease_name = predict_disease_from_symptoms(symptoms)
            precautions = get_precautions_for_disease(disease_name)
            explanation = generate_explanation(disease_name, query)
        else:
            disease_name = "Not enough structured symptoms"
            precautions = []
            explanation = (
                "GenAI explanation not available because no clear disease was predicted."
            )

        # 3) Combined output
        print("\n=== COMBINED OUTPUT ===")
        print("Input sentence :", query)
        print("AI Triage :", ai_result)
        print("Symptoms used :", symptoms)
        print("ML Disease :", disease_name)
        print("Precautions :", precautions)
        print("GenAI Explain :", explanation)
        print()

if __name__ == "__main__":
    run_demo()