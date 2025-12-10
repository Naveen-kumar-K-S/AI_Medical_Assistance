TRIAGE_RULES = {
    "cough": "Pneumonia Risk",
    "fever": "Infection",
    "chest pain": "Cardiac Issue",
    "breathless": "Respiratory",
    "shortness of breath": "Respiratory",
    "breathlessness": "Respiratory",
    "headache": "General Infection",
    "fatigue": "Chronic Condition",
    "joint pain": "Musculoskeletal",
    "stomach pain": "Abdominal Issue",
    "abdominal pain": "Abdominal Issue",
    "vomiting": "Gastrointestinal",
    "diarrhea": "Gastrointestinal",
    "nausea": "Gastrointestinal",
    "cold": "Mild Upper Respiratory",
    "sore throat": "Upper Respiratory",
    "throat pain": "Upper Respiratory",
}


def ai_triage(sentence: str) -> str:
    """
    Very simple rule-based AI triage.
    Looks for known keywords and maps them to triage categories.
    """
    text = sentence.lower()
    found = []

    # check multi-word keys first (like "chest pain")
    keys = sorted(TRIAGE_RULES.keys(), key=len, reverse=True)

    for key in keys:
        if key in text:
            found.append(f"{key} -> {TRIAGE_RULES[key]}")

    if not found:
        return "General Checkup - Needs ML Analysis"

    return "AI Triage: " + ", ".join(found)


if __name__ == "__main__":
    samples = [
        "patient has cough and fever",
        "chest pain and breathless patient",
        "headache and fatigue",
        "no symptoms mentioned",
    ]
    for s in samples:
        print(s, "=>", ai_triage(s))
