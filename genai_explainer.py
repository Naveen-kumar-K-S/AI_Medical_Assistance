# ================================================
# GenAI EXPLAINER: tiny RAG + GPT-2 on CPU
# ================================================

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, set_seed
from ml_model import prec_df  # precautions DataFrame already loaded there

# You can try "gpt2", "gpt2-medium", or another small causal LM
GEN_MODEL_NAME = "gpt2-medium"
GEN_MAX_NEW_TOKENS = 80
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_SEED = 42

# Globals
texts = []
disease_names = []
embed_model = None
embeddings = None
index = None
gen_pipeline = None


def initialize_genai():
    """
    Build a tiny RAG stack:
    - One text paragraph per disease (from precautions table)
    - SentenceTransformer embeddings + FAISS index
    - GPT‑2 generation pipeline for short explanations
    """
    global texts, disease_names, embed_model, embeddings, index, gen_pipeline

    if gen_pipeline is not None:
        # Already initialized
        return

    # 1) Build small knowledge base: one text per disease (from precautions)
    texts = []
    disease_names = []

    for _, row in prec_df.iterrows():
        disease = str(row["Disease"])
        precs = [
            str(v)
            for c, v in row.items()
            if "Precaution" in c or "precaution" in c
        ]
        paragraph = "Disease: " + disease + ". Precautions: " + ", ".join(precs) + "."
        disease_names.append(disease)
        texts.append(paragraph)

    # 2) Sentence embeddings + FAISS index (R part of RAG)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = embed_model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    embeddings = emb

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    index = faiss_index

    # 3) GPT‑2 text-generation pipeline (G part of RAG)
    set_seed(GEN_SEED)
    gen_pipeline = pipeline(
        "text-generation",
        model=GEN_MODEL_NAME,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
    )


def _extract_completion(prompt: str, raw: str) -> str:
    """
    GPT‑2 returns prompt + completion; strip the prompt prefix if present.
    """
    if raw.startswith(prompt):
        return raw[len(prompt) :].strip()
    return raw.strip()


def _postprocess_answer(answer: str, fallback: str) -> str:
    """
    Simple guard: if answer is empty, extremely short, or obviously
    just repeating instructions, return a safe fallback.
    """
    if not answer:
        return fallback

    lower = answer.lower()
    if "explain briefly" in lower or "keep the answer short" in lower:
        return fallback

    if len(answer.split()) < 5:
        return fallback

    return answer.strip()


def generate_explanation(disease_name, user_sentence):
    """
    RAG-style explanation using:
    - FAISS retrieval over disease+precaution paragraphs
    - GPT‑2 completion on a short instruction-style prompt
    """
    initialize_genai()

    global embed_model, index, texts, gen_pipeline

    # 1) Retrieve best matching context
    query_text = f"{disease_name}. {user_sentence}"
    q_vec = embed_model.encode(
        [query_text], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(q_vec, k=1)
    context = texts[int(ids[0][0])]

    # 2) Build prompt for GPT‑2
    prompt = (
        "Context: "
        + context
        + "\n"
        + "User symptoms: "
        + user_sentence
        + "\n"
        + "Explain briefly what this disease is and list the precautions in simple English. "
        + "Keep the answer short (1-3 sentences) and remind the user to consult a doctor.\n"
        + "Answer: "
    )

    raw_out = gen_pipeline(prompt)[0]["generated_text"]
    completion = _extract_completion(prompt, raw_out)

    fallback = (
        f"{disease_name} may be related to the symptoms you described. "
        f"Follow the listed precautions and consult a doctor for proper diagnosis and treatment."
    )

    return _postprocess_answer(completion, fallback)