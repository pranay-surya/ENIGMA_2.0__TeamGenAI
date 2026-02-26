import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are an expert clinical AI assistant. Based on the provided EEG risk score "
    "and anomalous frequency bands, write a highly professional, 3-sentence clinical "
    "interpretation for a doctor. Do not diagnose; advise on risk and next steps."
)


def generate_clinical_explanation(risk_score: float, top_anomalies: list) -> str:
    """
    Call Groq LLM to generate a clinical interpretation.
    Falls back to a template if API key is not set.
    """
    if not GROQ_API_KEY:
        return _fallback_explanation(risk_score, top_anomalies)

    client = Groq(api_key=GROQ_API_KEY)

    user_message = (
        f"EEG Risk Score: {risk_score:.1f}%\n"
        f"Most Anomalous Frequency Bands: {', '.join(top_anomalies)}\n\n"
        "Please provide your 3-sentence clinical interpretation."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=300,
            temperature=0.4,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM Error: {str(e)}] " + _fallback_explanation(risk_score, top_anomalies)


def _fallback_explanation(risk_score: float, top_anomalies: list) -> str:
    """Template-based fallback when no API key is available."""
    level = "elevated" if risk_score > 70 else ("moderate" if risk_score > 30 else "low")
    bands_str = ", ".join(top_anomalies) if top_anomalies else "no specific bands"
    return (
        f"The EEG analysis reveals a {level} neurophysiological risk score of {risk_score:.1f}%, "
        f"with primary anomalies detected in the {bands_str} frequency domains. "
        f"These findings are consistent with patterns warranting further psychiatric evaluation, "
        f"including structured clinical interview and neurocognitive assessment. "
        f"It is recommended that the patient be referred to a specialist for a comprehensive "
        f"diagnostic workup; this tool should be used as a screening adjunct only."
    )