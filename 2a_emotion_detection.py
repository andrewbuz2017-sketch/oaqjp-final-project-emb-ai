"""Emotion detection using the Watson NLP service."""
import requests


def emotion_detector(text_to_analyse):
    """Return the raw Watson emotion prediction response."""
    response = requests.post(
        "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict",
        headers={"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"},
        json={"raw_document": {"text": text_to_analyse}},
        timeout=30,
    )
    return response.text
