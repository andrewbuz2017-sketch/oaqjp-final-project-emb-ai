"""Detect emotions using the Watson NLP service."""
import json
import requests


def emotion_detector(text_to_analyze):
    """Return five emotion scores and the dominant emotion."""
    response = requests.post(
        "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict",
        headers={"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"},
        json={"raw_document": {"text": text_to_analyze}},
        timeout=30,
    )
    if response.status_code == 400:
        return dict.fromkeys(("anger", "disgust", "fear", "joy", "sadness", "dominant_emotion"))
    response.raise_for_status()
    scores = json.loads(response.text)["emotionPredictions"][0]["emotion"]
    result = {name: scores[name] for name in ("anger", "disgust", "fear", "joy", "sadness")}
    result["dominant_emotion"] = max(result, key=result.get)
    return result
