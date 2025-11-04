from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import os
import numpy as np
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

MODEL_PATH = "./model"
MODEL_NAME = "roberta-base"
MAX_LEN = 256

print("Đang tải model từ:", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


class RoBERTaClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


device = torch.device("cpu")
model = RoBERTaClassifier()
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location=device))
model.to(device)
model.eval()
print(f"✅ Model loaded successfully on {device}!")


def predict_proba(texts):
    all_probs = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
        with torch.no_grad():
            logits = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device)
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)
    return np.array(all_probs)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        explain = data.get("explain", False)

        if not text:
            return jsonify({"error": "Vui lòng nhập nội dung bài đăng Reddit!"})

        probs = predict_proba([text])[0]
        pred = int(probs.argmax())
        confidence = float(probs[pred])

        highlight_words = []
        if explain:
            explainer = LimeTextExplainer(class_names=["Not Stress", "Stress"])
            exp = explainer.explain_instance(
                text,
                predict_proba,
                num_features=10,
                num_samples=500,
                labels=[0, 1]
            )
            lime_features = exp.as_list(label=pred)
            lime_features = sorted(lime_features, key=lambda x: abs(x[1]), reverse=True)
            highlight_words = [w for w, s in lime_features[:8] if len(w) > 1]

        result = {
            "text": text,
            "prediction": "CÓ DẤU HIỆU STRESS" if pred == 1 else "KHÔNG STRESS",
            "confidence": f"{confidence:.1%}",
            "color": "red" if pred == 1 else "green",
            "prob_stress": f"{probs[1]:.1%}",
            "prob_normal": f"{probs[0]:.1%}",
            "highlight_words": highlight_words,
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
