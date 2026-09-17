import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "delpot/steganograph-ia-detector"

processor = None
model = None


def load_model():
    global processor, model

    if model is None:
        print("Loading DeepGuard AI detection model...")

        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

        model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True
        )

        model.to("cpu")
        model.eval()

        torch.set_num_threads(1)

        print("DeepGuard AI detection model loaded successfully.")


def detect_deepfake(image_path):
    try:
        load_model()

        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        labels = model.config.id2label

        real_probability = 0.0
        fake_probability = 0.0

        for i, probability in enumerate(probabilities):

            label = str(labels[i]).lower()
            value = float(probability.item())

            if (
                "ai" in label
                or "generated" in label
                or "fake" in label
                or "synthetic" in label
            ):
                fake_probability += value
            else:
                real_probability += value

        if fake_probability >= real_probability:
            result = "FAKE"
            confidence = fake_probability * 100
        else:
            result = "REAL"
            confidence = real_probability * 100

        if result == "FAKE":
            explanation = (
                "The AI model detected visual characteristics associated "
                "with AI-generated or manipulated imagery."
            )
        else:
            explanation = (
                "The AI model detected visual characteristics that are "
                "more consistent with a real photograph."
            )

        return {
            "result": result,
            "confidence": round(confidence, 2),
            "real_probability": round(real_probability * 100, 2),
            "fake_probability": round(fake_probability * 100, 2),
            "model": MODEL_NAME,
            "explanation": explanation
        }

    except Exception as e:

        print("Detection error:", e)

        return {
            "result": "ERROR",
            "confidence": 0,
            "real_probability": 0,
            "fake_probability": 0,
            "model": MODEL_NAME,
            "explanation": f"Detection failed: {str(e)}"
        }