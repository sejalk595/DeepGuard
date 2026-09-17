from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch


# DeepGuard AI detection model
MODEL_NAME = "delpot/steganograph-ia-detector"


print("Loading DeepGuard AI model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

model.eval()

print("DeepGuard AI model loaded successfully!")


def detect_deepfake(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Preprocess image
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Run AI model
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    # Get model labels
    labels = model.config.id2label

    results = {}

    for index, probability in enumerate(probabilities):

        label = labels[index].lower()

        results[label] = float(probability)

    # Initialize probabilities
    fake_probability = 0.0
    real_probability = 0.0

    # Identify Real / AI-generated classes
    for label, probability in results.items():

        if (
            "ai" in label
            or "generated" in label
            or "fake" in label
        ):
            fake_probability += probability

        elif "real" in label:

            real_probability += probability

    # Convert to percentage
    fake_percentage = fake_probability * 100
    real_percentage = real_probability * 100

    # Determine final result
    if fake_probability >= real_probability:

        result = "FAKE"

        confidence = fake_percentage

        explanation = (
            "The AI model detected patterns that are commonly "
            "associated with AI-generated or manipulated imagery."
        )

    else:

        result = "REAL"

        confidence = real_percentage

        explanation = (
            "The AI model found the image more consistent with "
            "a real photograph than with AI-generated imagery."
        )

    # Final analysis result
    analysis = {

        "result": result,

        "confidence": round(
            confidence,
            2
        ),

        "real_probability": round(
            real_percentage,
            2
        ),

        "fake_probability": round(
            fake_percentage,
            2
        ),

        "model": MODEL_NAME,

        "explanation": explanation
    }

    # Display result in terminal
    print("\n------------------------------")
    print("      DeepGuard Analysis")
    print("------------------------------")

    print(
        "Result:",
        result
    )

    print(
        "Confidence:",
        round(confidence, 2),
        "%"
    )

    print(
        "Real Probability:",
        round(real_percentage, 2),
        "%"
    )

    print(
        "AI Probability:",
        round(fake_percentage, 2),
        "%"
    )

    print(
        "Model:",
        MODEL_NAME
    )

    print("------------------------------\n")

    return analysis