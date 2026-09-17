import os
from PIL import Image
import torch

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)


# ============================================================
# DEEPGUARD MODEL EVALUATION
# ============================================================

MODEL_NAME = "delpot/steganograph-ia-detector"

DATASET_FOLDER = "test_dataset"

REAL_FOLDER = os.path.join(
    DATASET_FOLDER,
    "real"
)

FAKE_FOLDER = os.path.join(
    DATASET_FOLDER,
    "fake"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("DEEPGUARD MODEL EVALUATION")
print("=" * 60)

print("Loading model:")
print(MODEL_NAME)

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Model loaded successfully!")


# ============================================================
# DETECT IMAGE
# ============================================================

def detect_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    labels = model.config.id2label

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_label = labels[
        predicted_index
    ].lower()

    # --------------------------------------------------------
    # Identify probabilities
    # --------------------------------------------------------

    real_probability = 0.0
    fake_probability = 0.0

    for index, probability in enumerate(
        probabilities
    ):

        label = labels[index].lower()

        value = probability.item()

        if "real" in label:

            real_probability = value

        elif (
            "ai" in label
            or "generated" in label
            or "fake" in label
        ):

            fake_probability = value


    # --------------------------------------------------------
    # Convert model prediction to DeepGuard label
    # --------------------------------------------------------

    if (
        "ai" in predicted_label
        or "generated" in predicted_label
        or "fake" in predicted_label
    ):

        prediction = "FAKE"

    else:

        prediction = "REAL"


    return (
        prediction,
        real_probability,
        fake_probability
    )


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    )

    images = []

    if not os.path.exists(folder):

        return images

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            extensions
        ):

            images.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    return images


# ============================================================
# EVALUATION
# ============================================================

real_images = get_images(
    REAL_FOLDER
)

fake_images = get_images(
    FAKE_FOLDER
)


print("\n" + "=" * 60)

print(
    "REAL IMAGES:",
    len(real_images)
)

print(
    "FAKE IMAGES:",
    len(fake_images)
)

print("=" * 60)


# ============================================================
# CHECK DATASET
# ============================================================

if len(real_images) == 0:

    print(
        "\nNo images found in:"
    )

    print(
        REAL_FOLDER
    )

    print(
        "\nAdd real images first."
    )

    exit()


if len(fake_images) == 0:

    print(
        "\nNo images found in:"
    )

    print(
        FAKE_FOLDER
    )

    print(
        "\nAdd fake images first."
    )

    exit()


# ============================================================
# CONFUSION MATRIX COUNTERS
# ============================================================

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0


# ============================================================
# REAL IMAGE TEST
# ============================================================

print("\n")
print("=" * 60)
print("TESTING REAL IMAGES")
print("=" * 60)


for image_path in real_images:

    try:

        prediction, real_prob, fake_prob = (
            detect_image(image_path)
        )

        filename = os.path.basename(
            image_path
        )

        print(
            f"{filename:30} "
            f"→ {prediction:5} "
            f"(Real: {real_prob * 100:.2f}%, "
            f"Fake: {fake_prob * 100:.2f}%)"
        )


        # Actual = REAL
        # Prediction = REAL

        if prediction == "REAL":

            true_negative += 1

        else:

            false_positive += 1


    except Exception as e:

        print(
            "ERROR:",
            os.path.basename(image_path),
            e
        )


# ============================================================
# FAKE IMAGE TEST
# ============================================================

print("\n")
print("=" * 60)
print("TESTING FAKE IMAGES")
print("=" * 60)


for image_path in fake_images:

    try:

        prediction, real_prob, fake_prob = (
            detect_image(image_path)
        )

        filename = os.path.basename(
            image_path
        )

        print(
            f"{filename:30} "
            f"→ {prediction:5} "
            f"(Real: {real_prob * 100:.2f}%, "
            f"Fake: {fake_prob * 100:.2f}%)"
        )


        # Actual = FAKE
        # Prediction = FAKE

        if prediction == "FAKE":

            true_positive += 1

        else:

            false_negative += 1


    except Exception as e:

        print(
            "ERROR:",
            os.path.basename(image_path),
            e
        )


# ============================================================
# METRICS
# ============================================================

total = (
    true_positive
    + true_negative
    + false_positive
    + false_negative
)


# Accuracy

if total > 0:

    accuracy = (
        true_positive
        + true_negative
    ) / total

else:

    accuracy = 0


# Precision

if (
    true_positive
    + false_positive
) > 0:

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
    )

else:

    precision = 0


# Recall

if (
    true_positive
    + false_negative
) > 0:

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
        )
    )

else:

    recall = 0


# F1 Score

if (
    precision
    + recall
) > 0:

    f1_score = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )

else:

    f1_score = 0


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("DEEPGUARD EVALUATION RESULTS")
print("=" * 60)


print(
    f"Total images: {total}"
)

print(
    f"Correct predictions: "
    f"{true_positive + true_negative}"
)

print(
    f"Incorrect predictions: "
    f"{false_positive + false_negative}"
)


print("\nMETRICS")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1_score * 100:.2f}%"
)


print("\nCONFUSION MATRIX")

print(
    f"True Positive  (Fake → Fake): "
    f"{true_positive}"
)

print(
    f"True Negative  (Real → Real): "
    f"{true_negative}"
)

print(
    f"False Positive (Real → Fake): "
    f"{false_positive}"
)

print(
    f"False Negative (Fake → Real): "
    f"{false_negative}"
)


print("=" * 60)

print(
    "\nEvaluation completed."
)

print("=" * 60)
