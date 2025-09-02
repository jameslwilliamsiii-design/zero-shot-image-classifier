# Zero-Shot Image Classification with Hugging Face CLIP
# Works in Google Colab with file upload support

import io
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import (
    pipeline,
    AutoProcessor,
    AutoModelForZeroShotImageClassification
)
from google.colab import files

# Upload and load image
def upload_image_widget():
    uploaded = files.upload()
    if not uploaded:
        print("❌ No file uploaded.")
        return None

    for fn in uploaded.keys():
        try:
            image = Image.open(io.BytesIO(uploaded[fn])).convert("RGB")
            return image
        except Exception as e:
            print(f"Failed to load image: {e}")
            return None

# Inference using HF pipeline
def classify_with_pipeline(image, labels, checkpoint="openai/clip-vit-large-patch14"):
    detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
    return detector(image, candidate_labels=labels)

# Manual inference using processor + model
def classify_manually(image, labels, checkpoint="openai/clip-vit-large-patch14"):
    model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint)
    processor = AutoProcessor.from_pretrained(checkpoint)

    prompt_labels = [f"This is a photo of {label}." for label in labels]
    inputs = processor(images=image, text=prompt_labels, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=-1).numpy()

    return [
        {"score": float(score), "label": label}
        for score, label in sorted(zip(probs, labels), key=lambda x: -x[0])
    ]

# Visualize classification results
def plot_image_with_results(image, results, title="Zero-Shot Classification Results"):
    plt.imshow(image)
    plt.title(f"{title}\n" + "\n".join([f"{r['label']}: {r['score']:.2f}" for r in results]))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Main logic
if __name__ == "__main__":
    print("📸 Please upload an image for classification...")
    image = upload_image_widget()

    if image:
        labels = ["cat", "dog", "mountain", "car", "tree", "ocean"]

        print("\n🔍 Running classification with Hugging Face pipeline...")
        results_pipeline = classify_with_pipeline(image, labels)
        plot_image_with_results(image, results_pipeline, title="Pipeline Classification")

        print("\n🧠 Running manual model inference...")
        results_manual = classify_manually(image, labels)
        plot_image_with_results(image, results_manual, title="Manual Inference")
    else:
        print("❌ Image loading failed.")
