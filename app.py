import requests
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import (
    pipeline,
    AutoProcessor,
    AutoModelForZeroShotImageClassification
)

def load_image_from_url(url):
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        return image
    except Exception as e:
        print(f"Failed to load image: {e}")
        return None

def classify_with_pipeline(image, labels, checkpoint="openai/clip-vit-large-patch14"):
    detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
    return detector(image, candidate_labels=labels)

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

def plot_image_with_results(image, results, title="Zero-Shot Classification Results"):
    plt.imshow(image)
    plt.title(f"{title}\n" + "\n".join([f"{r['label']}: {r['score']:.2f}" for r in results]))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example 1: Using pipeline
    url1 = "https://unsplash.com/photos/g8oS8-82DxI/download?force=true&w=640"
    image1 = load_image_from_url(url1)
    labels1 = ["fox", "bear", "seagull", "owl"]
    if image1:
        results1 = classify_with_pipeline(image1, labels1)
        plot_image_with_results(image1, results1, title="Pipeline Classification")

    # Example 2: Manual inference
    url2 = "https://unsplash.com/photos/xBRQfR2bqNI/download?force=true&w=640"
    image2 = load_image_from_url(url2)
    labels2 = ["tree", "car", "bike", "cat"]
    if image2:
        results2 = classify_manually(image2, labels2)
        plot_image_with_results(image2, results2, title="Manual Classification")
