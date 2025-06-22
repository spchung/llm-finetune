from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, os
import torch.nn.functional as F

# Load your fine-tuned model from checkpoint
model_path = "./test_trainer/checkpoint-375"  # Your checkpoint folder
base_dir = "/Users/stephen/Nottingham/fine_tuning/bert_classification"
checkpoint_dir = os.path.join(base_dir, model_path)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")  # Same tokenizer used during training

# Set model to evaluation mode
model.eval()

# Example text for prediction
text = "The food was great but the service was slow."
# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get the predicted class
outputs = model(**inputs)

logits = outputs.logits
probabilities = F.softmax(logits, dim=-1)

# Get predicted class and confidence
predicted_class = torch.argmax(probabilities, dim=-1).item()
confidence = probabilities[0][predicted_class].item()

# Convert to star rating
print(f"Predicted class: {predicted_class + 1} stars, Confidence: {confidence:.4f}")
