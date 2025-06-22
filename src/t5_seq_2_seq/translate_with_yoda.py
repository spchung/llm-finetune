
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

def translate_to_yoda(text, model_path='./yoda-translator-final'):
    try:
        trained_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        trained_tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e
    

    input_text = f"translate to yoda voice: {text}"
        
    inputs = trained_tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=128, 
        truncation=True
    )
    
    with torch.no_grad():
        outputs = trained_model.generate(
            inputs["input_ids"],
            max_length=64,  # Shorter for stability
            num_beams=2,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=False
        )
    
    result = trained_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result if result else "Strong with errors, this model is. Try again, you must."

if __name__ == "__main__":
    while True:
        input_text = input("Enter text to translate to Yoda (or 'exit' to quit): ")
        try:
            result = translate_to_yoda(input_text)
            # print(f"Input Text: {input_text}")
            print(f"Yoda Translation: {result if result else 'Strong with errors, this model is. Try again, you must.'}")
        except Exception as e:
            print(f"❌ Error during translation: {e}")