# src/translation.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

def load_translation_model(model_name):
    """Load the translation model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, model_name="RogerB/marian-finetuned-multidataset-kin-to-en", batch_size=512):
    """
    Translate Kinyarwanda text to English
    
    Args:
        text: Text to translate
        model_name: Name of the translation model to use
        batch_size: Maximum number of characters to process at once
        
    Returns:
        Translated English text
    """
    # Load model and tokenizer
    tokenizer, model = load_translation_model(model_name)
    
    # If text is short enough, translate it all at once
    if len(text) < batch_size:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        
        # Generate translation
        with torch.no_grad():
            output_ids = model.generate(**inputs)
        
        # Decode the generated tokens
        translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return translation
    
    # For longer text, split into sentences and translate in batches
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Group sentences into batches
    batches = []
    current_batch = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > batch_size and current_batch:
            batches.append(" ".join(current_batch))
            current_batch = [sentence]
            current_length = len(sentence)
        else:
            current_batch.append(sentence)
            current_length += len(sentence)
    
    if current_batch:
        batches.append(" ".join(current_batch))
    
    # Translate each batch
    translations = []
    for batch in batches:
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        
        # Generate translation
        with torch.no_grad():
            output_ids = model.generate(**inputs)
        
        # Decode the generated tokens
        batch_translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        translations.append(batch_translation)
    
    # Join all translations
    complete_translation = " ".join(translations)
    
    return complete_translation

def translate_chunks(chunks, model_name="RogerB/marian-finetuned-multidataset-kin-to-en"):
    """
    Translate each chunk separately while preserving timing information
    
    Args:
        chunks: List of dictionaries with start_time, end_time, and text
        model_name: Name of the translation model to use
        
    Returns:
        List of dictionaries with start_time, end_time, original_text, and translated_text
    """
    # Load model and tokenizer
    tokenizer, model = load_translation_model(model_name)
    
    translated_chunks = []
    
    for chunk in chunks:
        # Skip empty chunks
        if not chunk["text"].strip():
            continue
            
        inputs = tokenizer(chunk["text"], return_tensors="pt", padding=True)
        
        # Generate translation
        with torch.no_grad():
            output_ids = model.generate(**inputs)
        
        # Decode the generated tokens
        translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        translated_chunks.append({
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "original_text": chunk["text"],
            "translated_text": translation
        })
    
    return translated_chunks