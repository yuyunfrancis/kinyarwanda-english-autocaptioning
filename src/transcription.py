import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment variable
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")

def load_transcription_model(model_name):
    """ Load the whisper transcription model and processor"""
    processor = WhisperProcessor.from_pretrained(model_name, token=hugging_face_token)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, token=hugging_face_token)
    return processor, model
def transcribe_audio_file(audio_path, model_name="mbazaNLP/Whisper-Small-Kinyarwanda", 
                        chunk_size_seconds=15, overlap_seconds=3, language="sw", task="transcribe"):
    
    """
    Transcribe an audio file in chunks
    
    Args:
        audio_path: Path to the audio file
        model_name: Name of the Whisper model to use
        chunk_size_seconds: Size of audio chunks in seconds
        overlap_seconds: Overlap between chunks in seconds
        language: Language code for transcription
        task: Task type (transcribe or translate)
        
    Returns:
        full_transcription: Complete transcription text
        chunks: List of dictionaries with start_time, end_time, and text
    """
    
    # Load model and processor
    processor, model = load_transcription_model(model_name)
    
    # Set language and task
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    
    # Load and resample the audio file ensuring consistency
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Calculate the chunk size and overlap in samples
    chunk_size = chunk_size_seconds * sr
    overlap = overlap_seconds * sr
    
    # Initialize transcription results
    full_transcription = []
    chunks = []
    
    # Process audio in smaller chunks with greater overlap
    for i in range(0, len(audio), chunk_size - overlap):
        # Calculate the timestamps for each chunk
        start_time = max(0, i / sr)
        
        # Extract chunk with proper boundary handling
        end_idx = min(i + chunk_size, len(audio))
        chunk = audio[i:end_idx]
        end_time = end_idx / sr
        
        # Skip very short chunks
        if len(chunk) < sr: # Skip chunks shorter than 1 second
            continue
        
        # Process the chunk
        input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
        
        # Use a proper attention mask
        attention_mask = torch.ones(input_features.shape, dtype=torch.long)
        
        # Generate transcription for this chunk
        with torch.no_grad():
            # Use beam search for better transcription quality
            predicted_ids = model.generate(
                input_features, 
                attention_mask=attention_mask,
                num_beams=5,         # Use beam search with 5 beams
                max_length=256,      # Limit output length
                min_length=1,        # Allow short outputs
                no_repeat_ngram_size=3  # Avoid repeating trigrams
            )
        
        chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Add chunk details to the results
        chunks.append({
            "start_time": start_time,
            "end_time": end_time,
            "text": chunk_transcription.strip()
        })
        
        # Add chunk transcription to the full transcription
        full_transcription.append(chunk_transcription.strip())
    
    # Post-process chunks to improve continuity
    processed_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Skip empty chunks
        if not chunk["text"]:
            continue
            
        # If not the first chunk, check for overlap with previous text
        if i > 0 and processed_chunks:
            prev_chunk = processed_chunks[-1]
            prev_words = prev_chunk["text"].split()
            curr_words = chunk["text"].split()
            
            # Find word overlap to smooth transitions
            overlap_found = False
            
            # Check for overlapping phrases (at least 2 words)
            for j in range(min(10, len(prev_words))):
                overlap_size = min(len(prev_words) - j, len(curr_words))
                if overlap_size >= 2:
                    prev_phrase = " ".join(prev_words[-overlap_size:])
                    curr_phrase = " ".join(curr_words[:overlap_size])
                    
                    # Check for approximate match (handle slight variations)
                    if prev_phrase.lower() == curr_phrase.lower():
                        # Remove the overlapping portion from current chunk
                        chunk["text"] = " ".join(curr_words[overlap_size:])
                        overlap_found = True
                        break
        
        # Add to processed chunks
        if chunk["text"]:
            processed_chunks.append(chunk)
    
    # Rebuild full transcription from processed chunks
    full_text = " ".join([chunk["text"] for chunk in processed_chunks])
    
    return full_text, processed_chunks