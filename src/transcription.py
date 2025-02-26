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

def transcribe_audio_file(audio_path, model_name="mbazaNLP/Whisper-Small-Kinyarwanda", chunck_size_seconds=30, overlap_seconds=5, language="sw", task="transcribe"):
    
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
    
    # Load and resample the audio file ensuring consistency and that the audio is in the correct format
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Calculate the chunk size and overlap in samples
    chunk_size = chunck_size_seconds * sr
    overlap = overlap_seconds * sr
    
    # Initialize transcription results
    full_transcription = []
    chunks = []
    
    # Iterate over chunks in the audio file
    for i in range(0, len(audio), chunk_size - overlap):
        # Calculate the timestamps for each chunk
        start_time = i / sr
        
        # Extract chunk
        end_indx = min(i + chunk_size, len(audio))
        chunk = audio[i:end_indx]
        end_time = end_indx / sr
        
        # Skip very short chunks
        if len(chunk) < sr * 2: # skips chunks shorter than 2 seconds
            continue
        
        # Process the chunk
        input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
        attention_mask = torch.ones(input_features.shape, dtype=torch.long)
        
        # Generate transcription for this chunk
        with torch.no_grad():
            predicted_ids = model.generate(input_features, attention_mask=attention_mask)
        
        chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Add chunk details to the results
        chunks.append({
            "start_time": start_time,
            "end_time": end_time,
            "text": chunk_transcription
        })
        
        # Add chunk transcription to the full transcription
        full_transcription.append(chunk_transcription)
    
    # Join all transcriptions
    complete_transcription = " ".join(full_transcription)
    
    return complete_transcription, chunks