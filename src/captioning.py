import os
import subprocess
from datetime import timedelta
from src.translation import translate_chunks

def format_timestamp(seconds, format_type="srt"):
    """Convert seconds to formatted timestamp"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    
    if format_type == "srt":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    elif format_type == "vtt":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def generate_captions(transcription, translation, chunks, format_type="srt"):
    """
    Generate caption file in specified format
    
    Args:
        transcription: Original Kinyarwanda transcription (full text)
        translation: English translation (full text)
        chunks: List of dictionaries with start_time, end_time, and text
        format_type: Caption format (srt or vtt)
        
    Returns:
        Formatted caption file content
    """
    if not chunks:
        return "No caption data available"
    
    # Generate captions based on format
    if format_type == "srt":
        return generate_srt_captions(transcription, translation, chunks)
    elif format_type == "vtt":
        return generate_vtt_captions(transcription, translation, chunks)
    else:
        raise ValueError(f"Unsupported caption format: {format_type}")

def generate_srt_captions(transcription, translation, chunks):
    """Generate SRT format captions"""
    # If we have both transcription and translation, we can add both
    if transcription and translation:
        # First get translation for each chunk
        translated_chunks = translate_chunks(chunks)
        
        srt_content = []
        for i, chunk in enumerate(translated_chunks, 1):
            # Skip invalid chunks
            if not chunk.get("original_text") or not chunk.get("translated_text"):
                continue
                
            start_time = format_timestamp(chunk["start_time"], "srt")
            end_time = format_timestamp(chunk["end_time"], "srt")
            
            # Add both original and translated text
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(f"{chunk['original_text']}")
            srt_content.append(f"{chunk['translated_text']}")
            srt_content.append("")  # Empty line between entries
    
    # If we only have translation, just use that
    elif translation:
        # Improved timing based on sentence structure rather than just character count
        import re
        sentences = re.split(r'(?<=[.!?])\s+', translation)
        
        total_chars = sum(len(s) for s in sentences)
        # Estimate audio duration (assuming average speaking rate)
        # Typical speaking rate is 150-160 words per minute
        # Assuming average word length of 5 characters, that's ~30 chars/second
        estimated_duration = total_chars / 30
        chars_per_second = total_chars / max(estimated_duration, len(sentences))
        
        srt_content = []
        current_time = 0
        chunk_count = 1
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Calculate duration based on sentence length
            sentence_duration = len(sentence) / chars_per_second
            
            # Add minimum duration for very short sentences
            sentence_duration = max(sentence_duration, 1.5)
            
            # Split long sentences
            words = sentence.split()
            if len(words) > 12:
                # Split into segments of ~10 words
                segments = []
                for j in range(0, len(words), 10):
                    segments.append(" ".join(words[j:j+10]))
                
                segment_duration = sentence_duration / len(segments)
                for segment in segments:
                    start_time = format_timestamp(current_time, "srt")
                    current_time += segment_duration
                    end_time = format_timestamp(current_time, "srt")
                    
                    srt_content.append(f"{chunk_count}")
                    srt_content.append(f"{start_time} --> {end_time}")
                    srt_content.append(segment)
                    srt_content.append("")  # Empty line
                    chunk_count += 1
            else:
                start_time = format_timestamp(current_time, "srt")
                current_time += sentence_duration
                end_time = format_timestamp(current_time, "srt")
                
                srt_content.append(f"{chunk_count}")
                srt_content.append(f"{start_time} --> {end_time}")
                srt_content.append(sentence)
                srt_content.append("")  # Empty line
                chunk_count += 1
    
    # If we only have transcription (original text)
    else:
        srt_content = []
        for i, chunk in enumerate(chunks, 1):
            if not chunk.get("text", "").strip():
                continue
                
            start_time = format_timestamp(chunk["start_time"], "srt")
            end_time = format_timestamp(chunk["end_time"], "srt")
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(f"{chunk['text']}")
            srt_content.append("")  # Empty line between entries
    
    return "\n".join(srt_content)

def generate_vtt_captions(transcription, translation, chunks):
    """Generate WebVTT format captions"""

    
    # Start with VTT header
    vtt_content = ["WEBVTT", ""]
    
    # Similar logic to SRT but with VTT formatting
    if transcription and translation:
        translated_chunks = translate_chunks(chunks)
        
        for i, chunk in enumerate(translated_chunks):
            start_time = format_timestamp(chunk["start_time"], "vtt")
            end_time = format_timestamp(chunk["end_time"], "vtt")
            
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(f"{chunk['original_text']}")
            vtt_content.append(f"{chunk['translated_text']}")
            vtt_content.append("")  # Empty line between entries
    
    elif translation:
        # Estimated timing similar to SRT function
        chars_per_second = 20
        words = translation.split()
        
        current_chunk = []
        current_duration = 0
        start_time = 0
        
        for word in words:
            word_duration = len(word) / chars_per_second
            
            if current_duration + word_duration > 5:  # Max 5 seconds per caption
                end_time = start_time + current_duration
                
                vtt_start = format_timestamp(start_time, "vtt")
                vtt_end = format_timestamp(end_time, "vtt")
                
                vtt_content.append(f"{vtt_start} --> {vtt_end}")
                vtt_content.append(" ".join(current_chunk))
                vtt_content.append("")  # Empty line
                
                start_time = end_time
                current_chunk = [word]
                current_duration = word_duration
            else:
                current_chunk.append(word)
                current_duration += word_duration
        
        # Add the last chunk
        if current_chunk:
            end_time = start_time + current_duration
            
            vtt_start = format_timestamp(start_time, "vtt")
            vtt_end = format_timestamp(end_time, "vtt")
            
            vtt_content.append(f"{vtt_start} --> {vtt_end}")
            vtt_content.append(" ".join(current_chunk))
            vtt_content.append("")
    
    else:
        for chunk in chunks:
            start_time = format_timestamp(chunk["start_time"], "vtt")
            end_time = format_timestamp(chunk["end_time"], "vtt")
            
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(f"{chunk['text']}")
            vtt_content.append("")  # Empty line between entries
    
    return "\n".join(vtt_content)

def create_captioned_video(video_path, caption_file, output_path, font_size=24, position="bottom"):
    """
    Create a video with captions using FFmpeg
    
    Args:
        video_path: Path to the input video
        caption_file: Path to the caption file (SRT or VTT)
        output_path: Path to save the output video
        font_size: Font size for captions
        position: Position of captions (bottom, top, middle)
    """
    # Check if FFmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not installed or not in PATH. Please install FFmpeg to create captioned videos.")
    
    # Determine position coordinates
    if position == "bottom":
        y_position = "(h-text_h-20)"
    elif position == "top":
        y_position = "20"
    else:  # middle
        y_position = "(h-text_h)/2"
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles={caption_file}:force_style='FontSize={font_size},Alignment=2,OutlineColour=&H40000000,BorderStyle=3'",
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "copy",
        output_path
    ]
    
    # Run FFmpeg
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")

def create_blank_video(duration, resolution, output_path):
    """
    Create a blank video using FFmpeg
    
    Args:
        duration: Duration of the video in seconds
        resolution: Resolution of the video (e.g., "1920x1080")
        output_path: Path to save the output video
    """
    # Build FFmpeg command to create a blank video
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s={resolution}:d={duration}",
        "-c:v", "libx264",
        output_path
    ]
    
    # Run FFmpeg
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode()}")