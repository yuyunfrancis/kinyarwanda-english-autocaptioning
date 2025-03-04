
import os
import logging
import yaml

def setup_logging(level_name="INFO", log_file=None):
    """Set up logging configuration"""
    # Map string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    level = level_map.get(level_name.upper(), logging.INFO)
    
    # Configure logging
    logging_config = {
        "level": level,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging_config["filename"] = log_file
    
    logging.basicConfig(**logging_config)
    
    return logging.getLogger("kiny2eng")

def load_config(config_path):
    """Load configuration from YAML file"""
    # Default config
    default_config = {
        "transcription": {
            "model_name": "mbazaNLP/Whisper-Small-Kinyarwanda",
            "chunk_size": 15,  
            "overlap": 3,      
            "language": "sw", 
            "task": "transcribe"
        },
        "translation": {
            "model_name": "RogerB/marian-finetuned-multidataset-kin-to-en"
        },
        "captioning": {
            "format": "srt",
            "font_size": 24,
            "position": "bottom",
            "generate_video": False
        },
        "logging": {
            "level": "INFO",
            "file": "logs/kiny2eng.log"
        }
    }
    
    # Try to load from file
    try:
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Merge configs
        if user_config:
            # Update recursively
            def update_config(default, user):
                for key, value in user.items():
                    if key in default and isinstance(value, dict) and isinstance(default[key], dict):
                        update_config(default[key], value)
                    else:
                        default[key] = value
            
            update_config(default_config, user_config)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger = logging.getLogger("kiny2eng")
        logger.warning(f"Could not load config from {config_path}: {e}")
        logger.warning(f"Using default configuration")
    
    return default_config

def time_to_seconds(time_str):
    """Convert time string (HH:MM:SS.mmm) to seconds"""
    h, m, s = time_str.split(':')
    seconds = float(h) * 3600 + float(m) * 60
    
    # Handle milliseconds in different formats
    if ',' in s:
        s_parts = s.split(',')
        seconds += float(s_parts[0]) + float(s_parts[1]) / 1000
    elif '.' in s:
        seconds += float(s)
    else:
        seconds += float(s)
    
    return seconds

def parse_subtitle_file(subtitle_path):
    """Parse a subtitle file (SRT or VTT) into a list of subtitle entries"""
    with open(subtitle_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Determine format
    if content.strip().startswith('WEBVTT'):
        return parse_vtt(content)
    else:
        return parse_srt(content)

def parse_srt(content):
    """Parse SRT format"""
    import re
    
    # Split content into subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    subtitles = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # Parse timestamp line
        time_line = lines[1]
        times = time_line.split(' --> ')
        if len(times) != 2:
            continue
        
        start_time = time_to_seconds(times[0])
        end_time = time_to_seconds(times[1])
        
        # Join text lines
        text = '\n'.join(lines[2:])
        
        subtitles.append({
            'start': start_time,
            'end': end_time,
            'text': text
        })
    
    return subtitles

def parse_vtt(content):
    """Parse WebVTT format"""
    import re
    
    # Split content into subtitle blocks (skip header)
    content = re.sub(r'^WEBVTT\s*\n', '', content)
    blocks = re.split(r'\n\s*\n', content.strip())
    subtitles = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        
        # Find timestamp line
        time_line = None
        text_start_idx = 0
        
        for i, line in enumerate(lines):
            if ' --> ' in line:
                time_line = line
                text_start_idx = i + 1
                break
        
        if not time_line:
            continue
        
        # Parse timestamp
        times = time_line.split(' --> ')
        if len(times) != 2:
            continue
        
        start_time = time_to_seconds(times[0])
        end_time = time_to_seconds(times[1])
        
        # Join text lines
        text = '\n'.join(lines[text_start_idx:])
        
        subtitles.append({
            'start': start_time,
            'end': end_time,
            'text': text
        })
    
    return subtitles