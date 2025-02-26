
import os
import argparse
from src.transcription import transcribe_audio_file
from src.translation import translate_text
from src.captioning import generate_captions, create_captioned_video
from src.utils import setup_logging, load_config

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Kinyarwanda to English Audio Translation Pipeline')
    parser.add_argument('--audio', type=str, help='Path to the audio file or directory containing audio files')
    parser.add_argument('--output', type=str, default='output', help='Output directory for results')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--video', type=str, default=None, help='Path to video file if captioning a video')
    parser.add_argument('--mode', type=str, choices=['transcribe', 'translate', 'caption', 'full'], 
                        default='full', help='Mode of operation')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['logging']['level'], config['logging']['file'])
    logger.info("Starting Kinyarwanda to English translation pipeline")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Process single file or directory
    if os.path.isfile(args.audio):
        process_file(args.audio, args.output, args.video, args.mode, config, logger)
    elif os.path.isdir(args.audio):
        for filename in os.listdir(args.audio):
            if filename.endswith(('.mp3', '.wav', '.flac', '.ogg')):
                file_path = os.path.join(args.audio, filename)
                process_file(file_path, args.output, args.video, args.mode, config, logger)
    else:
        logger.error(f"Input path {args.audio} is not valid")

def process_file(audio_path, output_dir, video_path, mode, config, logger):
    """Process a single audio file through the pipeline"""
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    output_base = os.path.join(output_dir, base_filename)
    
    # Step 1: Transcription
    if mode in ['transcribe', 'full', 'translate', 'caption']:
        logger.info(f"Transcribing {audio_path}")
        transcription, chunks = transcribe_audio_file(
            audio_path, 
            config['transcription']['model_name'],
            config['transcription']['chunk_size'],
            config['transcription']['overlap'],
            config['transcription']['language'],
            config['transcription']['task']
        )
        
        # Save transcription
        with open(f"{output_base}_transcription.txt", 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        logger.info(f"Transcription saved to {output_base}_transcription.txt")
    else:
        # Load transcription if already exists
        with open(f"{output_base}_transcription.txt", 'r', encoding='utf-8') as f:
            transcription = f.read()
        chunks = None  # We don't have chunks when loading from file
    
    # Step 2: Translation
    if mode in ['translate', 'full', 'caption']:
        logger.info(f"Translating transcription to English")
        translation = translate_text(
            transcription,
            config['translation']['model_name']
        )
        
        # Save translation
        with open(f"{output_base}_translation.txt", 'w', encoding='utf-8') as f:
            f.write(translation)
        
        logger.info(f"Translation saved to {output_base}_translation.txt")
    else:
        # Skip translation
        translation = None
    
    # Step 3: Captioning
    if mode in ['caption', 'full'] and (video_path is not None or config.get('captioning', {}).get('generate_video', False)):
        logger.info(f"Generating captions")
        
        captions = generate_captions(
            transcription, 
            translation, 
            chunks,
            config['captioning']['format']
        )
        
        # Save captions
        caption_file = f"{output_base}_captions.{config['captioning']['format']}"
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(captions)
            
        logger.info(f"Captions saved to {caption_file}")
        
        # Create captioned video if requested
        if video_path is not None:
            logger.info(f"Creating captioned video")
            output_video = f"{output_base}_captioned.mp4"
            create_captioned_video(
                video_path,
                caption_file,
                output_video,
                config['captioning']['font_size'],
                config['captioning']['position']
            )
            logger.info(f"Captioned video saved to {output_video}")
    
    logger.info(f"Processing complete for {audio_path}")
    return {
        "transcription": transcription,
        "translation": translation,
        "audio_path": audio_path,
        "output_base": output_base
    }

if __name__ == "__main__":
    main()