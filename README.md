# Kinyarwanda to English Translation Pipeline

This project provides a pipeline for transcribing, translating, and captioning Kinyarwanda audio files into English. The pipeline can process single audio files or directories containing multiple audio files.

## Prerequisites

- Python 3.8 or higher
- FFmpeg
- Virtual environment (optional but recommended)

## Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yuyunfrancis/kinyarwanda-english-autocaptioning.git
   cd kinyarwanda-english-autocaptioning
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg**:

   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

5. **Create a .env file** in the root directory and add your Hugging Face token:
   ```
   HUGGING_FACE_TOKEN=your_hugging_face_token_here
   ```

## Usage

To run the pipeline, use the following command:

```bash
python main.py --audio [AUDIO_PATH] --output [OUTPUT_DIR] --config [CONFIG_PATH] --mode [MODE]
```

### Arguments

- `--audio`: Path to the audio file or directory containing audio files.
- `--output`: Output directory for results (default: output).
- `--config`: Path to the configuration file (default: config.yaml).
- `--video`: Path to the video file if captioning a video (optional).
- `--mode`: Mode of operation (transcribe, translate, caption, full).

### Example

```bash
python main.py --audio ./data/kinyarwanda_1.mp3 --output ./output --config ./config.yaml --mode full
```

## Configuration

The configuration file (`config.yaml`) should contain the necessary settings for the transcription, translation, and captioning processes. Here is an example configuration:

```yaml
transcription:
  model_name: "mbazaNLP/Whisper-Small-Kinyarwanda"
  chunk_size: 30
  overlap: 5
  language: "sw"
  task: "transcribe"

translation:
  model_name: "Helsinki-NLP/opus-mt-sw-en"

captioning:
  format: "srt"
  font_size: 24
  position: "bottom"

logging:
  level: "INFO"
  file: "./logs/kiny2eng.log"
```

## Logging

Logs are saved to the `kiny2eng.log` file. You can check this file for detailed information about the pipeline's execution.
