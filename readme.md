## Setup
```commandline
pip install -r requirements.txt
git clone https://github.com/shashank14k/timething.git
cd timething
pip install -e .
```

## Assignment-1

1. Main functions are defined in transcript.py
2. The gradio app is in app.py

## Assignment-2 scripts
1. Scraping is done with srape.py
2. Audio analysis-> audio_analysis.py
3. Vocabulary analysis -> vocab_analysis.py
4. Alignment analysis -> align.py
5. All analysis scripts assume data is the format DATA_DIR -> BOOK_DIR -> CHAPTER_IDS -> audio.mp3 and transcription.txt
6. The analysis scripts save json files which are then analyzed for insights.