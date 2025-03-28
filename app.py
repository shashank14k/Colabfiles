from transcript import transcribe_audio_segments, get_audio_segments, save_segments_to_json
import gradio as gr
import json
import os
import yt_dlp


def get_transcriptions(aud_file, save_path, language, vad_model, min_silence_length, perc_vocal_windows,
                       transcription_model, max_segment_length, quantize):
    print("INPUTPATH", aud_file)
    segments = get_audio_segments(aud_file, vad_model, perc_vocal_windows, min_silence_length=min_silence_length,
                                  max_segment_length=max_segment_length)
    if "large" in transcription_model:
        quantize = True
    segments = transcribe_audio_segments(aud_file, segments, transcription_model, quantize=quantize, language=language,
                                         max_length=max_segment_length)
    save_segments_to_json(segments, save_json_path=save_path)


def process_input(input_path, language, vad_model, min_silence_length, perc_vocal_windows, transcription_model,
                  max_segment_length, quantize):
    output_path = "temp_transcription.json"
    get_transcriptions(input_path, output_path, language, vad_model, min_silence_length, perc_vocal_windows,
                       transcription_model, max_segment_length, quantize)
    with open(output_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    os.remove(output_path)  # Clean up temporary file
    return json.dumps(result, indent=4)


def handle_input(input_str, language, vad_model, min_silence_length, perc_vocal_windows, transcription_model,
                 max_segment_length, quantize):
    if input_str.startswith(('http://', 'https://')):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': 'downloaded_audio.%(ext)s',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(input_str, download=True)
            audio_path = ydl.prepare_filename(info).replace('.webm', '.wav')

        audio_path = os.path.splitext(audio_path)[0] + ".wav"
        result = process_input(audio_path, language, vad_model, min_silence_length, perc_vocal_windows,
                               transcription_model, max_segment_length, quantize)
        os.remove(audio_path)
        return result
    else:
        return process_input(input_str, language, vad_model, min_silence_length, perc_vocal_windows,
                             transcription_model, max_segment_length, quantize)


with gr.Blocks() as app:
    gr.Markdown("## YouTube Audio Transcriber with Custom Parameters")
    with gr.Row():
        input_source = gr.Textbox(label="Enter YouTube URL or Audio File Path",
                                  placeholder="https://youtube.com/... or /path/to/audio.wav")

    language = gr.Textbox(label="Language Code" ,placeholder="e.g., en, hi, fr")
    vad_model = gr.Dropdown(["silero", "simple-buffer"], label="VAD Model", value="silero")
    min_silence_length = gr.Slider(0.1, 2.0, value=0.5, step=0.1, label="Min Silence Length (seconds)")
    perc_vocal_windows = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="Percentage of Vocal Windows")
    transcription_model = gr.Textbox(label="Transcription Model", value="openai/whisper-large-v3")
    max_segment_length = gr.Slider(5, 30, value=15, step=1, label="Max Segment Length (seconds)")
    quantize = gr.Checkbox(label="Use Quantized Model", value=True)

    output_json = gr.Textbox(label="Transcription Results", interactive=False, lines=20)
    submit_btn = gr.Button("Process")

    submit_btn.click(
        fn=handle_input,
        inputs=[input_source, language, vad_model, min_silence_length, perc_vocal_windows,
                transcription_model, max_segment_length, quantize],
        outputs=output_json
    )

app.launch()