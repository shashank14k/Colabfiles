from soundfile import SoundFile
import librosa
from typing import List, Optional
import os
import numpy as np
from dataclasses import dataclass
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch
from silero_vad import get_speech_timestamps
import json

WHISPER_MODELS = {}


@dataclass
class Segment:
    start_time: float
    end_time: float

    vocal: Optional[bool] = None
    text: Optional[str] = None
    language: Optional[str] = None

    @staticmethod
    def from_dict(dic):
        return Segment(**dic)

    def extend(self, step):
        self.end_time = self.end_time + step

    @property
    def duration(self):
        return self.end_time - self.start_time

    def __dict__(self):
        return {"start_time": self.start_time, "end_time": self.end_time, "vocal": bool(self.vocal), "text": self.text}


class WavReader:
    def __init__(self, audio_file: str):
        self.stream = SoundFile(audio_file)
        self.duration = len(self.stream) / self.stream.samplerate

    def read(self, start, end):
        start = int(start * self.stream.samplerate)
        self.stream.seek(start)
        end = int(end * self.stream.samplerate)
        samples = self.stream.read((end - start + 1))  ## shape is (n_samples, n_channels)
        if samples.shape[-1] == 2:
            samples = samples.mean(axis=-1)
        return samples


class Generator:
    def __init__(self, filepath: str, perc_vocal_windows: float, min_silence_length: float = 0.5,
                 processing_sr: int = 16000, buffer_limit_duration: float = 300, max_segment_length: int = 15):
        assert os.path.exists(filepath)
        self.file = filepath
        self.stream = WavReader(filepath)
        self._start = 0
        self._end = self.stream.duration
        self.buffer_limit_duration = buffer_limit_duration
        self.perc_vocal_windows = perc_vocal_windows
        self.min_silence_length = min_silence_length
        self.processing_sr = processing_sr
        self.max_segment_length = max_segment_length


class BufferGenerator(Generator):
    def __init__(self, filepath: str, perc_vocal_windows: float = 0.3, min_silence_length: float = 0.5,
                 processing_sr: int = 16000, buffer_limit_duration: float = 300, **kwargs):
        super().__init__(filepath, perc_vocal_windows, min_silence_length, processing_sr, buffer_limit_duration)
        self.win_length = kwargs.get("win_length", 320)
        self.hop_length = kwargs.get("hop_length", 160)
        self.perc_vocal_windows = perc_vocal_windows
        self.min_silence_length = min_silence_length
        self.min_buffer_len = int(self.min_silence_length * self.processing_sr / self.hop_length)
        # These params set the threshold. For an audio sample with mean dbfs of -36, frames below -40 dbfs will be considered non vocal
        self.base_dbfs = kwargs.get("base_dbfs", -36)
        self.base_sil = kwargs.get("base_sil", -14)
        assert self.base_sil < 0 and self.base_dbfs < 0
        self.window_pad = kwargs.get("window_pad", 3)
        self.segments: List[Segment] = []

    def chunk_global_timestamp(self, global_start, local_idx) -> float:
        return global_start + local_idx / self.processing_sr

    def adapt_silence_threshold(self, dbfs: float) -> float:
        perc_diff = (self.base_dbfs - dbfs) / self.base_dbfs
        sil_thresh = self.base_sil + (self.base_sil * perc_diff)
        sil_thresh = round(sil_thresh, 0)
        self.max_dbf = sil_thresh
        print("Audio Buffer dbfs={}, setting silence threshold at {}".format(dbfs, self.max_dbf))
        return self.max_dbf

    def compute_dbfs(self, samples: np.ndarray):
        if samples.ndim == 1:
            samples = samples.reshape(1, len(samples))
        padded_signal = np.pad(samples, ((0, 0), (self.win_length // 2, self.win_length // 2)), mode='constant') + 1e-6
        num_channels, signal_length = padded_signal.shape
        num_windows = (signal_length - self.win_length) // self.hop_length + 1
        sliding_windows = np.lib.stride_tricks.as_strided(
            padded_signal,
            shape=(num_channels, num_windows, self.win_length),
            strides=(
                padded_signal.strides[0],
                self.hop_length * padded_signal.strides[1],
                padded_signal.strides[1],
            )
        )
        rms = np.sqrt(np.mean(sliding_windows ** 2, axis=-1)).mean(axis=0)
        dbfs = 20 * np.log10(np.maximum(-100, rms))
        return dbfs

    def _split_segment(self, samples: np.ndarray, start: float, end: float, include_last: bool = True) -> float:
        rms_arr = self.compute_dbfs(samples)
        dbfs = np.quantile(rms_arr, 0.8)
        # Adapt dbfs
        max_dbfs = self.adapt_silence_threshold(dbfs)
        relative_to_max_db = rms_arr - dbfs
        # Get vocal window
        vocals = (relative_to_max_db > max_dbfs).astype(int)
        ptr1 = 0
        window = vocals[ptr1: ptr1 + self.min_buffer_len]
        val = sum(window)
        vocal = (val > int(self.perc_vocal_windows * self.min_buffer_len))
        ptr1 += self.min_buffer_len
        segment = Segment(self.chunk_global_timestamp(start, 0),
                          self.chunk_global_timestamp(start, self.hop_length * self.min_buffer_len),
                          vocal=vocal)

        while ptr1 < len(vocals):
            val = val + vocals[ptr1] - vocals[ptr1 - self.min_buffer_len]
            vocal = (val > int(self.perc_vocal_windows * self.min_buffer_len))
            if vocal != segment.vocal and segment.duration > self.min_silence_length:
                change = True
            else:
                change = False
            if not change:
                segment.extend(self.hop_length / self.processing_sr)
            else:
                if segment.vocal:
                    # trim trailing silent windows from vocal segment
                    trailing_sil_windows = np.nonzero(vocals[ptr1 - self.min_buffer_len: ptr1 + 1])[0]
                    if len(trailing_sil_windows):
                        sil_start_window = min(ptr1, ptr1 - self.min_buffer_len + trailing_sil_windows[-1] +
                                               self.window_pad)
                        segment.end_time = self.chunk_global_timestamp(start, sil_start_window * self.hop_length)
                else:
                    # trim trailing vocal windows from silent segment
                    trailing_voc_windows = np.nonzero(vocals[ptr1 - self.min_buffer_len: ptr1 + 1])[0]
                    if len(trailing_voc_windows):
                        voc_start_window = max(ptr1 - self.min_buffer_len,
                                               ptr1 - self.min_buffer_len + trailing_voc_windows[
                                                   0] - self.window_pad)
                        segment.end_time = self.chunk_global_timestamp(start, voc_start_window * self.hop_length)
                self.segments.append(segment)
                new_start = segment.end_time
                new_end = segment.end_time + (self.hop_length / self.processing_sr)
                segment = Segment(new_start, new_end, vocal=vocal)
            ptr1 += 1
        if include_last:
            segment.end_tim = end_dur = end
            self.segments.append(segment)
        else:
            end_dur = segment.start_time
        return end_dur

    def split_audio_to_segments(self, post_process: bool = True):
        while self._start < self._end:
            _end = min(self._start + self.buffer_limit_duration, self._end)
            samples = self.stream.read(self._start, _end)
            samples = librosa.resample(samples, orig_sr=self.stream.stream.samplerate, target_sr=self.processing_sr)
            if _end == self._end:
                _end = self._split_segment(samples, self._start, _end, True)
            else:
                _end = self._split_segment(samples, self._start, _end, False)
            self._start = _end

        for segment in self.segments:
            if segment.vocal:
                # subtract 0.1 as some pad
                if segment.duration < (self.min_silence_length - 0.1):
                    print(
                        "Vocal segment duration {} less than min vocal length {}, classifying it as non-vocal.".format(
                            segment.duration, self.min_silence_length))
                    segment.vocal = False
        if post_process:
            self.post_process_segments()
        return self.segments

    def post_process_segments(self):
        if len(self.segments) < 3:
            return
        processed_segments = []
        p1 = 0 if self.segments[0].vocal else 1
        p2 = p1 + 2
        if p1 == 1:
            processed_segments.append(self.segments[0])
        while p1 < len(self.segments) - 2 and p2 < len(self.segments):
            if self.segments[p1 + 1].duration <= 0.2:
                tot_duration = self.segments[p1].duration + self.segments[p1 + 1].duration + self.segments[p2].duration
                if tot_duration < self.max_segment_length:
                    print(
                        "Merging segments {}-{}-{}".format(self.segments[p1], self.segments[p1 + 1], self.segments[p2]))
                    merged_segment = Segment(self.segments[p1].start_time, self.segments[p2].end_time, vocal=True)
                    processed_segments.append(merged_segment)
                    p1 = p2 + 2
                    p2 = p1 + 2
                    continue
            processed_segments.append(self.segments[p1])
            processed_segments.append(self.segments[p1 + 1])
            p1 = p2
            p2 = p1 + 2
        while p1 < len(self.segments):
            processed_segments.append(self.segments[p1])
            p1 += 1
        self.segments = processed_segments


class VadBufferGenerator(Generator):
    model, utils = torch.hub.load('snakers4/silero-vad', model='silero_vad', force_reload=False)
    get_speech_timestamps, _, _, _, _ = utils

    def __init__(self, filepath: str, perc_vocal_windows: float = 0.5, min_silence_length: float = 0.4,
                 processing_sr: int = 16000, buffer_limit_duration: float = 300, **kwargs):
        super().__init__(filepath, perc_vocal_windows, min_silence_length, processing_sr, buffer_limit_duration)
        window_duration_ms = kwargs.get("window_duration_ms", 30)
        self.window_size_samples = int((window_duration_ms * self.processing_sr) / 1000)
        self.threshold = self.perc_vocal_windows
        self.max_segment_length = kwargs.get("max_segment_length", 15)
        self.min_silence_len = int(self.min_silence_length * 1000)  # ms

    def split_at_silence(self, segment: Segment):
        if segment.duration <= self.max_segment_length:
            return [segment]
        else:
            vad = BufferGenerator(self.file, perc_vocal_windows=self.perc_vocal_windows + 0.1)
            vad._start = segment.start_time
            vad._end = segment.end_time
            vad.split_audio_to_segments(False)
            return vad.segments

    def post_process_segments(self, segments: List[Segment]):
        f_segments = []
        while len(segments):
            segment = segments.pop(0)
            if segment.duration > self.max_segment_length:
                split_segments = self.split_at_silence(segment)
                f_segments += split_segments
            else:
                f_segments.append(segment)
        return f_segments

    def split_audio_to_segments(self):
        segments = []
        while self._start < self._end:
            _end = min(self._start + self.buffer_limit_duration, self._end)
            samples = self.stream.read(self._start, _end)
            samples = librosa.resample(samples, orig_sr=self.stream.stream.samplerate, target_sr=self.processing_sr)
            samples = torch.Tensor(samples)
            segments_ = get_speech_timestamps(
                samples,
                self.model,
                threshold=self.threshold,
                sampling_rate=self.processing_sr,
                window_size_samples=self.window_size_samples,
                min_silence_duration_ms=self.min_silence_len,
                max_speech_duration_s=self.max_segment_length - 0.5,  # Remove as pad to curb nearing max_segment_len
            )
            if len(segments) and len(segments_):
                print("merging {} and {}".format(segments[-1], segments_))
                offset = int(self._start * self.processing_sr)
                segments[-1]["end"] = segments_[0]["end"] + offset
                segments += [{"start": seg["start"] + offset, "end": seg["end"] + offset} for seg in segments_[1:]]
            else:
                segments += segments_
            self._start = _end
        return self.post_process_segments(
            [Segment(seg["start"] / self.processing_sr, seg["end"] / self.processing_sr, vocal=True)
             for seg in segments])

def split_transcription_into_multiple_segments(segment: Segment, word_chunks: list, max_length: float):
    new_segments = []
    current_words = []
    current_start = None
    current_end = None
    last_punctuation_index = None
    punctuation_set = {'.', '?', '!', ',', ';', ':'}

    for word in word_chunks:
        word_start, word_end = word["timestamp"][0], word["timestamp"][1]
        if current_start is None:
            current_start = word_start
        current_words.append(word)
        current_end = word_end
        text_stripped = word["text"].strip()
        if text_stripped and text_stripped[-1] in punctuation_set:
            last_punctuation_index = len(current_words) - 1
        if word_end - current_start > max_length:
            if last_punctuation_index is not None:
                split_index = last_punctuation_index + 1
            else:
                split_index = len(current_words) - 1
            segment_words = current_words[:split_index]
            if segment_words:
                new_seg = Segment(
                    start_time=segment.start_time + current_start,
                    end_time=segment.start_time + segment_words[-1]["timestamp"][1],
                    vocal=True,
                    text=" ".join(w["text"] for w in segment_words),
                )
                new_segments.append(new_seg)
            leftover_words = current_words[split_index:]
            if leftover_words:
                current_start = leftover_words[0]["timestamp"][0]
                current_end = leftover_words[-1]["timestamp"][1]
            else:
                current_start = None
                current_end = None
            current_words = leftover_words
            last_punctuation_index = None
            for i, w in enumerate(current_words):
                if w["text"].strip() and w["text"].strip()[-1] in punctuation_set:
                    last_punctuation_index = i
    if current_words:
        new_seg = Segment(
            start_time=segment.start_time + current_start,
            end_time=segment.start_time + current_end,
            vocal=True,
            text=" ".join(w["text"] for w in current_words),
        )
        new_segments.append(new_seg)
    print("split segments", new_segments)
    return new_segments


def load_whisper_model_and_processor(model_name: str = "openai/whisper-small", quantize: bool = False):
    models = WHISPER_MODELS.get(model_name, None)
    if models is None:
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                load_in_8bit=quantize,
                device_map="auto"
            )
            processor = WhisperProcessor.from_pretrained(model_name)
            WHISPER_MODELS[model_name] = model, processor
        except:
            raise ValueError("Unknown model on huggingface {}".format(model_name))
    return WHISPER_MODELS[model_name]


def transcribe_audio_segments(audio_file: str, segments: List[Segment], model_type: str = "small",
                              quantize: bool = False, language: Optional[str] = None, word_timestamps: bool = True,
                              max_length: float = 15, supress_tokens: bool = False):
    model, processor = load_whisper_model_and_processor(model_type, quantize)
    asr_pipe = pipeline("automatic-speech-recognition", model=model,
                        tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor,
                        return_timestamps=True if word_timestamps else False)
    if language is not None:
        asr_pipe.model.config.forced_decoder_ids = asr_pipe.tokenizer.get_decoder_prompt_ids(language=language,
                                                                                             task="transcribe")
    if supress_tokens:
        asr_pipe.generation_config.suppress_tokens = None
    reader = WavReader(audio_file)
    transcribed_segments = []
    while len(segments):
        segment = segments.pop(0)
        if segment.vocal:
            samples = reader.read(segment.start_time, segment.end_time)
            samples = librosa.resample(samples, orig_sr=reader.stream.samplerate, target_sr=16000)
            samples = samples.astype(np.float32)
            samples = np.concatenate([np.zeros(2000), samples, np.zeros(2000)],
                                     axis=-1)  # Padding helps with possible abrupt cuts at the start and end, for ex>) Hi everybody was going to bye everybody
            generate_kwargs = {}
            if language is not None:
                segment.language = language
                generate_kwargs.update({"language": language, "task": "transcribe"})
            if word_timestamps:
                res = asr_pipe(samples, generate_kwargs=generate_kwargs, return_timestamps="word")
            else:
                res = asr_pipe(samples, generate_kwargs=generate_kwargs)
            segment.text = res["text"].strip()
            print("transcription for segment {}-{} is {}".format(segment.start_time, segment.end_time, segment.text))
            if segment.duration > max_length and word_timestamps:
                print("Splitting transcription")
                transcribed_segments += split_transcription_into_multiple_segments(segment, res["chunks"], max_length)
            else:
                transcribed_segments.append(segment)
        else:
            transcribed_segments.append(segment)
    return transcribed_segments


def get_audio_segments(audio_file: str, vad_model: str, perc_vocal_windows: float = 0.5,
                       min_silence_length: float = 0.5,
                       processing_sr: int = 16000, buffer_limit_duration: float = 600, **kwargs):
    if vad_model == "silero":
        vad = VadBufferGenerator(audio_file, perc_vocal_windows, min_silence_length, processing_sr,
                                 buffer_limit_duration,
                                 **kwargs)
    else:
        vad = BufferGenerator(audio_file, perc_vocal_windows, min_silence_length, processing_sr, buffer_limit_duration,
                              **kwargs)

    segments = vad.split_audio_to_segments()
    return segments


def save_segments_to_json(segments: List[Segment], save_json_path: str):
    ct = 1
    seg_list = []
    for segment in segments:
        if segment.vocal:
            seg_list.append({"chunk_id": ct, "chunk_length": segment.duration, "text": segment.text,
                             "start_time": segment.start_time, "end_time": segment.end_time})
            ct += 1
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(seg_list, f, indent=4, ensure_ascii=False)
        print("Saved segments to {}".format(save_json_path))
    return seg_list

if __name__ == "__main__":
    aud_file = "sarvam_audio.wav"
    model = "openai/whisper-large-v3"
    transcription_json = "transcriptions2_sarvam.json"
    segments = get_audio_segments(aud_file, "silero", 0.4, min_silence_length=0.5, max_segment_length=14)
    segments = transcribe_audio_segments(aud_file, segments, model, quantize=True, language="en", word_timestamps=True, max_length=15)
    seg_list = save_segments_to_json(segments, transcription_json)