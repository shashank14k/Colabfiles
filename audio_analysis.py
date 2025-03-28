from transcript import WavReader
import numpy as np
import soundfile as sf
import torch
import librosa
import json
from df.enhance import enhance, load_audio, init_df
from transcript import get_audio_segments, save_segments_to_json, Segment
import torchaudio.compliance.kaldi as kaldi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
filter_model, df_state, _ = init_df()
predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEVICE).eval()

def build_style_model(model_path):
    from campplus.DTDNN import CAMPPlus
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_sd = torch.load(model_path, map_location='cpu')
    campplus_model.load_state_dict(campplus_sd)
    campplus_model.eval()
    campplus_model.to(DEVICE)
    return campplus_model


def compute_style(style_model, waves_16k, wave_lengths_16k):
    feat_list = []
    B = waves_16k.size(0)
    for bib in range(B):
        feat = kaldi.fbank(
            waves_16k[bib:bib + 1, :wave_lengths_16k[bib]],
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        feat_list.append(feat)
    y_list = []
    with torch.no_grad():
        for feat in feat_list:
            y = style_model(feat.unsqueeze(0))
            y_list.append(y)
    return torch.cat(y_list, dim=0)



def denoise_audio(audio_file: str, start, end):
    noisy_audio, sr = load_audio(audio_file, sr=df_state.sr(), format="wav")
    noisy_audio = noisy_audio[:, int(start * df_state.sr()):int(end * df_state.sr())]
    cleaned_audio = enhance(filter_model, df_state, noisy_audio)
    min_length = min(len(noisy_audio), len(cleaned_audio))
    noisy_audio = noisy_audio[:min_length]
    cleaned_audio = cleaned_audio[:min_length]
    estimated_noise = noisy_audio - cleaned_audio
    return cleaned_audio.numpy().mean(axis=0), estimated_noise.numpy().mean(axis=0), df_state.sr()


def estimate_snr(signal: np.ndarray, noise: np.ndarray):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    return float(snr_db)


def compute_rms(audio_signal):
    power = audio_signal ** 2
    return float(np.sqrt(np.quantile(power, 0.25))), float(np.sqrt(np.mean(power))), float(np.sqrt(np.quantile(power, 0.75)))


if __name__ == "__main__":
    import os
    import random
    DATA_DIR = None
    metas = []
    campplus_model = "campplus_cn_common.bin"
    segment_duration = 150 # Analysis on random 150 second segments
    if DATA_DIR is not None and os.path.exists(DATA_DIR):
        se = build_style_model(campplus_model)
        for book_id in os.listdir(DATA_DIR):
            book_dir = os.path.join(DATA_DIR, book_id)
            for chap in os.listdir(book_dir):
                chapter_dir = os.path.join(book_dir, str(chap))
                print("Processing chapter {}".format(chapter_dir))
                aud_file = os.path.join(chapter_dir, "audio.mp3")
                reader = WavReader(aud_file)
                meta = {"book": book_id+"_"+str(chap), "sample_rate": reader.stream.samplerate, "duration": reader.duration}
                start = random.uniform(0, max(1, reader.duration - segment_duration))
                signal, noise, sr = denoise_audio(aud_file, start, start + segment_duration)
                sf.write(os.path.join(chapter_dir, "cleaned.wav"), signal, sr)
                sf.write(os.path.join(chapter_dir, "noise.wav"), noise, sr)
                segments = get_audio_segments(aud_file, "silero", 0.3, 0.4)
                save_segments_to_json(segments, os.path.join(chapter_dir, "segments.json"))
                with open(os.path.join(chapter_dir, "segments.json"), "r") as f:
                    data = json.load(f)
                segments = [Segment(start_time=dic["start_time"], end_time=dic["end_time"], vocal=True, text=dic["text"])
                                  for dic in data]
                top_5_segments = sorted(segments, key=lambda s: s.end_time - s.start_time, reverse=True)[:5]
                meta["snr"] = estimate_snr(signal, noise)
                meta["rms_stats"] = compute_rms(signal)
                mos_score = []
                spk_emb = []
                for seg in top_5_segments:
                    wave = reader.read(seg.start_time, seg.end_time)
                    wave = librosa.resample(wave, orig_sr=reader.stream.samplerate, target_sr=16000)
                    wave = torch.tensor(wave).unsqueeze(0).float().cuda()
                    with torch.no_grad():
                        score = predictor(wave, reader.stream.samplerate)
                    spk_emb_ = compute_style(se, wave, torch.LongTensor([wave.shape[-1]])).cpu().numpy()
                    spk_emb.append(spk_emb_)
                    mos_score.append(score.item())
                spk_emb = torch.tensor(spk_emb).mean(0)
                meta["mos"] = np.mean(mos_score)
                meta["vocal_chunk_durations"] = [seg.duration for seg in segments]
                metas.append(meta)
                np.save(os.path.join(chapter_dir, "spk.npy"), spk_emb)
        with open(os.path.join(DATA_DIR, "audio_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(metas, f, indent=4, ensure_ascii=False)