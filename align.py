import json
from pathlib import Path
from timething import dataset, job, utils

def align_audio_text(
    language: str,
    audio_file: Path,
    transcript_file: Path,
    batch_size: int,
    n_workers: int,
    k_shingles: int,
    seconds_per_window: int,
    offline: bool,
):
    # retrieve the config for the given language
    cfg = utils.load_config(language, k_shingles, local_files_only=offline)

    # read in the transcript
    with open(transcript_file, "r") as f:
        transcript = f.read()
        transcript = " ".join(transcript.lower().splitlines())

    # construct the dataset
    ds = dataset.WindowedTrackDataset(
        Path(audio_file),
        Path(audio_file).suffix[1:],
        transcript,
        seconds_per_window * 1000,
        seconds_per_window * 1000,
        16000,
    )
    j = job.LongTrackJob(cfg, ds, batch_size=batch_size, n_workers=n_workers)
    alignment = j.run()
    meta = {
        "start": (alignment.chars_cleaned[0].start * 320) / 16000,
        "asr_similarity_score": alignment.partition_score,
        "avg_alignment_prob": sum([s.score for s in alignment.chars_cleaned])/len(alignment.chars_cleaned)
    }
    return meta

if __name__ == "__main__":
    import os
    metas = []
    failed = set()
    DATA_DIR = ""
    for book_id in os.listdir(DATA_DIR):
        book_dir = os.path.join(DATA_DIR, book_id)
        chap_ct = 0
        for chap in os.listdir(book_dir):
            if chap_ct == 5:
                continue
            chapter_dir = os.path.join(book_dir, str(chap))
            audio = os.path.join(chapter_dir, "audio.mp3")
            trans = os.path.join(chapter_dir, "transcription.txt")
            try:
                meta = align_audio_text("hindi", audio, trans, 4, 5, 2, 10, False)
                meta["chapter"] = chapter_dir
                metas.append(meta)
                print(meta)
                chap_ct += 1
            except Exception as e:
                print("Failed {}".format(e))
                failed.add(chapter_dir)
    print(metas)
    with open(os.path.join(DATA_DIR, "alignment_.json"), "w") as f:
        json.dump(metas, f)
    print(failed)

