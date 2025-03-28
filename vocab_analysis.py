import os
import json
from typing import Dict, List
from misaki import espeak


def count_phonemes_and_words(phoneme_map: Dict, word_map: Dict, phonemizer: espeak.EspeakG2P, transcriptions: List[str]):
    for text in transcriptions:
        phonemes = phonemizer(text)
        for ph in phonemes:
            if ph == " ":
                continue
            phoneme_map[ph] = phoneme_map.get(ph, 0) + 1
        for w in text.split():
            word_map[w] = word_map.get(w, 0) + 1


if __name__ == "__main__":
    phonemizer = espeak.EspeakG2P(language="hi")
    phoneme_map = {}
    word_map = {}
    DATA_DIR = ""
    for book_id in os.listdir(DATA_DIR):
        book_dir = os.path.join(DATA_DIR, book_id)
        for chap in os.listdir(book_dir):
            chapter_dir = os.path.join(book_dir, str(chap))
            txt_file = os.path.join(chapter_dir, "transcription.txt")
            if os.path.exists(txt_file):
                with open(txt_file, "r") as f:
                    transcriptions = [line.strip() for line in f.readlines()]
                print(f"Processing chapter {chap}")
                count_phonemes_and_words(phoneme_map, word_map, phonemizer, transcriptions)

    with open("phoneme_ct.json", "w", encoding="utf-8") as f:
        json.dump(phoneme_map, f, indent=4)
    with open("word_ct.json", "w", encoding="utf-8") as f:
        json.dump(word_map, f, indent=4)