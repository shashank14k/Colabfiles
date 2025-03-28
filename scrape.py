import os
from typing import List
from multiprocessing import Process, Lock, Manager
from selenium import webdriver
from selenium.webdriver.common.by import By
import json
import time
import requests
import hashlib
from transcript import WavReader

def compute_sha256(data_bytes):
    sha = hashlib.sha256()
    sha.update(data_bytes)
    return sha.hexdigest()

def get_all_book_ids(base_url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    book_ids = []
    try:
        driver.get(base_url)
        script_element = driver.find_element(By.ID, "__NEXT_DATA__")
        json_data = json.loads(script_element.get_attribute("innerHTML"))
        books = json_data["props"]["pageProps"]["books"]
        for book in books:
            book_id = book["book_id"]
            book_name = book["name"]
            book_ids.append({"book_id": book_id, "name": book_name, "chapters": book["chapters"]})
        print("Found book IDs:", book_ids)
    except Exception as e:
        print("Error fetching book IDs:", e)
    finally:
        driver.quit()
    return book_ids

def verify_worker(base_url, chapter_ids, lock, save_dir, failed_ids):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    print("Worker {} received {}".format(os.getpid(), chapter_ids))
    for chap in chapter_ids:
        folder = f"{save_dir}/chapter_{chap}"
        os.makedirs(folder, exist_ok=True)
        if os.path.exists(os.path.join(folder, "transcription.txt")):
            if os.path.exists(os.path.join(folder, "audio.mp3")):
                url = f"{base_url}/{chap}"
                driver.get(url)
                time.sleep(5)
                try:
                    video_element = driver.find_element(By.CLASS_NAME, "audio-player")
                    audio_url = video_element.get_attribute("src")
                    audio_response = requests.get(audio_url)
                    with open(os.path.join(folder, "audio.mp3"), "rb") as f:
                        local_content = f.read()
                    local_sha = compute_sha256(local_content)
                    remote_sha = compute_sha256(audio_response.content)
                    if local_sha == remote_sha:
                        print("Verified {}".format(folder))
                    else:
                        with lock:
                            failed_ids.append(chapter_ids)
                except Exception as e:
                    print("Error:", e)


def scrape_worker(base_url, chapter_ids, lock, save_dir, failed_ids):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    print("Worker {} received {}".format(os.getpid(), chapter_ids))
    for chap in chapter_ids:
        folder = f"{save_dir}/chapter_{chap}"
        os.makedirs(folder, exist_ok=True)
        if os.path.exists(os.path.join(folder, "transcription.txt")):
            if os.path.exists(os.path.join(folder, "audio.mp3")):
                try:
                    WavReader(os.path.join(folder, "audio.mp3"))
                    print("Skipping existing data for chapter {}".format(chap))
                    continue
                except:
                    pass
        print("Worker {} Scraping chapter {}".format(os.getpid(), chap))
        time.sleep(5)
        url = f"{base_url}/{chap}"
        driver.get(url)
        time.sleep(5)
        try:
            video_element = driver.find_element(By.CLASS_NAME, "audio-player")
            audio_url = video_element.get_attribute("src")

            if audio_url and audio_url != "_":
                print("Audio URL:", audio_url)
            else:
                print("Could not find Audio URL")

            script_element = driver.find_element(By.ID, "__NEXT_DATA__")
            json_data = json.loads(script_element.get_attribute("innerHTML"))
            verses = []
            for chapter in json_data["props"]["pageProps"]["chapterText"]:
                verse_text = chapter.get("verse_text", "")
                verses.append(verse_text)

            with open(os.path.join(folder, "transcription.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(verses))

            if audio_url and audio_url != "_":
                try:
                    audio_response = requests.get(audio_url)
                    with open(os.path.join(folder, "audio.mp3"), "wb") as f:
                        f.write(audio_response.content)
                except Exception as e:
                    print("Error downloading audio:", e)
                    with lock:
                        failed_ids.append(chap)
                try:
                    WavReader(os.path.join(folder, "audio.mp3"))
                    print("Audio saved as 'audio.mp3'.")
                except:
                    with lock:
                        failed_ids.append(chap)
                    continue


        except Exception as e:
            print("Error:", e)
            with Lock:
                failed_ids.append(chap)
    driver.quit()

def scrape_audio_and_text(url, save_dir, chapter_ids: List, n_workers: int = 8):
    lock = Lock()
    workers = []
    os.makedirs(save_dir, exist_ok=True)
    failed_ids = Manager().list()
    chunk_size = len(chapter_ids) // n_workers
    remainder = len(chapter_ids) % n_workers
    start = 0

    for pid in range(n_workers):
        end = start + chunk_size + (1 if pid < remainder else 0)
        chapters_chunk = chapter_ids[start:end]
        start = end
        if chapters_chunk:
            proc = Process(target=scrape_worker, args=(url, chapters_chunk, lock, save_dir, failed_ids))
            workers.append(proc)
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    print("Scraping failed for {}".format(failed_ids))
    return list(failed_ids)

def verify_scraping(url, save_dir, chapter_ids, n_workers):
    lock = Lock()
    workers = []
    failed_ids = Manager().list()
    chunk_size = len(chapter_ids) // n_workers
    remainder = len(chapter_ids) % n_workers
    start = 0

    for pid in range(n_workers):
        end = start + chunk_size + (1 if pid < remainder else 0)
        chapters_chunk = chapter_ids[start:end]
        start = end
        if chapters_chunk:
            proc = Process(target=verify_worker, args=(url, chapters_chunk, lock, save_dir, failed_ids))
            workers.append(proc)
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    print("Scraping failed for {}".format(failed_ids))
    return failed_ids

if __name__ == "__main__":
    n_workers = 4
    url = "https://live.bible.is/bible/HINHNV/"
    save_dir = "data"
    #book_data = get_all_book_ids(url)
    with open("book_data.json", "r") as f:
        # json.dump(book_data, f, indent=4,
        #           ensure_ascii=False)
        book_data = json.load(f)
    failed = {}
    nn = 0
    for book in book_data:
        book_url = url + book["book_id"]
        save_dir_ = os.path.join(save_dir, book["book_id"])
        n_chaps = len(book["chapters"])
        nn += n_chaps
        if len(os.listdir(save_dir_)) != n_chaps:
            print("ISSUE", save_dir_)
        #failed[book["book_id"]] = scrape_audio_and_text(book_url, save_dir_, book["chapters"], n_workers)
        failed[book["book_id"]] = verify_scraping(book_url, save_dir_, book["chapters"], n_workers)
    print("--------------------Failed------------------")
    print(failed)
    print(nn)
