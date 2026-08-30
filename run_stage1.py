import os
import sys
import logging
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime, timezone
import pandas as pd
import soundfile as sf
import imageio_ffmpeg

# Ensure ffmpeg binary from imageio_ffmpeg is in PATH BEFORE importing yt_dlp
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)
if ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_dir + os.path.pathsep + os.environ.get("PATH", "")

import yt_dlp

print(f"Using ffmpeg executable: {ffmpeg_exe}")

# Project Root Setup
PROJECT_ROOT = os.path.abspath("./legal_speech_rag_dataset")
FOLDERS = {
    "raw_videos":  os.path.join(PROJECT_ROOT, "raw_videos"),
    "audio":       os.path.join(PROJECT_ROOT, "audio"),
    "diarization": os.path.join(PROJECT_ROOT, "diarization"),
    "transcripts": os.path.join(PROJECT_ROOT, "transcripts"),
    "processed":   os.path.join(PROJECT_ROOT, "processed"),
    "metadata":    os.path.join(PROJECT_ROOT, "metadata"),
    "documents":   os.path.join(PROJECT_ROOT, "documents"),
    "logs":        os.path.join(PROJECT_ROOT, "logs"),
}

for path in FOLDERS.values():
    os.makedirs(path, exist_ok=True)

METADATA_CSV = os.path.join(FOLDERS["metadata"], "video_metadata.csv")
LOG_FILE = os.path.join(FOLDERS["logs"], "stage1_acquisition.log")

# Setup Logging
logger = logging.getLogger("stage1")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(sh)

logger.info("Stage 1 logging initialized.")
print("Folders ready:")
for k, v in FOLDERS.items():
    print(f"  {k:12s} -> {v}")

TARGET_VIDEO_COUNT = 100
MAX_PARALLEL_WORKERS = 16
csv_lock = Lock()

SEARCH_KEYWORDS = [
    "Indian national moot court competition final round",
    "NLU moot court final round India",
    "moot court competition India semi final",
    "Nani Palkhivala moot court",
    "K K Luthra memorial moot court",
    "Bar Council of India moot court competition",
    "national law university moot court oral round",
    "Symbiosis Law School moot court final",
    "ILS Law College moot court",
    "India law school moot court final round",
    "NLSIU moot court final",
    "NALSAR moot court oral argument",
    "GNLU moot court competition",
    "WBNUJS moot court competition",
    "Supreme Court of India argument",
    "High Court India oral arguments",
    "NLIU Bhopal moot court",
    "RGNUL moot court final",
    "HNLU moot court",
    "Amity Law School moot court final",
]

INDIAN_MARKERS = [
    "india", "nls", "nalsar", "nlu", "nliu", "ils law", "symbiosis", "gnlu", "rgnul",
    "hnlu", "dnlu", "clc", "nujs", "nlsiu", "bar council", "high court", "supreme court",
    "amity", "jindal", "delhi university", "mumbai", "christ law", "vit law", "moot"
]

EXCLUDE_MARKERS = [
    "wagner", "froessel", "nyu", "siu law", "animal law", "cultural heritage",
    "stetson", "willamette", "jessup", "ames moot", "uk supreme court", "us supreme court",
    "icj", "international criminal court", "harvard", "yale", "oxford", "cambridge"
]

def is_indian_legal(title: str) -> bool:
    t = (title or "").lower()
    if any(m in t for m in EXCLUDE_MARKERS):
        return False
    if any(m in t for m in INDIAN_MARKERS):
        return True
    return False

CASE_ID_PREFIX = "MOOT_IND"

METADATA_COLUMNS = [
    "video_id", "case_id", "case_name", "video_url", "source",
    "upload_date", "duration_sec", "video_path", "audio_path",
    "status", "error_message", "downloaded_at"
]

if os.path.exists(METADATA_CSV):
    metadata_df = pd.read_csv(METADATA_CSV, dtype=str)
    valid_mask = metadata_df["case_name"].apply(is_indian_legal)
    if not valid_mask.all():
        logger.info(f"Filtering existing metadata to keep only Indian law entries ({valid_mask.sum()}/{len(metadata_df)})")
        metadata_df = metadata_df[valid_mask].reset_index(drop=True)
        metadata_df.to_csv(METADATA_CSV, index=False)
else:
    metadata_df = pd.DataFrame(columns=METADATA_COLUMNS)
    metadata_df.to_csv(METADATA_CSV, index=False)
    logger.info("Created new metadata file")

def make_video_id(url: str) -> str:
    return "VID_IND_" + hashlib.sha1(url.encode()).hexdigest()[:10]

def process_single_video(idx, row):
    video_id = row["video_id"]
    url = row["video_url"]
    audio_out = os.path.abspath(os.path.join(FOLDERS["audio"], f"{video_id}.wav"))
    temp_dl = os.path.abspath(os.path.join(FOLDERS["audio"], f"{video_id}_temp"))

    if row.get("status") == "audio_extracted" and os.path.exists(audio_out):
        return idx, {"status": "audio_extracted", "audio_path": audio_out}

    common_opts = {
        "quiet": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 10,
        "ffmpeg_location": ffmpeg_exe,
    }

    try:
        ydl_opts = {
            **common_opts,
            "format": "bestaudio/best",
            "outtmpl": temp_dl + ".%(ext)s",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            dl_file = ydl.prepare_filename(info)

        # Convert downloaded audio stream to mono 16kHz WAV using ffmpeg
        cmd = [ffmpeg_exe, "-y", "-i", dl_file, "-ac", "1", "-ar", "16000", audio_out]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Clean up temp download file
        if os.path.exists(dl_file) and dl_file != audio_out:
            try:
                os.remove(dl_file)
            except Exception:
                pass

        duration = info.get("duration", 0)
        upload_date = info.get("upload_date", "")

        res = {
            "status": "audio_extracted",
            "video_path": "",
            "audio_path": audio_out,
            "duration_sec": str(duration),
            "upload_date": str(upload_date),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "error_message": "",
        }
    except Exception as e:
        logger.error(f"Download/Extraction failed for {video_id}: {e}")
        res = {"status": "failed", "error_message": str(e)}

    with csv_lock:
        for k, v in res.items():
            metadata_df.at[idx, k] = v
        metadata_df.to_csv(METADATA_CSV, index=False)

    return idx, res

pending_mask = metadata_df["status"] != "audio_extracted"
pending_indices = metadata_df[pending_mask].index.tolist()
logger.info(f"Processing {len(pending_indices)} pending Indian legal video(s) using {MAX_PARALLEL_WORKERS} parallel workers...")

completed_counter = 0
with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
    futures = {
        executor.submit(process_single_video, idx, metadata_df.loc[idx]): idx
        for idx in pending_indices
    }
    for future in as_completed(futures):
        idx = futures[future]
        completed_counter += 1
        row = metadata_df.loc[idx]
        try:
            _, res = future.result()
            if res.get("status") == "audio_extracted":
                logger.info(f"[{completed_counter}/{len(pending_indices)}] COMPLETED: {row['video_id']} - {row['case_name'][:50]}")
            else:
                logger.warning(f"[{completed_counter}/{len(pending_indices)}] FAILED: {row['video_id']}")
        except Exception as e:
            logger.error(f"Error processing index {idx}: {e}")

extracted_count = len(metadata_df[metadata_df["status"] == "audio_extracted"])
print(f"\nSTAGE 1 HIGH-SPEED PARALLEL EXTRACTION COMPLETE! Total audio_extracted files: {extracted_count}/{len(metadata_df)}")
