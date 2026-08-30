import os
import sys
import logging
import hashlib
import subprocess
import re
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

# ──────────────────────────────────────────────────────────────────────
# Project root is "supreme court" subfolder next to this script
# ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "supreme court")

FOLDERS = {
    "audio":       os.path.join(PROJECT_ROOT, "audio"),
    "diarization": os.path.join(PROJECT_ROOT, "diarization"),
    "transcripts": os.path.join(PROJECT_ROOT, "transcripts"),
    "metadata":    os.path.join(PROJECT_ROOT, "metadata"),
    "logs":        os.path.join(PROJECT_ROOT, "logs"),
}

for path in FOLDERS.values():
    os.makedirs(path, exist_ok=True)

METADATA_CSV = os.path.join(FOLDERS["metadata"], "video_metadata.csv")
LOG_FILE = os.path.join(FOLDERS["logs"], "stage1_download.log")

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger("sc_stage1")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(sh)

logger.info("Supreme Court Stage 1 — Audio Download initialized.")
print("Folders ready:")
for k, v in FOLDERS.items():
    print(f"  {k:12s} -> {v}")

# ──────────────────────────────────────────────────────────────────────
# Video list scraped from https://www.sci.gov.in/previous-sessions/#
# Each entry: (date_str, title, youtube_embed_url)
# ──────────────────────────────────────────────────────────────────────
RAW_VIDEOS = [
    # 2024 hearings
    ("05-12-2024", "Swearing-in ceremony of Hon'ble Shri Justice Manmohan", "https://www.youtube.com/embed/dyNek9ykiDk?rel=0"),
    ("08-11-2024", "Dated 08.11.2024", "https://www.youtube.com/embed/OtZlYzQ7Kok?rel=0"),
    ("08-11-2024", "Ceremonial Bench 08.11.2024", "https://www.youtube.com/embed/MEZGD-cWTDE?rel=0"),
    ("07-11-2024", "Constitution Bench dated 07.11.2024", "https://www.youtube.com/embed/ZLFGaqhatdc?rel=0"),
    ("06-11-2024", "Constitution Bench dated 06.11.2024", "https://www.youtube.com/embed/YTeR8rVWh40?rel=0"),
    ("05-11-2024", "Constitution Bench dated 05.11.2024", "https://www.youtube.com/embed/jQbY9mqdDYU?rel=0"),
    ("05-11-2024", "President Draupadi Murmu releases three publication of SCI", "https://www.youtube.com/embed/zyXnsqHcNiM?rel=0"),
    ("23-10-2024", "Date 23-10-2024", "https://www.youtube.com/embed/B728Jhuntqc?rel=0"),
    ("17-10-2024", "IN RE SECTION 6A OF THE CITIZENSHIP ACT 1955 WP(C) No. 274/2009", "https://www.youtube.com/embed/qnOo1L_ci4o?rel=0"),
    ("15-10-2024", "IN RE Alleged Rape and Murder RG Kar Medical College SMW(Crl) No. 2/2024", "https://www.youtube.com/embed/SlyxxBAhncA?rel=0"),
    ("14-10-2024", "Groundbreaking Ceremony of Expansion Building 14.10.2024", "https://www.youtube.com/embed/-J6HeXwoWM4?rel=0"),
    ("30-09-2024", "IN RE Alleged Rape and Murder RG Kar Medical College SMW(Crl) No. 2/2024 (2)", "https://www.youtube.com/embed/P2JUaoC2ij4?rel=0"),
    ("25-09-2024", "IN RE Remarks by HC Judge During Court Proceedings SMW(Crl) No. 9/2024", "https://www.youtube.com/embed/57J5RlHc89Q?rel=0"),
    ("25-09-2024", "IN RE Heritage Building Bombay HC SMW(C) No. 5/2024", "https://www.youtube.com/embed/qOXHLwRBOIk?rel=0"),
    ("17-09-2024", "Date 17.09.2024", "https://www.youtube.com/embed/aM9s_lVngb4?rel=0"),
    ("13-09-2024", "Inaugural Function International Arbitration and Rule of Law", "https://www.youtube.com/embed/YMjdNKXe_r0?rel=0"),
    ("12-09-2024", "Full Court Reference 12.09.2024", "https://www.youtube.com/embed/CRLelol7bDg?rel=0"),
    ("09-09-2024", "Date 09.09.2024", "https://www.youtube.com/embed/Y1x1Is_zqp8?rel=0"),
    ("30-08-2024", "Ceremonial Bench 30.08.2024", "https://www.youtube.com/embed/dQ2tGe9qJJY?rel=0"),
    ("30-08-2024", "Date 30.08.2024", "https://www.youtube.com/embed/oAwE_p6d1DA?rel=0"),
    ("29-08-2024", "Date 29.08.2024", "https://www.youtube.com/embed/-Qw-KeuL6XI?rel=0"),
    ("28-08-2024", "Date 28.08.2024", "https://www.youtube.com/embed/sJTsMrdkrvQ?rel=0"),
    ("27-08-2024", "Date 27.08.2024", "https://www.youtube.com/embed/rM8M2ivuNP4?rel=0"),
    ("14-08-2024", "Date 14.08.2024", "https://www.youtube.com/embed/L-XT3hmolds?rel=0"),
    ("07-08-2024", "Full Court Reference 07.08.2024", "https://www.youtube.com/embed/-0dVzF2qGgo?rel=0"),
    ("07-08-2024", "Date 07.08.2024", "https://www.youtube.com/embed/ibry_ix0JnE?rel=0"),
    ("03-08-2024", "Special Lok Adalat 2024", "https://www.youtube.com/embed/yNQYqNi1ss8?rel=0"),
    ("01-08-2024", "Date 01.08.2024", "https://www.youtube.com/embed/jrbN24rJ100?rel=0"),
    ("31-07-2024", "Date 31.07.2024", "https://www.youtube.com/embed/Li-lUDjO9TI?rel=0"),
    ("25-07-2024", "Date 25.07.2024", "https://www.youtube.com/embed/cit-VTZiyFk?rel=0"),
    ("23-07-2024", "Date 23.07.2024", "https://www.youtube.com/embed/7hmMOAf5nho?rel=0"),
    ("22-07-2024", "Date 22.07.2024", "https://www.youtube.com/embed/e-nl7jaUHVI?rel=0"),
    ("18-07-2024", "Swearing-in Ceremony 18.07.2024", "https://www.youtube.com/embed/BNsOZvU3AJ4?rel=0"),
    ("18-07-2024", "Date 18-07-2024", "https://www.youtube.com/embed/Zq0w3oeSNUM?rel=0"),
    ("16-07-2024", "Date 16.07.2024", "https://www.youtube.com/embed/9BjtzG4jEMM?rel=0"),
    ("11-07-2024", "Date 11.07.2024", "https://www.youtube.com/embed/j48Z2NJea3o?rel=0"),
    ("17-05-2024", "Date 17.05.2024", "https://www.youtube.com/embed/InVtyFQbvG0?rel=0"),
    ("08-05-2024", "Full Court Reference 08.05.2024", "https://www.youtube.com/embed/l9kmtaKn8uA?rel=0"),
    ("01-05-2024", "Date 01.05.2024", "https://www.youtube.com/embed/c4pDCzsiVoM?rel=0"),
    ("30-04-2024", "Date 30.04.2024", "https://www.youtube.com/embed/u7vdoNS33PA?rel=0"),
    ("25-04-2024", "Date 25.04.2024", "https://www.youtube.com/embed/DC-tK5uPCI4?rel=0"),
    ("24-04-2024", "Date 24.04.2024", "https://www.youtube.com/embed/QG0jF859eIY?rel=0"),
    ("23-04-2024", "Date 23.04.2024", "https://www.youtube.com/embed/7scTgf6MT1Q?rel=0"),
    ("18-04-2024", "Date 18.04.2024", "https://www.youtube.com/embed/iGrc_0KiVXM?rel=0"),
    ("16-04-2024", "Date 16.04.2024", "https://www.youtube.com/embed/HJYUA1BluJ4?rel=0"),
    ("16-04-2024", "Date 16.04.2024 (2)", "https://www.youtube.com/embed/SXXkRepwjJ8?rel=0"),
    ("10-04-2024", "Date 10.04.2024", "https://www.youtube.com/embed/Zj-PNeKF3aM?rel=0"),
    ("09-04-2024", "Date 09-04-2024", "https://www.youtube.com/embed/fDtShBUR7JY?rel=0"),
    ("04-04-2024", "Full Court Reference 04.04.2024", "https://www.youtube.com/embed/Q1yh9GnAziQ?rel=0"),
    ("04-04-2024", "Date 04.04.2024", "https://www.youtube.com/embed/wPfTD7Yz2EY?rel=0"),
    ("03-04-2024", "Date 03.04.2024", "https://www.youtube.com/embed/qns3DyQVkj8?rel=0"),
    ("02-04-2024", "Date 02-04-2024", "https://www.youtube.com/embed/xqlAvLPnv94?rel=0"),
    ("18-03-2024", "Date 18-03-2024", "https://www.youtube.com/embed/PVABVuNNReQ?rel=0"),
    ("15-03-2024", "Date 15-03-2024", "https://www.youtube.com/embed/TuvPAZWw9ag?rel=0"),
    ("14-03-2024", "Date 14-03-2024", "https://www.youtube.com/embed/9D-B9xQh1sU?rel=0"),
    ("13-03-2024", "Date 13-03-2024", "https://www.youtube.com/embed/8u0vGuSSoQs?rel=0"),
    ("12-03-2024", "Date 12-03-2024", "https://www.youtube.com/embed/sQrG6vRQjPg?rel=0"),
    ("11-03-2024", "Date 11-03-2024", "https://www.youtube.com/embed/fcJuqNAGCBg?rel=0"),
    ("06-03-2024", "Date 06-03-2024", "https://www.youtube.com/embed/fTHfxj52XEU?rel=0"),
    ("05-03-2024", "Date 05-03-2024", "https://www.youtube.com/embed/F-Otcfzfu8s?rel=0"),
    ("04-03-2024", "Date 04-03-2024", "https://www.youtube.com/embed/2DpJFt9TzwQ?rel=0"),
    ("29-02-2024", "Date 29-02-2024 CB Event 1", "https://www.youtube.com/embed/b_aOBT3eY9o?rel=0"),
    ("29-02-2024", "Date 29-02-2024 CB Event 2", "https://www.youtube.com/embed/7XYLPujLq0Q?rel=0"),
    ("28-02-2024", "Date 28-02-2024", "https://www.youtube.com/embed/iwISw1AKywQ?rel=0"),
    ("27-02-2024", "Date 27-02-2024", "https://www.youtube.com/embed/O8LK6Q5FzUw?rel=0"),
    ("15-02-2024", "Date 15-02-2024", "https://www.youtube.com/embed/bRCnCysazq8?rel=0"),
    ("10-02-2024", "Date 10-02-2024", "https://www.youtube.com/embed/8rK1eOnC1Sg?rel=0"),
    ("08-02-2024", "Date 08-02-2024", "https://www.youtube.com/embed/W9DypD696AM?rel=0"),
    ("07-02-2024", "Date 07-02-2024", "https://www.youtube.com/embed/Ez5jpAlNOj8?rel=0"),
    ("06-02-2024", "Date 06-02-2024", "https://www.youtube.com/embed/0LwZkaxXtUM?rel=0"),
    ("01-02-2024", "Date 01-02-2024", "https://www.youtube.com/embed/-JCpLjQIezw?rel=0"),
    ("31-01-2024", "Date 31-01-2024", "https://www.youtube.com/embed/KAtkdxXJDyM?rel=0"),
    ("30-01-2024", "Date 30-01-2024", "https://www.youtube.com/embed/el3O6EaIC8o?rel=0"),
    ("24-01-2024", "Date 24-01-2024", "https://www.youtube.com/embed/K3n0YVSsFrw?rel=0"),
    ("23-01-2024", "Date 23-01-2024", "https://www.youtube.com/embed/EZ6utg2ntpI?rel=0"),
    ("17-01-2024", "Date 17-01-2024", "https://www.youtube.com/embed/sWrsWLaHW5Q?rel=0"),
    ("11-01-2024", "Date 11-01-2024", "https://www.youtube.com/embed/-LkDvPEPhuE?rel=0"),
    ("10-01-2024", "Date 10-01-2024", "https://www.youtube.com/embed/QtizM2j5-CY?rel=0"),
    ("09-01-2024", "Date 09-01-2024", "https://www.youtube.com/embed/dEtwNsT4_CQ?rel=0"),
]


def extract_youtube_video_id(embed_url: str) -> str:
    """Extract the YouTube video ID from an embed URL."""
    # Handle normal embed: https://www.youtube.com/embed/VIDEO_ID?rel=0
    # Handle nested: https://www.youtube.com/embed/https://youtube.com/live/VIDEO_ID?...
    m = re.search(r"/embed/(?:https?://youtube\.com/live/)?([A-Za-z0-9_-]+)", embed_url)
    if m:
        return m.group(1)
    return ""


def make_video_id(youtube_id: str) -> str:
    """Create a deterministic short ID for a YouTube video."""
    return "SC_" + hashlib.sha1(youtube_id.encode()).hexdigest()[:10]


# ──────────────────────────────────────────────────────────────────────
# Build / load metadata
# ──────────────────────────────────────────────────────────────────────
METADATA_COLUMNS = [
    "video_id", "youtube_id", "case_name", "date", "video_url",
    "source", "duration_sec", "audio_path",
    "status", "error_message", "downloaded_at",
]

if os.path.exists(METADATA_CSV):
    metadata_df = pd.read_csv(METADATA_CSV, dtype=str)
    logger.info(f"Loaded existing metadata with {len(metadata_df)} rows")
else:
    metadata_df = pd.DataFrame(columns=METADATA_COLUMNS)
    logger.info("Created new metadata DataFrame")

existing_yt_ids = set(metadata_df["youtube_id"].dropna().values) if "youtube_id" in metadata_df.columns else set()

new_rows = []
for date_str, title, embed_url in RAW_VIDEOS:
    yt_id = extract_youtube_video_id(embed_url)
    if not yt_id:
        logger.warning(f"Could not extract YouTube ID from: {embed_url}")
        continue
    if yt_id in existing_yt_ids:
        continue
    vid_id = make_video_id(yt_id)
    watch_url = f"https://www.youtube.com/watch?v={yt_id}"
    new_rows.append({
        "video_id": vid_id,
        "youtube_id": yt_id,
        "case_name": title,
        "date": date_str,
        "video_url": watch_url,
        "source": "sci.gov.in",
        "duration_sec": "",
        "audio_path": "",
        "status": "pending",
        "error_message": "",
        "downloaded_at": "",
    })
    existing_yt_ids.add(yt_id)

if new_rows:
    metadata_df = pd.concat([metadata_df, pd.DataFrame(new_rows)], ignore_index=True)
    metadata_df.to_csv(METADATA_CSV, index=False)
    logger.info(f"Added {len(new_rows)} new video entries (total: {len(metadata_df)})")
else:
    logger.info("No new videos to add")

print(f"\nTotal videos in metadata: {len(metadata_df)}")

# ──────────────────────────────────────────────────────────────────────
# Download + extract audio
# ──────────────────────────────────────────────────────────────────────
MAX_PARALLEL_WORKERS = 4
csv_lock = Lock()


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

        # Convert to mono 16 kHz WAV
        cmd = [ffmpeg_exe, "-y", "-i", dl_file, "-ac", "1", "-ar", "16000", audio_out]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Cleanup temp file
        if os.path.exists(dl_file) and dl_file != audio_out:
            try:
                os.remove(dl_file)
            except Exception:
                pass

        duration = info.get("duration", 0)

        res = {
            "status": "audio_extracted",
            "audio_path": audio_out,
            "duration_sec": str(duration),
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
logger.info(f"Processing {len(pending_indices)} pending video(s) using {MAX_PARALLEL_WORKERS} workers...")

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
                logger.info(f"[{completed_counter}/{len(pending_indices)}] DONE: {row['video_id']} — {row['case_name'][:60]}")
            else:
                logger.warning(f"[{completed_counter}/{len(pending_indices)}] FAILED: {row['video_id']} — {res.get('error_message', '')[:80]}")
        except Exception as e:
            logger.error(f"Error processing index {idx}: {e}")

extracted_count = len(metadata_df[metadata_df["status"] == "audio_extracted"])
print(f"\nSTAGE 1 COMPLETE — Audio extracted: {extracted_count}/{len(metadata_df)}")
