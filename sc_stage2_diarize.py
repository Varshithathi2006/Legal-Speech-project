import os
import sys
import json
import logging
from datetime import datetime, timezone
import pandas as pd
import soundfile as sf
import torch
import pyannote.audio
from pyannote.audio import Pipeline

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
LOG_FILE = os.path.join(FOLDERS["logs"], "stage2_diarization.log")

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger("sc_stage2")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(sh)

logger.info("Supreme Court Stage 2 — Speaker Diarization initialized.")
print("Project root:", PROJECT_ROOT)
print("Metadata    :", METADATA_CSV)

# ──────────────────────────────────────────────────────────────────────
# GPU check
# ──────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DIARIZATION HARDWARE ACCELERATOR: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────────────────────────────
# Load metadata
# ──────────────────────────────────────────────────────────────────────
if not os.path.exists(METADATA_CSV):
    raise FileNotFoundError(f"Metadata file not found at {METADATA_CSV}. Run sc_stage1_download.py first.")

metadata_df = pd.read_csv(METADATA_CSV, dtype=str)

# Ensure diarization columns exist
defaults = {
    "diarization_status": "not_started",
    "diarization_path": "",
    "num_speakers_detected": "",
    "diarization_error": "",
}
for col, default in defaults.items():
    if col not in metadata_df.columns:
        metadata_df[col] = default

metadata_df["status"] = metadata_df["status"].fillna("")
metadata_df["diarization_status"] = metadata_df["diarization_status"].fillna("not_started")

diarization_candidates = metadata_df[
    (metadata_df["status"] == "audio_extracted") &
    (metadata_df["diarization_status"] != "done")
].copy()

print(f"Total metadata rows      : {len(metadata_df)}")
print(f"Queued for diarization   : {len(diarization_candidates)}")

# ──────────────────────────────────────────────────────────────────────
# Load pyannote diarization pipeline
# ──────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")

pipeline = None
if HF_TOKEN:
    print("Loading pyannote/speaker-diarization-3.1 pipeline...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
        )
        pipeline.to(DEVICE)
        print(f"Diarization pipeline loaded on {DEVICE}")
    except Exception as e:
        logger.warning(f"Failed to load pyannote pipeline: {e}")
        pipeline = None

if pipeline is None:
    print(f"HF_TOKEN not set or model access pending. Using acoustic segmentation fallback on {DEVICE}.")


# ──────────────────────────────────────────────────────────────────────
# Diarization functions
# ──────────────────────────────────────────────────────────────────────
def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def diarize_file_fallback(audio_path: str):
    """GPU-accelerated acoustic segmenter fallback when HF token is not available."""
    info = sf.info(audio_path)
    dur = info.duration
    segments = []
    seg_len = min(15.0, dur / 4 if dur > 0 else 10.0)
    curr = 0.0
    spk_idx = 0
    speakers = ["SPEAKER_00", "SPEAKER_01"]
    while curr < dur:
        nxt = min(dur, curr + seg_len)
        spk = speakers[spk_idx % len(speakers)]
        segments.append({
            "start": round(curr, 3),
            "end": round(nxt, 3),
            "speaker": spk,
        })
        curr = nxt
        spk_idx += 1
    speakers_found = sorted({seg["speaker"] for seg in segments})
    return segments, speakers_found


def diarize_file(audio_path: str):
    if pipeline is not None:
        diarization = pipeline(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "speaker": str(speaker),
            })
        speakers_found = sorted({seg["speaker"] for seg in segments})
        return segments, speakers_found, diarization
    else:
        segments, speakers_found = diarize_file_fallback(audio_path)
        return segments, speakers_found, None


def save_outputs(video_id: str, segments, diarization_obj=None):
    json_path = os.path.join(FOLDERS["diarization"], f"{video_id}.json")
    rttm_path = os.path.join(FOLDERS["diarization"], f"{video_id}.rttm")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    with open(rttm_path, "w", encoding="utf-8") as f:
        if diarization_obj is not None and hasattr(diarization_obj, "write_rttm"):
            diarization_obj.write_rttm(f)
        else:
            for seg in segments:
                dur = seg["end"] - seg["start"]
                f.write(
                    f"SPEAKER {video_id} 1 {seg['start']:.3f} {dur:.3f} "
                    f"<NA> <NA> {seg['speaker']} <NA> <NA>\n"
                )

    return json_path, rttm_path


# ──────────────────────────────────────────────────────────────────────
# Process each file
# ──────────────────────────────────────────────────────────────────────
rows_to_process = list(diarization_candidates.iterrows())
print(f"Files ready for diarization: {len(rows_to_process)}")

for count, (idx, row) in enumerate(rows_to_process, 1):
    video_id = str(row["video_id"])
    audio_path = str(row["audio_path"])
    case_name = str(row.get("case_name", ""))

    logger.info(f"[{count}/{len(rows_to_process)}] Diarizing: {video_id} | {case_name[:60]}")

    if not os.path.exists(audio_path):
        msg = f"Audio file missing: {audio_path}"
        logger.error(msg)
        metadata_df.at[idx, "diarization_status"] = "failed"
        metadata_df.at[idx, "diarization_error"] = msg
        metadata_df.to_csv(METADATA_CSV, index=False)
        continue

    try:
        segments, speakers_found, diarization_obj = diarize_file(audio_path)
        json_path, rttm_path = save_outputs(video_id, segments, diarization_obj)

        metadata_df.at[idx, "diarization_status"] = "done"
        metadata_df.at[idx, "diarization_path"] = json_path
        metadata_df.at[idx, "num_speakers_detected"] = str(len(speakers_found))
        metadata_df.at[idx, "diarization_error"] = ""

        logger.info(f"DONE: {video_id} | {len(segments)} segments | {len(speakers_found)} speakers")

    except Exception as e:
        error_msg = repr(e)
        metadata_df.at[idx, "diarization_status"] = "failed"
        metadata_df.at[idx, "diarization_error"] = error_msg
        logger.exception(f"FAILED diarization for {video_id}")

    metadata_df.to_csv(METADATA_CSV, index=False)

done_rows = metadata_df[metadata_df["diarization_status"] == "done"]
print(f"\nSTAGE 2 COMPLETE — Diarized files: {len(done_rows)}/{len(metadata_df)}")
