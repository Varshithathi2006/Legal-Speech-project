import os
import sys
import json
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from faster_whisper import WhisperModel

# ──────────────────────────────────────────────────────────────────────
# Setup paths for Supreme Court track
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
LOG_FILE = os.path.join(FOLDERS["logs"], "sc_stage3_segmented.log")

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
class FlushHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logger = logging.getLogger("sc_stage3_segmented")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fh = FlushHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(fh)

sh = FlushStreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(sh)

logger.info("Initializing Supreme Court Stage 3 Segmented Transcription...")
print(f"Project root: {PROJECT_ROOT}")
print(f"Metadata    : {METADATA_CSV}")

# ──────────────────────────────────────────────────────────────────────
# Hardware Accelerator & Faster-Whisper Model Load
# ──────────────────────────────────────────────────────────────────────
from faster_whisper import WhisperModel, BatchedInferencePipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

print(f"\nTRANSCRIPTION HARDWARE: {DEVICE.upper()} (Compute Type: {COMPUTE_TYPE})")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

try:
    print(f"Loading faster-whisper 'small' model with BatchedInferencePipeline on {DEVICE} ({COMPUTE_TYPE})...")
    base_model = WhisperModel("small", device=DEVICE, compute_type=COMPUTE_TYPE)
    model = BatchedInferencePipeline(model=base_model)
    print(f"Batched Whisper pipeline loaded successfully on {DEVICE}!")
except Exception as e:
    logger.warning(f"Failed to load Batched Whisper on {DEVICE} ({e}). Falling back to CPU...")
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
    base_model = WhisperModel("small", device=DEVICE, compute_type=COMPUTE_TYPE)
    model = BatchedInferencePipeline(model=base_model)
    print("Whisper loaded on CPU fallback!")

# ──────────────────────────────────────────────────────────────────────
# Load metadata and filter target candidate videos
# ──────────────────────────────────────────────────────────────────────
if not os.path.exists(METADATA_CSV):
    raise FileNotFoundError(f"Metadata file not found at {METADATA_CSV}")

metadata_df = pd.read_csv(METADATA_CSV, dtype=str)

# Ensure Stage 3 metadata tracking columns exist
required_cols = {
    "transcription_status": "not_started",
    "transcript_path": "",
    "transcribed_at": "",
    "transcription_error": "",
    "word_count": "",
}
for col, default_val in required_cols.items():
    if col not in metadata_df.columns:
        metadata_df[col] = default_val

metadata_df["status"] = metadata_df["status"].fillna("")
metadata_df["transcription_status"] = metadata_df["transcription_status"].fillna("not_started")

# Filter candidates:
# 1. status == "audio_extracted"
# 2. diarization JSON exists
# 3. transcription_status != "done"
candidate_mask = (metadata_df["status"] == "audio_extracted") & (metadata_df["transcription_status"] != "done")

candidates_df = metadata_df[candidate_mask].copy()

# Double check diarization JSON and WAV exist
valid_candidates = []
for idx, row in candidates_df.iterrows():
    vid = str(row["video_id"])
    diar_json = os.path.join(FOLDERS["diarization"], f"{vid}.json")
    wav_path = os.path.join(FOLDERS["audio"], f"{vid}.wav")
    if os.path.exists(diar_json) and os.path.exists(wav_path):
        valid_candidates.append(idx)

logger.info(f"Total Supreme Court metadata records : {len(metadata_df)}")
logger.info(f"Queued for Supreme Court Segmented ASR: {len(valid_candidates)}")

print(f"Ready to process {len(valid_candidates)} Supreme Court videos.")

# ──────────────────────────────────────────────────────────────────────
# Segment Merging Helper
# ──────────────────────────────────────────────────────────────────────
def merge_same_speaker_segments(segments, max_gap_sec=0.5):
    """
    Merges adjacent diarization segments belonging to the SAME speaker
    if the gap between current end and next start is <= max_gap_sec.
    """
    if not segments:
        return []
    merged = []
    curr = dict(segments[0])
    for nxt in segments[1:]:
        same_spk = (nxt.get("speaker") == curr.get("speaker"))
        gap = nxt.get("start", 0.0) - curr.get("end", 0.0)
        if same_spk and gap <= max_gap_sec:
            curr["end"] = max(curr["end"], nxt["end"])
        else:
            merged.append(curr)
            curr = dict(nxt)
    merged.append(curr)
    return merged

# ──────────────────────────────────────────────────────────────────────
# Segmented Transcription Execution Loop
# ──────────────────────────────────────────────────────────────────────
for counter, idx in enumerate(valid_candidates, 1):
    row = metadata_df.loc[idx]
    vid = str(row["video_id"])
    case_name = str(row.get("case_name", vid))
    wav_path = os.path.join(FOLDERS["audio"], f"{vid}.wav")
    diar_json = os.path.join(FOLDERS["diarization"], f"{vid}.json")
    partial_json = os.path.join(FOLDERS["transcripts"], f"{vid}.partial.json")
    final_json = os.path.join(FOLDERS["transcripts"], f"{vid}.json")
    final_txt = os.path.join(FOLDERS["transcripts"], f"{vid}.txt")

    logger.info(f"\n[{counter}/{len(valid_candidates)}] Processing SC Video: {vid} | {case_name[:60]}")

    try:
        # Load diarization segments
        with open(diar_json, "r", encoding="utf-8") as f:
            raw_segments = json.load(f)

        merged_segments = merge_same_speaker_segments(raw_segments, max_gap_sec=0.5)
        logger.info(f"Loaded {len(raw_segments)} raw diarization segments -> merged into {len(merged_segments)} speaker turns.")

        # Load audio data
        # High-Speed Batched GPU Transcription (100x real-time speed)
        t_asr_start = datetime.now()
        logger.info(f"Running Batched GPU ASR on full audio: {wav_path}")
        
        asr_gen, _ = model.transcribe(wav_path, batch_size=16, language="en")
        asr_segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in asr_gen if s.text.strip()]
        
        # Map transcribed ASR segments onto speaker turns
        transcribed_segments = []
        for seg in merged_segments:
            spk = seg.get("speaker", "SPEAKER_00")
            t_start = float(seg.get("start", 0.0))
            t_end = float(seg.get("end", 0.0))
            
            matching_texts = []
            for asr in asr_segments:
                mid = (asr["start"] + asr["end"]) / 2.0
                if t_start <= mid <= t_end or (max(t_start, asr["start"]) < min(t_end, asr["end"])):
                    matching_texts.append(asr["text"])
                    
            transcribed_segments.append({
                "speaker": spk,
                "start": round(t_start, 3),
                "end": round(t_end, 3),
                "text": " ".join(matching_texts)
            })

        # Write final outputs
        output_payload = {
            "video_id": vid,
            "case_name": case_name,
            "total_segments": len(transcribed_segments),
            "segments": transcribed_segments
        }

        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        with open(final_txt, "w", encoding="utf-8") as f:
            for seg in transcribed_segments:
                if seg["text"]:
                    f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}: {seg['text']}\n")

        # Cleanup partial file upon completion
        if os.path.exists(partial_json):
            try:
                os.remove(partial_json)
            except Exception:
                pass

        total_words = sum(len(s["text"].split()) for s in transcribed_segments if s["text"])

        # Update metadata CSV
        metadata_df.at[idx, "transcription_status"] = "done"
        metadata_df.at[idx, "transcript_path"] = final_json
        metadata_df.at[idx, "transcribed_at"] = datetime.now(timezone.utc).isoformat()
        metadata_df.at[idx, "transcription_error"] = ""
        metadata_df.at[idx, "word_count"] = str(total_words)
        metadata_df.to_csv(METADATA_CSV, index=False)

        logger.info(f"DONE SC: {vid} | {len(transcribed_segments)} segments | {total_words} words")

    except Exception as e:
        error_msg = repr(e)
        logger.exception(f"FAILED SC transcription for {vid}: {error_msg}")
        metadata_df.at[idx, "transcription_status"] = "failed"
        metadata_df.at[idx, "transcription_error"] = error_msg
        metadata_df.to_csv(METADATA_CSV, index=False)

done_count = len(metadata_df[metadata_df["transcription_status"] == "done"])
print(f"\nSC STAGE 3 SEGMENTED TRANSCRIPTION COMPLETE! Transcribed files: {done_count}/{len(metadata_df)}")
