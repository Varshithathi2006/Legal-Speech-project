import os
import sys
import json
import logging
import pandas as pd
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline

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
LOG_FILE = os.path.join(FOLDERS["logs"], "stage3_transcription.log")

print("Initializing High-Speed Batched GPU Transcription...", flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
print(f"TRANSCRIPTION HARDWARE ACCELERATOR: {DEVICE.upper()} (Compute Type: {COMPUTE_TYPE})", flush=True)
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}", flush=True)

WHISPER_MODEL_SIZE = "tiny"
print(f"Loading faster-whisper '{WHISPER_MODEL_SIZE}' model with Batched GPU Pipeline...", flush=True)
base_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
batched_model = BatchedInferencePipeline(model=base_model)
print("Batched Whisper GPU ASR Model loaded successfully!", flush=True)

all_wavs = sorted([f for f in os.listdir(FOLDERS["audio"]) if f.endswith(".wav")])
queued_files = []

for wav_name in all_wavs:
    vid = wav_name.replace(".wav", "")
    json_out = os.path.join(FOLDERS["transcripts"], f"{vid}.json")
    if not os.path.exists(json_out):
        audio_p = os.path.join(FOLDERS["audio"], wav_name)
        queued_files.append((vid, audio_p))

print(f"Total WAV Audio Files        : {len(all_wavs)}", flush=True)
print(f"Queued for GPU Transcription : {len(queued_files)}", flush=True)

def match_speaker(start_sec, end_sec, diar_segments):
    if not diar_segments:
        return "SPEAKER_00"
    mid = (start_sec + end_sec) / 2.0
    for seg in diar_segments:
        if seg["start"] <= mid <= seg["end"]:
            return seg["speaker"]
    best = min(diar_segments, key=lambda s: abs(s["start"] - start_sec))
    return best["speaker"]

def assign_role_labels(speaker_list, text_snippets):
    role_map = {}
    for spk in set(speaker_list):
        spk_texts = " ".join([t for s, t in zip(speaker_list, text_snippets) if s == spk]).lower()
        if any(w in spk_texts for w in ["my lord", "your lordship", "may it please", "learned counsel", "counsel for", "petitioner", "respondent"]):
            role_map[spk] = "Counsel / Advocate"
        elif any(w in spk_texts for w in ["court", "bench", "question", "submit", "statute", "order", "dismissed"]):
            role_map[spk] = "Presiding Judge / Bench"
        else:
            role_map[spk] = "Speaker / Court Participant"
    return role_map

metadata_df = pd.read_csv(METADATA_CSV, dtype=str) if os.path.exists(METADATA_CSV) else None

completed_count = 0
for vid, audio_path in queued_files:
    completed_count += 1
    print(f"[{completed_count}/{len(queued_files)}] Batched GPU Transcribing: {vid}", flush=True)
    diar_json = os.path.join(FOLDERS["diarization"], f"{vid}.json")
    diar_segments = []
    if os.path.exists(diar_json):
        try:
            with open(diar_json, "r", encoding="utf-8") as f:
                diar_segments = json.load(f)
        except Exception:
            pass

    try:
        segments_gen, info = batched_model.transcribe(audio_path, batch_size=16, language="en")
        segments_raw = list(segments_gen)
        
        transcript_segments = []
        full_texts = []
        speaker_sequence = []
        
        for seg in segments_raw:
            spk = match_speaker(seg.start, seg.end, diar_segments)
            text_str = seg.text.strip()
            if not text_str:
                continue
            transcript_segments.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "speaker": spk,
                "text": text_str
            })
            speaker_sequence.append(spk)
            full_texts.append(text_str)

        role_map = assign_role_labels(speaker_sequence, full_texts)
        
        for seg in transcript_segments:
            seg["role"] = role_map.get(seg["speaker"], "Speaker / Court Participant")

        json_out = os.path.join(FOLDERS["transcripts"], f"{vid}.json")
        txt_out = os.path.join(FOLDERS["transcripts"], f"{vid}.txt")

        structured_data = {
            "video_id": vid,
            "language": info.language,
            "duration_sec": round(info.duration, 2),
            "speaker_roles": role_map,
            "segments": transcript_segments
        }

        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2)

        with open(txt_out, "w", encoding="utf-8") as f:
            for seg in transcript_segments:
                f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']} ({seg['role']}): {seg['text']}\n")

        total_words = sum(len(t.split()) for t in full_texts)
        print(f"DONE [{completed_count}/{len(queued_files)}]: {vid} | {len(transcript_segments)} ASR segments | {total_words} words", flush=True)

        if metadata_df is not None and vid in metadata_df["video_id"].values:
            idx = metadata_df[metadata_df["video_id"] == vid].index[0]
            metadata_df.at[idx, "transcription_status"] = "done"
            metadata_df.at[idx, "transcript_path"] = json_out
            metadata_df.at[idx, "word_count"] = str(total_words)
            metadata_df.to_csv(METADATA_CSV, index=False)

    except Exception as e:
        print(f"FAILED transcription for {vid}: {e}", flush=True)

done_count = len([f for f in os.listdir(FOLDERS["transcripts"]) if f.endswith(".json")])
print(f"\nSTAGE 3 BATCHED GPU TRANSCRIPTION COMPLETE! Total transcribed files: {done_count}/{len(all_wavs)}", flush=True)
