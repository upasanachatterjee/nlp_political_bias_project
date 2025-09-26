# dedupe_v2theme_m2_upload_always.py
# pip install datasets huggingface_hub
import os
import math
from typing import Any, Dict, List
from datasets import load_dataset, Dataset
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG (tuned for M2 Mac, 16GB RAM) ----------
repo_or_path = "dragonslayer631/bignewsalign-with-gdelt"   # if local_dataset_path is None, replace with repo id
split = "train"   # set to your split name if needed (or None)
output_dir = "./processed_dataset"    # output path (local)

# REQUIRED: set this to the target HF dataset repo id (e.g. "your-username/processed-dataset-name")
huggingface_repo = "dragonslayer631/bignewsalign-with-gdelt"

# Provide token via environment variable HF_TOKEN or HUGGINGFACE_TOKEN or rely on `huggingface-cli login`
hf_token = os.getenv("hf_token")
# Tunable performance params (safe defaults for 16GB RAM)
num_proc = 2                # safe on 16GB RAM; bump to 2 only if confident on memory
batch_size = 500            # small per-batch memory footprint
writer_batch_size = 1000    # flush every 1000 rows
# ---------------------------------------------------------

def parse_and_transform_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    - If V2Theme is None -> keep None
    - Else: parse & dedupe into list (preserve first-occurrence order)
    - If V2Tone is None -> keep None
    - Else: parse floats, take first 6, pad with math.nan to length 6
    """
    out_themes = []
    out_tones = []

    themes_list = batch.get("V2Theme", [])
    tones_list = batch.get("V2Tone", [])

    # Zip over max length to be robust if one column missing in batch
    max_len = max(len(themes_list), len(tones_list))
    # Make safe accessors
    def _get(lst, i):
        try:
            return lst[i]
        except Exception:
            return None

    for i in range(max_len):
        theme_raw = _get(themes_list, i)
        tone_raw = _get(tones_list, i)

        # --- V2Theme handling ---
        if theme_raw is None:
            # keep it None (leave empty)
            out_themes.append(None)
        else:
            s_str = str(theme_raw).strip()
            if s_str == "":
                # non-None but empty string -> produce empty list (keeps it explicit)
                out_themes.append([])
            else:
                parts = [p for p in s_str.split(";") if p and p.strip()]
                seen = set()
                result = []
                for part in parts:
                    if ',' in part:
                        theme = part.rsplit(',', 1)[0].strip()
                    else:
                        theme = part.strip()
                    if theme and theme not in seen:
                        seen.add(theme)
                        result.append(theme)
                out_themes.append(result)

        # --- V2Tone handling ---
        if tone_raw is None:
            # keep it None (leave empty)
            out_tones.append(None)
        else:
            t_str = str(tone_raw).strip()
            if t_str == "":
                # non-None but empty string -> produce list of NaNs (consistent length)
                out_tones.append([math.nan] * 6)
            else:
                parts = [p.strip() for p in t_str.split(",") if p is not None and p != ""]
                parsed = []
                for p in parts:
                    try:
                        parsed.append(float(p))
                    except Exception:
                        # skip elements that don't parse
                        continue
                first6 = parsed[:6]
                if len(first6) < 6:
                    first6.extend([math.nan] * (6 - len(first6)))
                out_tones.append(first6)

    return {"V2Theme": out_themes, "V2Tone": out_tones}

if __name__ == "__main__":
    if not huggingface_repo:
        raise ValueError("Please set huggingface_repo to your target HF dataset repo (e.g. 'your-username/processed-dataset').")

    ds = load_dataset(repo_or_path, split=split, streaming=False)
    assert isinstance(ds, (Dataset)), "Loaded dataset is not of expected type."
    print("Dataset loaded. #rows:", len(ds))

    processed = ds.map(
        parse_and_transform_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        remove_columns=None
    )

    # Sanity-check some transformed rows (first 5)
    print("Sample transformed rows (first 5):")
    def sample_rows(dset, n=5):
        for i in range(min(n, len(dset))):
            row = dset[i]
            print(i, "V2Theme:", row.get("V2Theme"), "V2Tone (first6):", row.get("V2Tone"))
    
    sample_rows(processed, 5)

    # Save to disk
    print("Saving to disk:", output_dir)
    processed.save_to_disk(output_dir)
    print(f"Saved processed dataset to: {output_dir}")

    # Always push to HF Hub
    print("Uploading processed dataset to Hugging Face Hub:", huggingface_repo)
    # prefer token from env if provided; else rely on CLI login
    token = hf_token
    try:
        processed.push_to_hub(repo_id=huggingface_repo, token=token, private=False)
        print("Upload complete.")
    except Exception as e:
        print("Failed to upload to Hugging Face Hub. Error:", e)
        