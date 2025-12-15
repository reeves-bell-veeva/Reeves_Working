# file_mapper_2025.py
# Title-first mapping with date guard, folder normalization, folder-blocked fuzzy passes,
# one-to-one enforcement, expected folder, and adjacent-duplicate folder collapse for matching.

import re
import numpy as np
import pandas as pd
from datetime import datetime
from rapidfuzz import fuzz, process, utils

# -----------------------------
# Config
# -----------------------------
MAIN_CSV = "csws2025_main.csv"
LISTING_CSV = "csws2025_listing.csv"

OUT_MATCHES   = "csws2025_matches.csv"
OUT_UNMATCHED = "csws2025_unmatched.csv"

# -----------------------------
# Helpers
# -----------------------------
def is_attachment_path(p: str) -> bool:
    """Ignore staging items that are known attachments: '%/_a/%' or '%_a.zip'."""
    return isinstance(p, str) and ("/_a/" in p or p.endswith("_a.zip"))

def _collapse_ws_and_dashes(s: str) -> str:
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

def normalize_title(name: str) -> str:
    """
    Normalize titles. Do not strip after '.' so things like V2.0 are preserved.
    """
    if pd.isna(name):
        return ""
    s = str(name).lower()
    s = _collapse_ws_and_dashes(s)
    return s

def normalize_filename(name: str) -> str:
    """
    Normalize filenames by removing only the final extension (text after the last '.'),
    so '...V2.0_24Apr2023.pdf' -> '...V2.0_24Apr2023'. This keeps version numbers intact.
    """
    if pd.isna(name):
        return ""
    s = str(name).lower()
    # remove only the final extension if it exists
    if "." in s:
        s = s.rsplit(".", 1)[0]
    s = _collapse_ws_and_dashes(s)
    return s

def normalize_folder(name: str) -> str:
    """
    Normalize a folder segment for matching:
    - lowercase
    - remove apostrophes (straight and curly)
    - strip non a-z0-9 _/-
    - unify spaces and dashes to '_'
    """
    if not isinstance(name, str):
        return ""
    s = name.lower()
    s = re.sub(r"[’']", "", s)
    s = re.sub(r"[^a-z0-9 _/\-]", "", s)
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

def collapse_adjacent_duplicate_folders(path: str) -> str:
    """
    Remove only back-to-back duplicate folder names (normalized comparison).
    Keeps original casing of the first of the pair.
    Example:
      '.../09 Third parties/09 Third parties/09 03 General/...'
      -> '.../09 Third parties/09 03 General/...'
    """
    if not isinstance(path, str) or not path.strip():
        return path
    segs = [p for p in path.split("/") if p.strip()]
    if len(segs) < 2:
        return path

    kept = []
    last_norm = None
    for seg in segs:
        curr_norm = normalize_folder(seg)
        if curr_norm != last_norm:
            kept.append(seg)
            last_norm = curr_norm
        # else: skip exact adjacent duplicate (by normalized comparison)
    return "/".join(kept)

def normalize_full_path(path: str) -> str:
    """Normalize every segment in a path for robust folder contains() checks."""
    if not isinstance(path, str) or not path.strip():
        return ""
    segs = [normalize_folder(p) for p in path.split("/") if p.strip()]
    return "/".join(segs)

def normalize_full_path_with_adjdup(path: str) -> str:
    """
    Collapse adjacent duplicate folders, then normalize the full path for matching.
    """
    collapsed = collapse_adjacent_duplicate_folders(path)
    return normalize_full_path(collapsed)

MONTHS = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12
}

def to_iso(y,m,d):
    try:
        return datetime(int(y), int(m), int(d)).strftime("%Y-%m-%d")
    except Exception:
        return None

def parse_date_token(text: str):
    """
    Extract a date token from text and normalize to YYYY-MM-DD.
    Supports: 22Sep2021, 22Sept2021, 2021-09-22, 2021_09_22, 09-22-2021, 22-09-21, etc.
    Returns first match or None.
    """
    if not isinstance(text, str) or not text:
        return None
    s = text.lower()

    # 22Sep2021 / 22Sept2021 / 22Sep21
    m = re.search(r"\b(\d{1,2})(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(\d{2,4})\b", s)
    if m:
        d = int(m.group(1)); mon = MONTHS[m.group(2)]; y = int(m.group(3))
        if y < 100: y += 2000
        return to_iso(y, mon, d)

    # 2021-09-22 / 2021_09_22 / 2021.09.22 / 2021/09/22
    m = re.search(r"\b(20\d{2}|19\d{2})[._/-](\d{1,2})[._/-](\d{1,2})\b", s)
    if m:
        y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return to_iso(y, mon, d)

    # 09-22-2021 (M-D-Y)
    m = re.search(r"\b(\d{1,2})[._/-](\d{1,2})[._/-](19|20)\d{2}\b", s)
    if m:
        mon, d, y_prefix = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = int(re.search(r"(19|20)\d{2}", m.group(0)).group(0))
        return to_iso(y, mon, d)

    # 22-09-21 or 22_09_2021 (D-M-Y; 2-digit year -> 2000+)
    m = re.search(r"\b(\d{1,2})[._/-](\d{1,2})[._/-](\d{2,4})\b", s)
    if m:
        d, mon, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100: y += 2000
        return to_iso(y, mon, d)

    return None

def enforce_date_guard(meta_date, file_date) -> bool:
    """
    Return True if row should be kept.
    If both dates are present and differ, return False.
    """
    if pd.isna(meta_date) or pd.isna(file_date) or meta_date is None or file_date is None:
        return True
    return str(meta_date) == str(file_date)

def detect_project_root(full_path: str) -> str:
    """
    Detect project like 'CSWS2025' from the first segment of full_path.
    Fallback to 'CSWS2025' if not found.
    """
    default = "CSWS2025"
    if not isinstance(full_path, str) or not full_path.strip():
        return default
    first = str(full_path).split("/", 1)[0]
    m = re.search(r"(CSWS\d{4})", first, re.IGNORECASE)
    return m.group(1).upper() if m else default

def expected_staging_folder(full_path: str) -> str:
    """
    Build an expected staging path that preserves the original folder names exactly.
    Only the Wave4 root and detected project (e.g., CSWS2025) are injected.
    """
    if not isinstance(full_path, str) or not full_path.strip():
        return None
    parts = [p for p in full_path.split("/") if p.strip()]
    remainder = parts[1:] if len(parts) > 1 else []
    project_root = detect_project_root(full_path)
    return f"/Wave4/{project_root}/" + "/".join(remainder) if remainder else f"/Wave4/{project_root}"

def last_folder_token(full_path: str) -> str:
    """Normalized last folder name from full_path (for folder-blocked fuzzy)."""
    if not isinstance(full_path, str) or not full_path.strip():
        return None
    parts = [p for p in full_path.split("/") if p.strip()]
    if not parts:
        return None
    return normalize_folder(parts[-1])

# -----------------------------
# Load data
# -----------------------------
main_df = pd.read_csv(MAIN_CSV)
listing_df = pd.read_csv(LISTING_CSV)

# Candidate files (exclude attachments) + normalized columns
files_df = listing_df[listing_df["kind"] == "file"].copy()
files_df = files_df[~files_df["path"].map(is_attachment_path)].copy()
files_df["filename_file"] = files_df["path"].str.split("/").str[-1]
files_df["norm_filename"] = files_df["filename_file"].map(normalize_filename)
files_df["date_key"] = files_df["filename_file"].map(parse_date_token)

# Use adjacent-duplicate collapse before normalizing full path (used for folder-blocked matching)
files_df["norm_path"] = files_df["path"].map(normalize_full_path_with_adjdup)

# Metadata normalized columns (title-first)
main = main_df.copy()
main["norm_title"] = main["title"].map(normalize_title)
main["date_key"] = main["title"].map(parse_date_token)
main["expected_folder"] = main["full_path"].map(expected_staging_folder)
main["folder_token"] = main["full_path"].map(last_folder_token)

total = len(main)

# -----------------------------
# Stage A: Title-first (Exact + Fuzzy≥90) with date guard
# -----------------------------
# A1) Exact normalized title ↔ normalized filename
pass1 = main.merge(
    files_df[["path", "filename_file", "norm_filename", "date_key"]],
    left_on="norm_title",
    right_on="norm_filename",
    how="inner",
)
pass1 = pass1[
    pass1.apply(lambda r: enforce_date_guard(r["date_key_x"], r["date_key_y"]), axis=1)
]
pass1["match_type"] = "exact_norm_title"
pass1["intra_score"] = pass1.apply(
    lambda r: fuzz.QRatio(utils.default_process(str(r["title"])),
                          utils.default_process(str(r["filename_file"]))),
    axis=1,
)

matched_ids = set(pass1["document_id"])

# A2) Fuzzy ≥90 on title vs filename (date guard)
remaining = main[~main["document_id"].isin(matched_ids)].copy()
idx = files_df[["norm_filename", "path", "filename_file", "date_key"]].drop_duplicates().reset_index(drop=True)

pass2_rows = []
if not remaining.empty and not idx.empty:
    mat = process.cdist(
        remaining["norm_title"].fillna("").tolist(),
        idx["norm_filename"].tolist(),
        scorer=fuzz.token_sort_ratio,
    )
    best_idx = mat.argmax(axis=1)
    best_scores = mat[np.arange(mat.shape[0]), best_idx]
    for (doc_id, title, m_date), j, score in zip(
        remaining[["document_id", "title", "date_key"]].itertuples(index=False),
        best_idx, best_scores
    ):
        if score < 90:
            continue
        mr = idx.iloc[int(j)]
        if not enforce_date_guard(m_date, mr["date_key"]):
            continue
        pass2_rows.append({
            "document_id": doc_id,
            "title": title,
            "path": mr["path"],
            "filename_file": mr["filename_file"],
            "match_type": "fuzzy_title_90",
            "intra_score": float(score)
        })
pass2 = pd.DataFrame(pass2_rows)

stageA = pd.concat(
    [
        pass1[["document_id", "title", "path", "filename_file", "match_type", "intra_score"]],
        pass2[["document_id", "title", "path", "filename_file", "match_type", "intra_score"]],
    ],
    ignore_index=True,
)

# One-to-one within Stage A
stageA_best = stageA.sort_values(["match_type","intra_score"], ascending=[True, False]).drop_duplicates(["document_id"])
used_paths = set(); keep = []
for _, r in stageA_best.iterrows():
    if r["path"] in used_paths:
        continue
    used_paths.add(r["path"])
    keep.append(r)
stageA_final = pd.DataFrame(keep)
stageA_ids = set(stageA_final["document_id"])

# -----------------------------
# Stage B: Date-aligned + Folder-blocked fuzzy (uses normalized folder tokens)
# -----------------------------
remainingB = main[~main["document_id"].isin(stageA_ids)].copy()

# B1) Date-aligned fuzzy ≥85
rem_with_date = remainingB[remainingB["date_key"].notna()].copy()
files_with_date = files_df[files_df["date_key"].notna()].copy()
b1_rows = []
if not rem_with_date.empty and not files_with_date.empty:
    for date_val, rem_block in rem_with_date.groupby("date_key"):
        cand_block = files_with_date[files_with_date["date_key"] == date_val]
        if cand_block.empty:
            continue
        mat = process.cdist(
            rem_block["norm_title"].fillna("").tolist(),
            cand_block["norm_filename"].tolist(),
            scorer=fuzz.token_sort_ratio,
        )
        best_idx = mat.argmax(axis=1)
        best_scores = mat[np.arange(mat.shape[0]), best_idx]
        for (doc_id, title), j, score in zip(rem_block[["document_id","title"]].itertuples(index=False), best_idx, best_scores):
            if score < 85:
                continue
            mr = cand_block.iloc[int(j)]
            b1_rows.append({
                "document_id": doc_id,
                "title": title,
                "path": mr["path"],
                "filename_file": mr["filename_file"],
                "match_type": "date_align_title85",
                "intra_score": float(score)
            })
b1 = pd.DataFrame(b1_rows)

# B2) Folder-blocked fuzzy ≥90 (normalized folder tokens; norm path includes adjacent-dup collapse)
remainingC = remainingB[~remainingB["document_id"].isin(set(b1["document_id"]))].copy()
folder_map = {}
for ftok in remainingC["folder_token"].dropna().unique():
    ft_norm = normalize_folder(ftok)
    cand = files_df[files_df["norm_path"].str.contains(ft_norm, na=False)][["norm_filename","path","filename_file"]].drop_duplicates()
    folder_map[ft_norm] = cand

def folder_blocked_fuzzy(df, low, high, label):
    out = []
    for _, row in df.iterrows():
        ft = normalize_folder(row["folder_token"]) if pd.notna(row["folder_token"]) else None
        if not ft or ft not in folder_map:
            continue
        cands = folder_map[ft]
        if cands.empty:
            continue
        mat = process.cdist([row["norm_title"]], cands["norm_filename"].tolist(), scorer=fuzz.token_sort_ratio)
        score = float(mat.max())
        idx_max = int(mat.argmax())
        if low <= score < high:
            mr = cands.iloc[idx_max]
            out.append({
                "document_id": row["document_id"],
                "title": row["title"],
                "path": mr["path"],
                "filename_file": mr["filename_file"],
                "match_type": label,
                "intra_score": score
            })
    return pd.DataFrame(out)

b2 = folder_blocked_fuzzy(remainingC, 90, 101, "folder_block_fuzzy_90")
b3 = folder_blocked_fuzzy(remainingC[~remainingC["document_id"].isin(set(b2["document_id"]))], 80, 90, "folder_block_fuzzy_80_89_needs_review")

# -----------------------------
# Final combine + one-to-one
# -----------------------------
all_candidates = pd.concat([stageA_final, b1, b2, b3], ignore_index=True, sort=False)

priority = {
    "exact_norm_title": 1,
    "fuzzy_title_90": 2,
    "date_align_title85": 3,
    "folder_block_fuzzy_90": 4,
    "folder_block_fuzzy_80_89_needs_review": 5
}
all_candidates["priority"] = all_candidates["match_type"].map(priority).fillna(99).astype(int)

best_per_doc = all_candidates.sort_values(["priority","intra_score"], ascending=[True, False]).drop_duplicates(["document_id"])

used_paths = set(); final_rows = []
for _, r in best_per_doc.iterrows():
    if r["path"] in used_paths:
        continue
    used_paths.add(r["path"])
    final_rows.append(r)
final_matches = pd.DataFrame(final_rows)

# -----------------------------
# Unmatched (include metadata filename + expected folder)
# -----------------------------
matched_n = final_matches["document_id"].nunique()
unmatched = main[~main["document_id"].isin(set(final_matches["document_id"]))][
    ["document_id", "title", "filename", "full_path", "expected_folder"]
].copy()

# -----------------------------
# Save CSVs
# -----------------------------
final_cols = ["document_id","title","path","filename_file","match_type","intra_score"]
final_matches[final_cols].to_csv(OUT_MATCHES, index=False)
unmatched.to_csv(OUT_UNMATCHED, index=False)

print({
    "totals": {"total": int(total), "matched": int(matched_n), "unmatched": int(total - matched_n)},
    "outputs": {"matches_csv": OUT_MATCHES, "unmatched_csv": OUT_UNMATCHED}
})
