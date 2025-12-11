# file_mapper_csws2010.py

import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils

# -----------------------------
# Config (override via CLI if you want—kept simple for CSWS2010)
# -----------------------------
MAIN_CSV = "CSWS2010_Main.csv"
LISTING_CSV = "CSWS2010_Listing.csv"

OUT_MATCHES                = "csws2010_matches.csv"
OUT_UNMATCHED_MAIN         = "csws2010_unmatched_main.csv"
OUT_UNMATCHED_LISTING      = "csws2010_listing_unmatched_after_mapping.csv"
OUT_EXCLUDED_LISTING       = "csws2010_listing_excluded_ignored.csv"

# -----------------------------
# Helpers: unicode + normalization
# -----------------------------
_WHITESPACE_RX = re.compile(r"\s+", flags=re.UNICODE)

def _normalize_unicode_spaces(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    return _WHITESPACE_RX.sub(" ", s)

def _collapse_ws_and_dashes(s: str) -> str:
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

def normalize_title(name: str) -> str:
    """Titles: keep dots (e.g., V1.0), normalize spaces and separators."""
    if pd.isna(name):
        return ""
    s = _normalize_unicode_spaces(str(name).lower())
    return _collapse_ws_and_dashes(s)

def normalize_filename(name: str) -> str:
    """Filenames: remove ONLY the final extension; keep internal dots (e.g., V1.0)."""
    if pd.isna(name):
        return ""
    s = _normalize_unicode_spaces(str(name).lower())
    if "." in s:
        s = s.rsplit(".", 1)[0]
    return _collapse_ws_and_dashes(s)

def normalize_folder(name: str) -> str:
    """Folder segment normalization (case-insensitive, apostrophes removed)."""
    if not isinstance(name, str):
        return ""
    s = _normalize_unicode_spaces(name).lower()
    s = re.sub(r"[’']", "", s)
    s = re.sub(r"[^a-z0-9 _/\-]", "", s)
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

def is_attachment_path(p: str) -> bool:
    """
    Exclude files in folders ending with '_A' (i.e., contains '_A/')
    or zips ending in '_A.zip'. Case-insensitive; handles backslashes.
    """
    if not isinstance(p, str):
        return False
    p = p.replace("\\", "/")
    pl = p.lower()
    return ("_a/" in pl) or pl.endswith("_a.zip")

def has_adjacent_duplicate_folders(path: str) -> bool:
    """
    True if path contains adjacent duplicate folder names (case-insensitive, normalized).
    Example: .../01 Trial Management/01 Trial Management/...
    """
    if not isinstance(path, str) or not path.strip():
        return False
    p = path.replace("\\", "/")
    segs = [s for s in p.split("/") if s.strip()]
    if len(segs) < 2:
        return False
    prev = None
    for s in segs:
        cur = normalize_folder(s)
        if prev is not None and cur == prev:
            return True
        prev = cur
    return False

def normalize_full_path(path: str) -> str:
    """Normalize every segment in a path for robust folder contains() checks."""
    if not isinstance(path, str) or not path.strip():
        return ""
    segs = [normalize_folder(p) for p in path.replace("\\", "/").split("/") if p.strip()]
    return "/".join(segs)

# -----------------------------
# Date helpers + guards
# -----------------------------
MONTHS = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12}

def to_iso(y,m,d):
    try:
        return datetime(int(y), int(m), int(d)).strftime("%Y-%m-%d")
    except Exception:
        return None

def parse_date_token(text: str):
    """Extract a date token and normalize to YYYY-MM-DD; supports several formats."""
    if not isinstance(text, str) or not text:
        return None
    s = str(text).lower()

    m = re.search(r"\b(\d{1,2})(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(\d{2,4})\b", s)
    if m:
        d = int(m.group(1)); mon = MONTHS[m.group(2)]; y = int(m.group(3))
        if y < 100: y += 2000
        return to_iso(y, mon, d)

    m = re.search(r"\b(20\d{2}|19\d{2})[._/-](\d{1,2})[._/-](\d{1,2})\b", s)
    if m:
        y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return to_iso(y, mon, d)

    m = re.search(r"\b(\d{1,2})[._/-](\\d{1,2})[._/-](19|20)\d{2}\b", s)  # M-D-YYYY
    if m:
        mon, d, _ = int(m.group(1)), int(m.group(2)), m.group(3)
        y = int(re.search(r"(19|20)\d{2}", m.group(0)).group(0))
        return to_iso(y, mon, d)

    m = re.search(r"\b(\d{1,2})[._/-](\d{1,2})[._/-](\d{2,4})\b", s)  # D-M-YY/YY
    if m:
        d, mon, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100: y += 2000
        return to_iso(y, mon, d)

    return None

def enforce_date_guard(meta_date, file_date) -> bool:
    """Return True unless both dates are present and differ."""
    if pd.isna(meta_date) or pd.isna(file_date) or meta_date is None or file_date is None:
        return True
    return str(meta_date) == str(file_date)

# -----------------------------
# Expected folder helpers
# -----------------------------
def detect_project_root(full_path: str) -> str:
    default = "CSWS2010"
    if not isinstance(full_path, str) or not full_path.strip():
        return default
    first = str(full_path).split("/", 1)[0]
    m = re.search(r"(CSWS\d{4})", first, re.IGNORECASE)
    return m.group(1).upper() if m else default

def expected_staging_folder(full_path: str) -> str:
    if not isinstance(full_path, str) or not full_path.strip():
        return None
    parts = [p for p in full_path.split("/") if p.strip()]
    remainder = parts[1:] if len(parts) > 1 else []
    project_root = detect_project_root(full_path)
    return f"/Wave4/{project_root}/" + "/".join(remainder) if remainder else f"/Wave4/{project_root}"

def last_folder_token(full_path: str) -> str:
    if not isinstance(full_path, str) or not full_path.strip():
        return None
    parts = [p for p in full_path.split("/") if p.strip()]
    if not parts:
        return None
    return normalize_folder(parts[-1])

# -----------------------------
# Load inputs
# -----------------------------
main_df = pd.read_csv(MAIN_CSV)
listing_df = pd.read_csv(LISTING_CSV)

# -----------------------------
# Build candidate pool + keep an explicit 'excluded' report
# -----------------------------
mask_file  = listing_df["kind"].astype(str).str.lower().eq("file")
mask_att   = listing_df["path"].map(is_attachment_path)
mask_dup   = listing_df["path"].map(has_adjacent_duplicate_folders)

files_df_candidates = listing_df[mask_file & ~mask_att & ~mask_dup].copy()
listing_excluded    = listing_df[mask_file & (mask_att | mask_dup)].copy()

# tag excluded rows with reason(s)
reasons = []
for idx in listing_excluded.index:
    r = []
    if bool(mask_att.loc[idx]): r.append("attachment_folder(_A/ or _A.zip)")
    if bool(mask_dup.loc[idx]): r.append("adjacent_duplicate_folder")
    reasons.append("; ".join(r) if r else "")
listing_excluded["excluded_reason"] = reasons

# -----------------------------
# Candidate files: normalized columns
# -----------------------------
files_df = files_df_candidates.copy()
files_df["filename_file"] = files_df["path"].astype(str).str.split("/").str[-1]
files_df["norm_filename"] = files_df["filename_file"].map(normalize_filename)
files_df["date_key"] = files_df["filename_file"].map(parse_date_token)
files_df["norm_path"] = files_df["path"].map(normalize_full_path)

# -----------------------------
# Metadata normalized columns (title-first)
# -----------------------------
main = main_df.copy()
main["norm_title"] = main["title"].map(normalize_title)
main["date_key"] = main["title"].map(parse_date_token)
main["expected_folder"] = main["full_path"].map(expected_staging_folder)
main["folder_token"] = main["full_path"].map(last_folder_token)

total = len(main)

# -----------------------------
# Stage A: Title-first (Exact + Fuzzy≥90) with date guard
# -----------------------------
# A1) Exact
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
    lambda r: fuzz.QRatio(
        utils.default_process(str(r["title"])),
        utils.default_process(str(r["filename_file"]))
    ),
    axis=1,
)

matched_ids = set(pass1["document_id"])

# A2) Fuzzy ≥90
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
# Stage B: Date-aligned + Folder-blocked fuzzy
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

# B2/B3) Folder-blocked fuzzy
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
    "folder_block_fuzzy_80_89_needs_review": 5,
}
all_candidates["priority"] = all_candidates["match_type"].map(priority).fillna(99).astype(int)
best_per_doc = all_candidates.sort_values(["priority","intra_score"], ascending=[True, False]).drop_duplicates(["document_id"])

used_paths = set(); final_rows = []
for _, r in best_per_doc.iterrows():
    if r["path"] in used_paths:
        continue
    used_paths.add(r["path"])
    final_rows.append(r)
final_matches = Pd = pd.DataFrame(final_rows)

# -----------------------------
# Unmatched MAIN
# -----------------------------
matched_n = final_matches["document_id"].nunique()
unmatched_main = main[~main["document_id"].isin(set(final_matches["document_id"]))][
    ["document_id", "title", "filename", "full_path", "expected_folder"]
].copy()

# -----------------------------
# Unmatched LISTING (exclude attachments & dup-folder paths via candidates)
# -----------------------------
matched_paths = set(final_matches["path"].astype(str))
unmatched_listing = files_df[~files_df["path"].astype(str).isin(matched_paths)].copy()
unmatched_listing = unmatched_listing.drop_duplicates(subset=["path"])
cols_unmatched_listing = [c for c in ["path","filename_file","kind","norm_path"] if c in unmatched_listing.columns]
if not cols_unmatched_listing:
    cols_unmatched_listing = ["path"]

# -----------------------------
# Write outputs (format intra_score to 1 decimal ONLY in CSV)
# -----------------------------
final_cols = ["document_id","title","path","filename_file","match_type","intra_score"]
out_matches_df = final_matches[final_cols].copy()
out_matches_df["intra_score"] = out_matches_df["intra_score"].map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "")

out_matches_df.to_csv(OUT_MATCHES, index=False)
unmatched_main.to_csv(OUT_UNMATCHED_MAIN, index=False)
unmatched_listing[cols_unmatched_listing].to_csv(OUT_UNMATCHED_LISTING, index=False)

cols_ex = [c for c in ["path","kind","excluded_reason"] if c in listing_excluded.columns]
if "excluded_reason" not in cols_ex:
    cols_ex.append("excluded_reason")
listing_excluded[cols_ex].drop_duplicates(subset=["path"]).to_csv(OUT_EXCLUDED_LISTING, index=False)

print({
    "totals": {
        "main_rows": int(total),
        "matched_docs": int(matched_n),
        "unmatched_docs": int(total - matched_n),
        "listing_candidates": int(len(files_df)),
        "listing_excluded": int(len(listing_excluded)),
        "listing_unmatched": int(len(unmatched_listing)),
    },
    "outputs": {
        "matches_csv": OUT_MATCHES,
        "unmatched_main_csv": OUT_UNMATCHED_MAIN,
        "unmatched_listing_csv": OUT_UNMATCHED_LISTING,
        "excluded_listing_csv": OUT_EXCLUDED_LISTING
    }
})
