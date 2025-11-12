#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import musicbrainzngs as mb
from musicbrainzngs import NetworkError, ResponseError

from mutagen.flac import FLAC
try:
    from mutagen.mp3 import MP3
except Exception:
    MP3 = None

import yaml

__version__ = "0.1.5"

# small utils

def sec_round(x: float | int | None) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(round(float(x)))
    except Exception:
        return None

def file_sha256(p: Path) -> Optional[str]:
    if not p or not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def load_config(cfg_path: Optional[Path]) -> dict:
    default = {
        "user_agent": {
            "app": "hafizhak-music-tagger",
            "version": __version__,
            "contact": "https://github.com/hafizhak/music-tagger"
        },
        "thresholds": {
            "duration_tolerance_seconds": 3,
            "auto_apply_on_resolver_hit": True
        },
        "tagging": {
            "write_fields": [
                "MUSICBRAINZ_RELEASEID", "MUSICBRAINZ_RELEASEGROUPID", "MUSICBRAINZ_RECORDINGID",
                "ALBUM", "TITLE", "TRACKNUMBER", "TOTALTRACKS", "DATE", "LABEL", "CATALOGNUMBER",
                "MEDIA", "SOURCE"
            ]
        },
    }
    if not cfg_path:
        return default
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, dict) and k in default:
                default[k].update(v)
            else:
                default[k] = v
    except Exception:
        pass
    return default

def set_mb_useragent(cfg: dict) -> None:
    ua = cfg.get("user_agent", {})
    mb.set_useragent(ua.get("app", "hafizhak-music-tagger"),
                     ua.get("version", __version__),
                     ua.get("contact", "https://github.com/hafizhak/music-tagger"))
    mb.set_rate_limit(True)

def mb_call(fn, *args, retries=4, **kwargs):
    delay = 0.8
    last = None
    for _ in range(retries):
        try:
            return fn(*args, **kwargs)
        except NetworkError as e:
            last = e
            time.sleep(delay * (1 + 0.3 * random.random()))
            delay *= 1.6
        except ResponseError as e:
            code = None
            try:
                code = int(getattr(e, "code", 0))
            except Exception:
                pass
            if code and 500 <= code < 600:
                last = e
                time.sleep(delay); delay *= 1.6
                continue
            raise
    if last:
        raise last
    return fn(*args, **kwargs)

# Find TOC from log
# Currently only assumes log

FRAME_RATE = 75  # CD frames per second

def _decode_log(raw: bytes, debug: bool, log: logging.Logger) -> str:
    hint = None
    if raw.startswith(b"\xff\xfe"): hint = "utf-16-le"
    elif raw.startswith(b"\xfe\xff"): hint = "utf-16-be"
    elif raw.count(b"\x00") / max(1, len(raw)) > 0.15: hint = "utf-16-le"

    tried, text = set(), None
    for enc in ([hint] if hint else []) + ["utf-16", "utf-16-le", "utf-16-be", "utf-8", "cp932", "shift_jis", "cp1252", "latin-1"]:
        if not enc or enc in tried:
            continue
        tried.add(enc)
        try:
            text = raw.decode(enc)
            if debug: log.info("DEBUG: using encoding: %s", enc)
            break
        except Exception as e:
            if debug: log.info("DEBUG: decode failed for %s: %s", enc, type(e).__name__)
    if text is None:
        text = raw.decode("latin-1", errors="ignore")
        if debug: log.info("DEBUG: fell back to latin-1(errors=ignore)")
    return text

def parse_log_toc(
    log_path: Path,
    debug: bool = False,
    debug_out: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[dict]:
    log = logger or logging.getLogger(__name__)
    notes: List[str] = []
    note = (lambda m: (notes.append(m), log.info(m))) if debug else (lambda m: None)

    if not log_path or not log_path.exists():
        note("DEBUG: No log path or file missing.")
        return None

    text = _decode_log(log_path.read_bytes(), debug, log)
    # normalize pipes/spaces
    text = (text.replace("│", "|").replace("┃", "|").replace("｜", "|")
                 .replace("\t", "    ").replace("\r", "")
                 .replace("\u00A0", " ").replace("\u2007", " ")
                 .replace("\u202F", " ").replace("\u3000", " "))
    lines = text.splitlines()

    if debug:
        head = "\n".join(lines[:60])
        note("DEBUG: first 60 lines:\n" + head)

    # find the header line
    hdr = None
    for i, ln in enumerate(lines):
        if "TOC of the extracted CD" in ln:
            hdr = i
            break
    if hdr is None:
        note("DEBUG: TOC header not found.")
        if debug and debug_out: debug_out.write_text("\n".join(notes), encoding="utf-8")
        return None

    # Accept either mm:ss.ff or mm:ss:ff. Last two columns = start/end sector.
    row_re = re.compile(
        r"""^\s*\d+\s*\|\s*
            \d+:\d{2}(?:[.:]\d{2})?\s*\|\s*
            \d+:\d{2}(?:[.:]\d{2})?\s*\|\s*
            (-?\d+)\s*\|\s*(-?\d+)\s*$
        """, re.VERBOSE
    )

    # Skip the header/underline
    i = hdr + 1
    while i < len(lines) and (not lines[i].strip() or set(lines[i].strip()) <= {"-", "|"}):
        i += 1

    offsets: List[int] = []
    leadout: Optional[int] = None
    saw = False

    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("Track") and "|" not in ln:
            break
        m = row_re.match(ln)
        if m:
            start_sec = int(m.group(1))
            end_sec = int(m.group(2))
            offsets.append(start_sec)
            leadout = end_sec + 1
            saw = True
        elif saw:
            break
        i += 1

    if not offsets:
        if debug:
            near = []
            for j in range(hdr, min(len(lines), hdr + 80)):
                if lines[j].count("|") >= 4:
                    near.append(f"L{j+1}: {lines[j].replace('|','│')}")
            note("DEBUG: no rows matched row_re.")
            if near:
                note("DEBUG: nearby table-like lines that did NOT match:")
                for s in near[:12]:
                    note("  " + s)
            if debug_out: debug_out.write_text("\n".join(notes), encoding="utf-8")
        return None

    if debug:
        note(f"DEBUG: parsed {len(offsets)} TOC rows; leadout={leadout}")
        if debug_out: debug_out.write_text("\n".join(notes), encoding="utf-8")

    return {"offsets": offsets, "leadout": leadout}

def build_toc_variants_from_sectors(offsets: List[int], leadout: Optional[int]) -> List[str]:
    def mk(off, lo):
        last = len(off)
        lo2 = lo if lo is not None else off[-1] + 4 * 60 * FRAME_RATE
        return " ".join(["1", str(last), str(lo2)] + [str(x) for x in off])
    return [mk(offsets, leadout),
            mk([x + 150 for x in offsets], (leadout + 150) if leadout is not None else None)]

# audio helpers

def _tracknum_from_name(p: Path) -> int:
    m = re.match(r"^\s*(\d{1,3})[ ._-]", p.name)
    return int(m.group(1)) if m else 10_000

def list_audio(folder: Path) -> List[Path]:
    pats = ("*.flac", "*.mp3")
    seen, files = set(), []
    for pat in pats:
        for p in folder.glob(pat):
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                files.append(p)
    return sorted(files, key=_tracknum_from_name)

def read_duration(path: Path) -> Optional[int]:
    try:
        if path.suffix.lower() == ".flac":
            a = FLAC(path); return sec_round(a.info.length)
        elif path.suffix.lower() == ".mp3" and MP3 is not None:
            a = MP3(path);  return sec_round(a.info.length)
    except Exception:
        return None
    return None

# MB lookups/search

def lookup_by_toc(toc: str) -> dict | None:
    try:
        return mb_call(
            mb.get_releases_by_discid, "", toc=toc,
            includes=["recordings", "release-groups", "artist-credits", "labels"]
        )
    except (NetworkError, ResponseError):
        return None

def parse_folder_hints(folder: Path) -> dict:
    name = folder.name
    m_date = re.search(r"\[(\d{4}-\d{2}-\d{2})\]", name)
    m_cat  = re.search(r"\{([^}]+)\}", name)
    return {"date":  (m_date.group(1) if m_date else None),
            "catno": (m_cat.group(1)  if m_cat  else None)}

def search_release_fallback(hints: dict, track_count: int) -> List[dict]:
    q = []
    if hints.get("catno"): q.append(f'catno:"{hints["catno"]}"')
    if hints.get("date"):  q.append(f'date:{hints["date"]}')
    if track_count:        q.append(f"tracks:{track_count}")
    q = " AND ".join(q) if q else ""
    try:
        sr = mb_call(mb.search_releases, query=q, limit=5, strict=True)
        out = []
        for r in sr.get("release-list", [])[:5]:
            rid = r.get("id")
            if not rid: continue
            try:
                full = mb_call(
                    mb.get_release_by_id, rid,
                    includes=["recordings", "release-groups", "artist-credits", "labels"]
                )["release"]
                out.append(full)
            except (NetworkError, ResponseError):
                continue
        return out
    except (NetworkError, ResponseError):
        return []

# verify & tag

def _mb_ms_to_sec(msec) -> Optional[int]:
    if msec is None:
        return None
    try:
        return sec_round(int(msec) / 1000.0)
    except Exception:
        return None

def verify_tracks(local_files: List[Path], mb_tracks: List[dict], tol_sec: int) -> Tuple[bool, str]:
    if len(local_files) != len(mb_tracks):
        return False, "track_count_mismatch"
    for fp, t in zip(local_files, mb_tracks):
        ldur = read_duration(fp)
        if ldur is None:
            return False, f"missing_local_duration:{fp.name}"
        msec = t.get("length") or (t.get("recording", {}) or {}).get("length")
        mdur = _mb_ms_to_sec(msec)
        if mdur is None:
            return False, "missing_mb_duration"
        if abs(ldur - mdur) > tol_sec:
            return False, f"duration_drift_gt_{tol_sec}s:{fp.name}"
    return True, ""

def _select_medium_for_files(mb_release: dict, nfiles: int) -> dict:
    for m in mb_release.get("medium-list", []):
        tc = m.get("track-count")
        tracks = m.get("track-list", [])
        if tc == nfiles or len(tracks) == nfiles:
            return m
    return mb_release["medium-list"][0]

def write_tags(local_files: List[Path], mb_release: dict, write_fields: List[str], discid: Optional[str]) -> None:
    medium = _select_medium_for_files(mb_release, len(local_files))
    tracks = medium["track-list"]
    totaltracks = str(medium.get("track-count", len(tracks)))

    rel_id = mb_release.get("id", "")
    rg_id  = (mb_release.get("release-group") or {}).get("id", "")
    album  = mb_release.get("title", "")
    date   = mb_release.get("date", "")
    label = None; catno = None
    for li in mb_release.get("label-info-list", []) or []:
        lab = li.get("label") or {}
        if not label and lab.get("name"): label = lab["name"]
        if not catno and li.get("catalog-number"): catno = li["catalog-number"]

    media_format = (medium.get("format") or "")

    for idx, (f, t) in enumerate(zip(local_files, tracks), start=1):
        if f.suffix.lower() != ".flac":
            continue
        rec = t.get("recording") or {}
        title = t.get("title") or rec.get("title") or ""
        position = str(t.get("position") or t.get("number") or idx)
        rec_id = rec.get("id", "")

        audio = FLAC(f)
        mapping = {
            "MUSICBRAINZ_RELEASEID": rel_id,
            "MUSICBRAINZ_RELEASEGROUPID": rg_id,
            "MUSICBRAINZ_RECORDINGID": rec_id,
            "ALBUM": album,
            "TITLE": title,
            "TRACKNUMBER": position,
            "TOTALTRACKS": totaltracks,
            "SOURCE": "musicbrainz",
        }
        if date:         mapping["DATE"] = date
        if label:        mapping["LABEL"] = label
        if catno:        mapping["CATALOGNUMBER"] = catno
        if media_format: mapping["MEDIA"] = media_format

        for k, v in mapping.items():
            if k in write_fields and v:
                audio[k] = v
        audio.save()

# review/report

def write_review_card(folder: Path, reason: str, candidates: List[dict]) -> None:
    out = {
        "status": "review",
        "reason": reason,
        "candidates": [
            {
                "release_id": c.get("id"),
                "title": c.get("title"),
                "date": c.get("date", ""),
                "country": c.get("country", ""),
                "label_info": c.get("label-info-list", []),
            } for c in candidates[:5]
        ]
    }
    (folder / "tag_agent_review.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

def write_report(folder: Path, payload: dict) -> None:
    (folder / "tag_agent_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

# main flow

def run(folder: Path, cfg: dict, apply: bool, print_toc: bool, debug_parse: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for name in ("musicbrainzngs", "musicbrainzngs.musicbrainz", "musicbrainzngs.model"):
        lg = logging.getLogger(name); lg.setLevel(logging.WARNING); lg.propagate = False

    set_mb_useragent(cfg)

    files = list_audio(folder)
    logging.info("Found %d audio files", len(files))

    logs = [*folder.glob("*.log"), *folder.glob("*.LOG")]
    log_path = logs[0] if logs else None
    if log_path: logging.info("Using log: %s", log_path)
    log_hash = file_sha256(log_path) if log_path else None
    dbg_out = folder / "parse_debug.txt" if debug_parse else None

    if print_toc:
        if not log_path:
            print("No log found."); return
        toc = parse_log_toc(log_path, debug=True, debug_out=(folder / "parse_debug.txt"), logger=logging.getLogger())
        if not toc:
            print("Log found, but TOC could not be parsed. See parse_debug.txt"); return
        print(f"Log: {log_path}")
        print(f"SHA256: {log_hash}")
        for s in build_toc_variants_from_sectors(toc["offsets"], toc["leadout"]):
            print("TOC:", s)
        return

    releases: List[dict] = []
    toc_used: Optional[str] = None
    discid_used: Optional[str] = None

    if log_path:
        toc = parse_log_toc(log_path, debug=debug_parse, debug_out=dbg_out, logger=logging.getLogger())
        if toc:
            logging.info("Parsed TOC: offsets=%s leadout=%s", toc["offsets"], toc["leadout"])
            for s in build_toc_variants_from_sectors(toc["offsets"], toc["leadout"]):
                logging.info("Trying TOC: %s", s)
                res = lookup_by_toc(s)
                if not res:
                    continue
                disc = res.get("disc") or {}
                discid = disc.get("id")
                if "disc" in res and "release-list" in res["disc"]:
                    releases = res["disc"]["release-list"]
                elif "release-list" in res:
                    releases = res["release-list"]
                if releases:
                    toc_used = s
                    discid_used = discid
                    if discid_used:
                        logging.info("Matched DiscID: %s", discid_used)
                    break
        else:
            logging.info("Log found but TOC not parsed; %s", "see parse_debug.txt for details." if debug_parse else "will fallback to search.")
    else:
        logging.info("No log detected; will fallback to search.")

    if not releases:
        hints = parse_folder_hints(folder)
        logging.info("Trying fallback search with hints: %s", hints)
        releases = search_release_fallback(hints, track_count=len(files))

    if not releases:
        write_review_card(folder, "no_release_for_toc_or_search", [])
        write_report(folder, {
            "status": "review",
            "reason": "no_release_for_toc_or_search",
            "files": [str(f) for f in files],
            "log": {"path": str(log_path) if log_path else None, "sha256": log_hash},
            "disc": {"discid": discid_used, "toc_used": toc_used}
        })
        print("Review needed: could not resolve release by TOC or search")
        return

    rel = releases[0]
    medium = _select_medium_for_files(rel, len(files))
    tracks = medium["track-list"]

    ok, why = verify_tracks(files, tracks, cfg["thresholds"]["duration_tolerance_seconds"])
    if not ok:
        write_review_card(folder, why, releases[:5])
        write_report(folder, {
            "status": "review",
            "reason": why,
            "candidate_release": rel.get("id"),
            "log": {"path": str(log_path) if log_path else None, "sha256": log_hash},
            "disc": {"discid": discid_used, "toc_used": toc_used}
        })
        print(f"Review needed: {why}")
        return

    if apply and cfg["thresholds"]["auto_apply_on_resolver_hit"]:
        write_tags(files, rel, cfg["tagging"]["write_fields"], discid_used)
        write_report(folder, {
            "status": "applied",
            "release_id": rel.get("id"),
            "files": [str(f) for f in files],
            "log": {"path": str(log_path) if log_path else None, "sha256": log_hash},
            "disc": {"discid": discid_used, "toc_used": toc_used}
        })
        print(f"Applied tags for release {rel.get('id')}")
    else:
        write_report(folder, {
            "status": "dry-run-ok",
            "release_id": rel.get("id"),
            "files": [str(f) for f in files],
            "log": {"path": str(log_path) if log_path else None, "sha256": log_hash},
            "disc": {"discid": discid_used, "toc_used": toc_used}
        })
        print(f"Dry-run OK: would apply tags for release {rel.get('id')}")

def main():
    ap = argparse.ArgumentParser(description="MusicBrainz Tagging Agent")
    ap.add_argument("folder", type=str, help="Album folder containing per-track files")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config")
    ap.add_argument("--apply", action="store_true", help="Actually write tags (default is dry-run)")
    ap.add_argument("--print-toc", action="store_true",
                    help="Parse log, print TOC variants & SHA-256, then exit")
    ap.add_argument("--debug-parse", action="store_true",
                    help="Verbose TOC parsing diagnostics (parse_debug.txt)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    run(Path(args.folder), cfg, args.apply, args.print_toc, args.debug_parse)

if __name__ == "__main__":
    main()
