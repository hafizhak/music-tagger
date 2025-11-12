# Music Tagging Agent

Tool that tags prepared album folders using MusicBrainz. A practice project, cannot guarantee quality nor maintainence.

- Looks for DiscID from (EAC) log when possible
- Conservative fallback search (catno/date/tracks)
- Verify track count + per-track duration (±3s)
- Dry-run default; writes only on `--apply`
- Outputs review & audit JSONs

## Usage
```bash
python -m venv .venv && .\.venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Dry-run (no writes)
python main.py "C:\path\to\folder"

# Apply writes after verification
python main.py "C:\path\to\folder" --apply
```

## Config
See `config.yaml` for thresholds and user-agent.