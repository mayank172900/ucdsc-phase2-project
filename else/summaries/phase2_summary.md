# Phase 2 Summary

Phase 2 creates a separate experiment codebase and adds split-level artifact export without modifying the original repo or the locked baseline reproduction repo.

## Experiment Codebase

- New repo: `/Users/goodday/Downloads/pap/paper_conf_exp`

## What Was Added

- Per-split artifact export for:
  - known scores
  - unknown scores
  - predicted labels
  - true labels
  - standard logits
  - normalized `b9` logits
- Threshold summaries for each split:
  - recall-target threshold
  - best-DTACC threshold
- Operating metrics at each exported threshold:
  - recall
  - `KRR`
  - `FAR`
  - `TNR`
  - `DTACC`

## New CLI Flags

- `--export-artifacts`
- `--artifacts-root`
- `--threshold-recall-target`

## Artifact Layout

Artifacts are written under:

- `/Users/goodday/Downloads/pap/out/artifacts/<dataset>/<run_name>/split_<idx>/`

Each split export contains:

- `<tag>_arrays.npz`
- `<tag>_summary.json`

## Expected Use

- Keep `/Users/goodday/Downloads/pap/paper_conf_run` frozen for baseline reproduction.
- Use `/Users/goodday/Downloads/pap/paper_conf_exp` for theory-driven experiments and operating-point analysis.
