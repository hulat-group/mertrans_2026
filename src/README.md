#runs
# 💻 Source Code

This folder contains a selection of **non-core orchestration scripts** used during the development and execution of the HULAT2-UC3M MER-TRANS 2026 submissions.

These scripts illustrate **how the system was invoked, evaluated, and packaged** for submission. They do **not** include the core multi-agent implementation (LangGraph nodes, prompts, routing logic, or quality-signal thresholds), which remains part of the internal HULAT-UC3M infrastructure and is not released in this repository.

---

## ⚠️ Important note

All scripts in this folder import from an internal module (`src.graph`, `src.signals`) that is **not included** in this repository:

```python
from src.graph import setup_system, run_simplification
from src.signals import normalize_signals, sari, fernandez_huerta
```

This module contains the actual multi-agent workflow described in the paper — the 10 LangGraph nodes (lexical agent, pre-analysis, parallel generators, candidate evaluator, ECA router, merger-editor, final evaluator), the prompt templates, and the internal quality-signal thresholds used for routing and retry decisions.

**These scripts are therefore not runnable.** They are shared to document the system's I/O behaviour, configuration options, and evaluation methodology not as a reproducible package. If you are interested in the full implementation for research collaboration purposes, please contact the HULAT-UC3M group.

---

## 📂 Scripts

### `submission.py`
Generates the official MER-TRANS 2026 submission files (RUN1 / RUN2). Reads the official test CSV, runs each instance through the pipeline, writes the official format output and a detailed internal metrics CSV, and packages everything into the submission ZIP.

Includes:
- Resume.
- Checkpoint logic to recover from interrupted runs.

```bash
python submission.py --run RUN2 --lexical on
```

### `simplify.py`
On-demand simplification of a single input text. Reads a `.txt` file with the format `*Texto a simplificar: ...` (and an optional `*Texto de referencia: ...` line), runs it through the pipeline, and writes a human-readable report with quality signals and, when a reference is provided, reference-based metrics (SARI, BLEU, BERTScore).

### `run_parallel_evaluation.py`
Batch evaluation script with checkpointing and optional trace logging, used to evaluate the system on different corpus beyond the official MER-TRANS test. Including the Spanish Constitution in Easy-to-Read format and an exploratory clinical-text corpus used in separate internal testing. 

### `run_trial_evaluation.py`
Batch evaluation script that runs the pipeline over the official MER-TRANS trial set.

> Note: no dataset files are distributed in this repository, only the code that references them. Some of the resources used for calibration (e.g. the Spanish Constitution in Easy-to-Read format)publicly available and documented in Appendix A of the paper.

---

## 🧩 What is *not* included

- The `src/graph.py`, `src/signals.py` and `src/all_nodes`modules.
- Prompt templates used by the generators and the merger-editor.
- ECA routing rules, quality-signal thresholds, and retry logic.
- The HULAT-UC3M glossary and lexical-resource files.
- Any dataset files.

---

## 📬 Contact

For research collaboration inquiries regarding the full system implementation, please contact the HULAT-UC3M group: [https://hulat.inf.uc3m.es/](https://hulat.inf.uc3m.es/)
