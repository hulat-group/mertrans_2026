#runs
# 💻 Source Code

This folder contains a selection of **non-core orchestration scripts** used during the development and execution of the HULAT2-UC3M MER-TRANS 2026 submissions.

These scripts illustrate **how the system was invoked, evaluated, and packaged** for submission. They do **not** include the core multi-agent implementation (LangGraph nodes, prompts, routing logic, or quality-signal thresholds), which remains part of the internal HULAT-UC3M infrastructure and is not released in this repository.

---

## ⚠️ Important note on executability

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
Generates the official MER-TRANS 2026 submission files (RUN1 / RUN2). Reads the official test CSV, runs each instance through the pipeline, writes the official format output (`RUN{n}.csv`) and a detailed internal metrics CSV, and packages everything into the submission ZIP.

Includes:
- Resume.
- Checkpoint logic to recover from interrupted runs.

```bash
python submission.py --run RUN2 --lexical on
```

---

## 🧩 What is *not* included

- The `src/graph.py` and `src/signals.py` modules (and all underlying node implementations: lexical agent, pre-analysis, generators, candidate evaluator, router, merger-editor, final evaluator).
- Prompt templates used by the generators and the merger-editor.
- ECA routing rules, quality-signal thresholds, and retry logic.
- The HULAT-UC3M glossary and lexical-resource files.
- Any dataset files (official MER-TRANS data, trial data, or internally used corpora).

The extended technical specification (signal taxonomy, ECA rules, validation criteria) is archived separately in the Zenodo reproducibility record referenced in the main paper.

---

## 📬 Contact

For research collaboration inquiries regarding the full system implementation, please contact the HULAT-UC3M group: [https://hulat.inf.uc3m.es/](https://hulat.inf.uc3m.es/)
