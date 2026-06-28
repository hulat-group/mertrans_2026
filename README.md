# 🧠 HULAT-UC3M @MER-TRANS 2026 Shared Task

[![MER-TRANS 2026](https://img.shields.io/badge/MER--TRANS-2026-blue)](https://lastus-taln-upf.github.io/mertrans-iberlef-2026/#overview)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()

---

## 🌐 Official Task Website
👉 [MER-TRANS 2026 Shared Task](https://lastus-taln-upf.github.io/mertrans-iberlef-2026/#overview)

---

## 👥 Team

**HULAT-UC3M (Human Language and Accessibility Technologies)**  
Universidad Carlos III de Madrid  

- Paloma Martínez Fernández  
- Lourdes Moreno López 
- Miguel Domínguez Gómez 
- Marco Antonio Sánchez Escudero

---

## 📝 Approach Summary

We participated in the Spanish track of **MER-TRANS 2026**, a shared task on multilingual Easy-to-Read generation.

Our submission focused on automatic Spanish Easy-to-Read simplification using a governed multi-agent architecture. The system was designed to generate simplified texts while controlling semantic preservation, factual consistency, readability, lexical simplicity and generation robustness.

We submitted three automatic runs:

- **RUN1: Multi-Agent Workflow**
  RUN1 used a LangGraph-based multi-agent workflow combining parallel generation, internal quality signals, Event–Condition–Action routing and controlled editing. This was the best HULAT2 run according to the official SARI score.

- **RUN2: Multi-Agent Workflow with Lexical Support**
  RUN2 used the same architecture as RUN1, but activated an additional lexical-support layer before generation.
  This layer relies on our glossary-based and lexical-resource modules to identify difficult terms and propose simpler alternatives or explanatory formulations.

- **RUN3: RigoChat-Based generate–evaluate–regenerate baseline**
  RUN3 was implemented as a linear generate–evaluate–regenerate baseline. It used RigoChat-7B-v2 with prompt engineering, post-processing and LoRA-based adaptation. The system generated an Easy-to-Read-oriented output, evaluated it using internal quality criteria and attempted regeneration when the output did not satisfy the expected constraints.

---

## 🤖 Models
The submitted systems used the following models:

- **Gemini 2.5 Flash**
  Used in the multi-agent workflow for conservative Plain Language generation and Easy-to-Read-oriented generation.

  Official model: [Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash?hl=es-419)

- **RigoChat-7B-v2**
  Used both as one of the generators in the multi-agent workflow and as the main model in the RUN3 baseline. RigoChat-7B-v2 is a Spanish-oriented language model based on Qwen2.5-7B-Instruct and adapted for Spanish queries.

  Official model: [RigoChat-7B-v2](https://huggingface.co/IIC/RigoChat-7b-v2)  

---

## 🧩 Frameworks

**LangGraph**

- Used to implement the governed multi-agent workflow, including stateful execution, specialised nodes, conditional routing, retry loops and traceable decisions.

  Official framework: [LangGraph](https://www.langchain.com/langgraph)

---

📁 Repository Contents

This repository contains the implementation associated with the HULAT2-UC3M MER-TRANS 2026 submissions.

The repository includes:

- source code for the submitted systems.
- internal evaluation and validation utilities.

---

## 📖 Citation
For citing the conference paper:

Lourdes Moreno, Paloma Martínez, Miguel Domínguez-Gómez, and Marco Antonio Sanchez-Escudero. HULAT2 at MER-TRANS 2026: Governed Multi-Agent Simplification for Spanish Easy-to-Read Generation. In Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2026), CEUR Workshop Proceedings, León, Spain, September 2026. To appear.


For zenodo reference:

Moreno, L., & Martínez, P. (2026). AIGov-Access: AI Governance for Accessibility-Oriented Text Adaptation - MER-TRANS 2026 Profile (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.20855013

> Note: See [`CITATION.bib`](CITATION.bib), or use the DOI.
---

## Funding
This work has been supported by grant PID2023-148577OB-C21 (Human-Centered AI: User-Driven Adapted Language Models-HUMAN\_AI) by MICIU/AEI/10.13039/501100011033 and by FEDER/UE.

