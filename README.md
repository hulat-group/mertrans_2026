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

We submitted three fully automatic runs:

- **RUN1: Multi-Agent Workflow**
RUN1 used a LangGraph-based multi-agent workflow combining parallel generation, internal quality signals, Event–Condition–Action routing and controlled editing.

The workflow included:

pre-analysis of the input text,
parallel generation with different simplification strategies,
candidate assessment using internal quality signals,
ECA-based routing,
controlled merging/editing,
final validation before output generation.

This was the best HULAT2 run according to the official SARI score.

- **RUN2: Multi-Agent Workflow with Lexical Support**
RUN2 used the same architecture as RUN1, but activated an additional lexical-support layer before generation.

This layer relied on glossary-based and lexical-resource support to identify difficult terms and propose simpler alternatives or explanatory formulations.

Although lexical support can be useful from an accessibility-oriented perspective, RUN2 obtained a slightly lower official SARI score than RUN1, suggesting that lexical substitutions require careful calibration in reference-based evaluation settings.

- **RUN3: RigoChat-Based Baseline**
  RUN3 was implemented as a linear generate–evaluate–regenerate baseline.

  It used RigoChat-7B-v2 with prompt engineering, post-processing and LoRA-based adaptation. The system generated an Easy-to-Read-oriented output, evaluated it using internal quality criteria and attempted regeneration when the output did not satisfy the expected constraints.

---

## 🤖 Models
The submitted systems used the following models and frameworks:

- **Gemini 2.5 Flash**
Used in the multi-agent workflow for conservative Plain Language generation and Easy-to-Read-oriented generation.

Official model: [Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash?hl=es-419)

- **RigoChat-7B-v2**
Used both as one of the generators in the multi-agent workflow and as the main model in the RUN3 baseline.

RigoChat-7B-v2 is a Spanish-oriented language model based on Qwen2.5-7B-Instruct and adapted for Spanish queries.

Official model: [RigoChat-7B-v2](https://huggingface.co/IIC/RigoChat-7b-v2)  


## 📖 Citation

For citing the GitHub repository:


For citing the conference paper:


---

## Funding
This work has been supported by grant PID2023-148577OB-C21 (Human-Centered AI: User-Driven Adapted Language Models-HUMAN\_AI) by MICIU/AEI/10.13039/501100011033 and by FEDER/UE.

