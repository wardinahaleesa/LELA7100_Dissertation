# The Representation of Form and Meaning of Malay-English False Friends and True Cognates in Large Language Models (LLMs)

This repository contains the code and analysis pipeline for an MSc dissertation examining how decoder-only large language models (LLMs) represent Malay-English false friends and true cognates at the level of hidden-state activations. The study uses representational similarity analysis to test whether meaning-based similarity exceeds form-based similarity, and how this balance shifts across transformer layers and model architectures.

## Overview

Multilingual LLMs are often assumed to move from surface form toward abstract meaning as information passes through successive layers. This project tests that assumption directly by comparing hidden-state similarity for two lexical phenomena:

- **False friends** — words that look similar across Malay and English but differ in meaning (e.g. Malay *fail* meaning "file", not English "fail").
- **True cognates** — words that are similar in both form and meaning across the two languages.

For each lexical item, prompts were constructed under matched sentence conditions (a Malay target sentence, an English form-match sentence, an English meaning-match sentence, a Malay synonym sentence, and, for false friends only, an English synonym of the intended Malay meaning). Cosine similarity between hidden-state vectors at the target token is used to compute form-based and meaning-based similarity scores, which are then compared across 11 sampled layers in three models.

## Hypotheses

- **H1 (Final-layer form–meaning):** Meaning-based similarity (sim_AC) will exceed form-based similarity (sim_AB) at the final layer of each model, for both false friends and true cognates.
- **H2 (Layer-wise and cross-model variation):** The balance between form-based and meaning-based similarity will vary across layers and differ across model architectures.

## Models

Three pretrained, non-fine-tuned decoder-only models were probed, selected to span Malay-native, regional multilingual, and general multilingual training profiles.

| Model | Description | Total Layers | Sampled Layers |
|---|---|---|---|
| MaLLaM-1.1B | Malay-native model trained on ~90B Malay tokens | 22 | 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 |
| SEA-Lion v3 8B | Southeast Asian multilingual model supporting 11 regional languages | 32 | 1, 3, 5, 7, 10, 13, 16, 19, 22, 25, 27 |
| Qwen2.5-1.5B | General multilingual model with broad cross-lingual coverage | 28 | 1, 3, 5, 7, 10, 13, 16, 19, 22, 25, 27 |

Models are always processed in the fixed order **MaLLaM → SEA-Lion → Qwen**, both during extraction and when generating tables and figures.

## Dataset

The dataset consists of 450 manually constructed prompts covering 10 lexical items (5 false friends, 5 true cognates), verified against the Dewan Bahasa dan Pustaka Malay Dictionary and the Cambridge Dictionary. No preprocessing, exclusion criteria, or train/validation/test split were applied, since the models are used only for frozen representational probing rather than training.

| Phenomenon | Conditions | Prompts per Item | Total Prompts |
|---|---|---|---|
| True cognates | A (Malay target), B (English form match), C (English meaning match), D (Malay synonym) | 40 (4 conditions × 10 sets) | 200 |
| False friends | A, B, C, D, E (English synonym of intended meaning) | 50 (5 conditions × 10 sets) | 250 |

Input files expected by the pipeline:

- `False Friends.xlsx`
- `True Cognates.xlsx`

Each workbook contains one sheet per lexical item, with rows labelled by condition (`A`–`E`) and the corresponding prompt sentence in the third column.

## Similarity and Difference Measures

For each lexical item, model, layer, and prompt, the hidden state at the target token is extracted for each condition and compared to Condition A (the Malay target) via cosine similarity.

| Measure | Comparison | Interpretation |
|---|---|---|
| `sim_AB` | A vs. B | Form-based similarity |
| `sim_AC` | A vs. C | Meaning-based similarity |
| `sim_AD` | A vs. D | Within-language (Malay) semantic consistency |
| `sim_AE` | A vs. E (false friends only) | Meaning recovery via an unambiguous English synonym |

Four pairwise difference scores are derived from these:

| Difference score | Formula | Interpretation |
|---|---|---|
| Δ meaning–form | sim_AC − sim_AB | Positive = meaning dominates; negative = form dominates |
| Δ meaning–control | sim_AC − sim_AD | Cross-lingual meaning similarity vs. Malay baseline |
| Δ form–control | sim_AB − sim_AD | Form similarity vs. Malay baseline |
| Δ meaning–recovery (false friends only) | sim_AE − sim_AB | Recovery of intended meaning against the misleading form |

Across all three models, the pipeline produces 14,850 model–prompt–layer hidden-state extraction instances (450 prompts × 11 layers × 3 models).

## Pipeline Structure

The main script runs as a single sequential pipeline, organised into the following stages:

1. **Dependency installation** — installs required Python packages and, in Colab, checks whether a runtime restart is needed for `bitsandbytes` to work correctly with CUDA.
2. **Setup** — defines model configurations, layer-sampling schemes, colour palettes, and metric labels used throughout the analysis.
3. **Data loading** — parses the `False Friends.xlsx` and `True Cognates.xlsx` workbooks into per-word, per-condition prompt sets.
4. **Extraction** — loads each model in turn (4-bit quantised via `bitsandbytes`), registers a forward hook on the target transformer layer, extracts the hidden state at the target token position for every prompt and layer, and computes cosine similarities and difference scores. Progress is checkpointed to `checkpoint.pkl` so the run can resume after an interruption.
5. **Bootstrap confidence intervals** — aggregates the 10 prompt-level observations per lexical item into item-level estimates, then computes non-parametric bootstrap 95% confidence intervals (10,000 resamples) for each similarity and difference measure.
6. **Tables and figures** — generates four summary CSV tables (final-layer summary, per-layer similarity, per-layer Δ profiles, per-word final-layer summary) and three sets of figures per model (similarity profiles across layers, Δ profiles across layers, and word-by-layer Δ meaning–form heatmaps).
7. **Hypothesis testing** — evaluates H1 (whether the 95% CI for Δ meaning–form lies entirely above zero at the final layer) and H2 (whether CIs for early/middle/late layers, and for the same layer across models, fail to overlap, taken as evidence of variation).
8. **Packaging** — zips all output tables and figures into `dissertation_experiment_results.zip` for download.

## Repository Structure

```
.
├── dissertation_pipeline.py          # Main analysis script
├── False Friends.xlsx                # Input prompts (false friends)
├── True Cognates.xlsx                # Input prompts (true cognates)
└── dissertation_experiment_results/
    ├── tables/
    │   ├── raw_similarity_scores.csv
    │   ├── bootstrap_summary.csv
    │   ├── overall_bootstrap_summary.csv
    │   ├── table1_final_layer_summary.csv
    │   ├── table2_per_layer_similarity.csv
    │   ├── table3_per_layer_delta.csv
    │   ├── table4_per_word_final_layer.csv
    │   ├── h1_results.csv
    │   ├── h2_layer_variation_results.csv
    │   ├── h2_model_comparison_results.csv
    │   └── h2_summary.csv
    └── figures/
        ├── fig1_similarity_<model>.png
        ├── fig2_delta_<model>.png
        └── fig3_heatmap_<model>.png
```

## Requirements

- Python 3.10+
- A CUDA-capable GPU is strongly recommended (models are loaded in 4-bit via `bitsandbytes`); the pipeline will fall back to CPU but will run slowly.
- A Hugging Face account with access to the three model repositories (`mesolitica/mallam-1.1b-4096`, `aisingapore/Llama-SEA-LION-v3-8B-IT`, `Qwen/Qwen2.5-1.5B`), and a valid Hugging Face access token for `login()`.

Dependencies installed automatically by the script:

```
numpy==1.26.4, scipy, pandas, matplotlib, seaborn, transformers, torch,
accelerate, bitsandbytes>=0.46.1, huggingface_hub, statsmodels, einops, openpyxl
```

## Usage

1. Place `False Friends.xlsx` and `True Cognates.xlsx` in the working directory (or upload them when prompted, if running in Google Colab).
2. Run the script. On first execution, you will be prompted to authenticate with Hugging Face via `login()`.
3. The pipeline will:
   - extract hidden states and similarity scores for all three models in order (MaLLaM → SEA-Lion → Qwen),
   - compute bootstrap confidence intervals,
   - generate all tables and figures,
   - run the H1 and H2 hypothesis tests, and
   - package all outputs into `dissertation_experiment_results.zip`.

```bash
python dissertation_pipeline.py
```

If the extraction step is interrupted, re-running the script will resume from `checkpoint.pkl` rather than starting over.

## Summary of Findings

**Hypothesis 1 (final-layer meaning dominance) was not supported.** Across all three models and both lexical phenomena (0 of 6 model–phenomenon combinations), form-based similarity (sim_AB) exceeded meaning-based similarity (sim_AC) at the final layer. SEA-Lion-v3-8B showed the largest form–meaning separation, particularly for true cognates; Qwen2.5-1.5B showed the weakest separation, with similarity scores across conditions closely distributed.

**Hypothesis 2 (layer-wise and cross-model variation) was supported.** Five of six model–phenomenon combinations showed non-overlapping confidence intervals across early, middle, and final layers, consistent with a broad pattern of early-layer form anchoring, middle-layer movement toward meaning, and later-layer re-orientation toward output-shaping (lexical prediction). The magnitude and direction of this shift varied by model architecture and by lexical phenomenon, with the largest divergence found between MaLLaM-1.1B and SEA-Lion-v3-8B. At the lexical item level, false friends showed greater variability than true cognates, with a small number of items (e.g. *fail*, *am*, *had*) showing temporary meaning dominance at intermediate layers before form regained a lead by the final layer.

These findings suggest that representational depth in these models should not be understood as a simple, monotonic transition from form to meaning. Form remains a strong cue at the final layer — plausibly reflecting the next-token prediction objective — even though intermediate layers show clearer movement toward semantic organisation. The results are discussed in relation to shortcut learning, cross-lingual representation alignment, and implications for multilingual safety and Malay-language AI development, where surface-form reliance under false-friend conditions may be relevant to how models handle culturally and linguistically ambiguous inputs.

## Limitations

- The analysis measures representational similarity, not causal mechanisms; it cannot establish whether intermediate-layer shifts toward meaning causally affect model behaviour or downstream safety judgements.
- Cross-model differences are descriptive; training data, tokenisation, language exposure, architecture, and scale were not independently controlled.
- The dataset is a relatively small, manually constructed set of 10 lexical items; findings may not generalise to other lexical items, language pairs, or model families.
