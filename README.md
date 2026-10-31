# TechCrunch Multimodal Analysis

Video-level measurement of humor, facial expression, posture, voice, and inclusive collective language, with OLS analysis of venture funding.

## Install

Python 3.11 or 3.12:

```shell
python -m pip install -r requirements.txt
```

## Analyze the included data

```shell
python analysis/reproduce.py
```

Input: `data/analysis/input_428.csv`. Field definitions: `data/analysis/data_dictionary.csv`.

Outputs in `results/`: analysis sample, descriptive statistics, correlations, OLS coefficients, model statistics, VIFs, covariance sensitivity, predicted curves, and figure. The common complete-case sample contains 371 videos. Main results use classical OLS standard errors and two-sided t tests. `covariance_sensitivity.csv` also reports HC0–HC3, company-clustered CR1, and event-clustered CR1 estimates for the primary interaction; missing event labels form one Unknown group.

## Run examples

Aggregate timestamped video evidence:

```shell
python -m pipeline.summarize --input examples/multimodal_evidence.json --output outputs/video_metrics.csv
```

Reference output: `examples/multimodal_metrics.csv`. The command writes a new CSV to `outputs/video_metrics.csv`.

Read the saved scoring response for the complete lead-speaker transcript:

```shell
python text/score_inclusive_language.py --input examples/multimodal_evidence.json --response examples/text_response.json --output outputs/icl.json
```

Inclusive collective language is scored 0–2: absent/internal/ambiguous; external but generic; substantive shared collective. Prompt: `text/prompts/pitch_scores_only.txt`. The response contains one score field, `inclusive_collective_language`. The example score is 1.

## Process videos

```shell
python -m pip install -r requirements-multimodal.txt
```

Install FFmpeg on `PATH` and OpenFace 2.2.0 with its models. Set the shell variables listed in `.env.example`: `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, and `OPENFACE_BINARY`. `OPENAI_BASE_URL` selects an OpenAI-compatible endpoint. Audio models load on first use.

```shell
python -m pipeline.run --input-dir data/videos --output-dir outputs/multimodal
```

Settings: `configs/multimodal.yaml`. Outputs: `metrics_per_video.csv`, per-video evidence JSON, lead-speaker transcripts, and run status. Optional `--recorded-dir`, `--coding-a-dir`, and `--coding-b-dir` accept saved segmentation and humor coding, with JSON files named `<video_id>.json`.

Score a transcript through the API using the ICL-only prompt. Statistical analysis reads the saved scores in the input CSV.

```shell
python text/score_inclusive_language.py --input outputs/multimodal/lead_transcripts/VIDEO_ID.json --execute --output outputs/icl/VIDEO_ID.json
```

## Join measurements and metadata

```shell
python analysis/assemble.py --metrics outputs/multimodal/metrics_per_video.csv --scores outputs/icl --metadata data/analysis/input_428.csv --output outputs/analysis_input.csv
python analysis/reproduce.py --input outputs/analysis_input.csv --output outputs/analysis
```

Rows join by `video_id`. Metadata supplies funding, speaker gender, judge gender composition, transcript-quality status, and video identifiers. Language scores can be a directory of score JSON files or a CSV containing `video_id` and `inclusive_collective_language`.

Humor frequency averages Versions A and B attempts per effective speaking minute. Warmth cues average sample-standardized smiling and inclusive collective language. Facial and posture proportions use valid frames; behavioral rates use effective speaking time. All commands support `--help`.
