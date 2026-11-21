# Low-resource field extraction from Ghanaian vehicle registration documents

Code and results for the pilot study "Low-Resource Field Extraction from Ghanaian Vehicle Registration Documents: A Pilot Comparison of OCR and Vision-Language Prompting". The study compares a Tesseract OCR baseline with prompted Gemini vision extraction on photographs of Ghanaian vehicle registration documents. Five fields are scored per document: chassis number, registration number, make, model and colour.

## Evaluation design

The evaluation ran in two stages with the same extraction prompt and the same model (gemini-3.6-flash, free tier).

- Stage 1 (pilot): 20 documents, run 10. Labels and extractions for these were frozen before the second stage began.
- Stage 2: 35 further documents, labelled by the author from the photographs and extracted with the frozen prompt.
- Final evaluation: all 55 documents pooled (275 scored fields).

| System | Exact match | Character error rate | Documents fully correct |
|---|---|---|---|
| Gemini vision | 93.82% (258/275) | 1.43% | 40/55 (72.73%) |
| Tesseract 5.5.3 | 5.45% (15/275) | 93.25% | 0/55 |

The breakdown by stage is reported in the paper. The client consent obtained for this work was verbal.

## What is in this repository

- `results/final_55_public.csv`: per-field results for both systems on all 55 documents.
- `results/benchmark_results.csv`: per-field results for the 20-document pilot (run 10).
- `results/ground_truth_vrc_public.json`, `results/ground_truth_vrc_batch2_public.json`: labels for stage 1 and stage 2.
- `results/outputs_public/`, `results/batch2_public/`: Gemini outputs for stage 1 and stage 2.
- `results/ground_truth_corrections_log.csv`: the label corrections made during stage 1.
- `recompute_tables.py`: recomputes the tables, with exact binomial intervals, from a results CSV. Standard library only.
- `evaluate_benchmark.py`: the pilot evaluator. `evaluate_final.py`: the 55-document evaluator. `extract_batch2.py`: stage 2 extraction. `test_tesseract_batch2.py`: Tesseract on stage 2. `freeze_run10.py`: freezes the stage 1 files.
- `src/vrc_ocr.py`, `src/ocr_pipeline.py`: the Tesseract baseline.
- `src/gemini_vision_ocr.py`: the current Gemini extractor. `results/gemini_vision_ocr_run10_frozen.py` is the version used for the evaluated runs and contains the exact prompt.
- `run*.log`: console logs of the pilot runs.

## Reproduce the reported tables

    python3 recompute_tables.py results/final_55_public.csv

Expected: Gemini 93.82% (258/275) and Tesseract 5.45% (15/275) field-level exact match, with 40/55 and 0/55 documents fully correct.

For the 20-document pilot alone:

    python3 recompute_tables.py results/benchmark_results.csv

Expected: Gemini 97.00% (97/100) and Tesseract 8.00% (8/100).

## Privacy and what was removed

- Chassis and registration values are replaced by same-length hexadecimal pseudonyms (HMAC-SHA256 with a randomly generated secret key that is not released). Equal values stay equal, different values stay different, and empty values stay empty.
- The run 10 prompt asked the model to also read the owner's name. That field was removed from the prompt for stage 2 and is removed from every released file. Owner names, phone numbers and addresses are not released.
- Client photographs, the Ghana Card image and the unmasked labels are not released.

## What cannot be reproduced from this repository

The photographs contain personal data, so the extractions cannot be repeated. Because identifiers are pseudonymised, character-level measures cannot be recomputed from the released labels; the stored edit distance and error-rate columns, computed before masking, are used instead. Hosted-model outputs may change when the provider updates a model.

## Environment

Python 3.14.7, Tesseract 5.5.3, and the packages pinned in `requirements.txt`.

## Data handling warning

Client photographs were sent to a hosted API on a free tier whose terms advise against submitting personal data (see the paper). Do not repeat this with real client documents. Use a paid tier or an on-premises model.

## AI assistance

Gemini and Claude assisted with writing and debugging the code and the manuscript. See the AI Assistance Statement in the paper.

## License

MIT. See `LICENSE`.

## Notes on the run logs

- `run8.log` and `run10.log` use the same saved Gemini extractions. Their per-document lines are identical and their summary scores differ; `run10` was scored against the corrected labels listed in `results/ground_truth_corrections_log.csv`. `run10` is the evaluation reported in the paper.
- Logs for runs 5 and 6 were not retained, so the log numbering has a gap. `run9` did not complete because of API errors and free-tier quota limits.
