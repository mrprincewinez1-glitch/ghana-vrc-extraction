# Low-resource document extraction for Ghanaian motor insurance: pilot study

Code and results for the pilot study "Low-Resource Natural Language Processing (NLP) for Document Extraction in Ghanaian Motor Insurance: A Pilot Study". The study compares a Tesseract OCR baseline with prompted Gemini vision extraction on 20 photographs of Ghanaian vehicle registration documents (five scored fields per document: chassis number, registration number, make, model, colour) and one Ghana Card.

## What is in this repository

- `results/benchmark_results.csv`: per-field results for both systems (run 10). Chassis and registration values are replaced by length-preserving pseudonyms (equal values stay equal) and Ghana Card values are masked. Edit distances were stored before masking.
- `recompute_tables.py`: recomputes the paper's Tables 1 and 2, with exact binomial intervals, from the results CSV. Standard library only.
- `evaluate_benchmark.py`: the evaluator that produced the results (see below for what cannot be reproduced).
- `src/vrc_ocr.py`, `src/ocr_pipeline.py`: the Tesseract baseline (Method A).
- `src/gemini_vision_ocr.py`: the current Gemini extractor. `results/gemini_vision_ocr_run10_frozen.py` is the version used for the evaluated runs and contains the exact prompt (Appendix A of the paper).
- `results/ground_truth_vrc_public.json` and `results/outputs_public/`: labels and Gemini outputs with chassis and registration values pseudonymised and owner names removed.
- `results/ground_truth_corrections_log.csv`: the nine label corrections (Appendix B of the paper).
- `run*.log`: console logs of the evaluation runs.

## Reproduce the reported tables

    python3 recompute_tables.py

Expected: Gemini 97.00% (97/100) and Tesseract 8.00% (8/100) field-level exact match on the 20 VRC documents, with 17/20 and 0/20 documents fully correct.

## What cannot be reproduced from this repository

The photographs contain clients' personal data and are not released, so the extractions cannot be repeated and the raw-text diagnostic (Table 3 of the paper) cannot be recomputed. `evaluate_benchmark.py` needs the photographs, Tesseract, a Gemini API key and the private labels. Because identifiers are pseudonymised, edit distances cannot be recomputed from the released labels, and the stored values in the results CSV are used instead. Hosted-model outputs may change when the provider updates a model.

## Environment

Python 3.14.7, Tesseract 5.5.3, and the packages pinned in `requirements.txt`.

## Data handling

Client photographs were sent to a hosted API on a free tier whose terms advise against submitting personal data (see Section 6 of the paper). Do not repeat this with real client documents. Use a paid tier or an on-premises model.

## AI assistance

Gemini and Claude assisted with writing and debugging the code and the manuscript. See the AI Assistance Statement in the paper.

## License

MIT. See `LICENSE`.
