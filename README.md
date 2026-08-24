# CityLens — Pothole Detection

AI-powered pothole detection for automated road-condition monitoring across diverse
urban road imagery. A single-class (`pothole`) object detector built on YOLO11l,
trained and evaluated at 1024×1024, with an RT-DETR baseline and SAHI tiled-inference
comparison.

> **Status:** research / hackathon project (2026). Numbers below are reproducible from
> the notebooks in [`notebooks/`](notebooks/).

---

## Results

All metrics below are measured on the **validation split** (901 images, 2,572 boxes)
using Ultralytics `model.val()`. See the [caveat](#a-note-on-the-numbers) — these are
not held-out test numbers.

| Model | Params | mAP@50 | mAP@50–95 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| YOLO11l — earlier baseline (`v2`) | 25.3M | 0.7032 | 0.3520 | 0.7482 | 0.6377 |
| **YOLO11l — final (`run3_continued`)** | **25.3M** | **0.7147** | **0.3525** | **0.7248** | **0.6715** |
| RT-DETR-l (40 ep, matched) | 32.0M | 0.5750 | 0.2630 | 0.616 | 0.579 |
| SAHI tiled inference (512px slices)¹ | 25.3M | — | — | 0.6629 | 0.6637 |

Headline final model: **71.47% mAP@50, 35.25% mAP@50–95, 72.48% precision, 67.15% recall.**
Recall improved **+3.4 pp** over the earlier baseline (0.6377 → 0.6715); this was a
deliberate recall-oriented trade — precision fell ~2.3 pp and mAP@50–95 stayed flat.
For road-safety monitoring, a missed pothole is costlier than a false alarm, so recall
was prioritized.

¹ SAHI was *explored* and did not beat full-image inference (precision dropped), so it is
not part of the final model. Its P/R come from a custom IoU@0.5 counter at conf 0.25, a
different protocol from the mAP rows — not directly comparable.

### A note on the numbers

- **Reported on validation, not test.** A 504-image test split exists but was not scored
  for the headline. The validation set also drove checkpoint selection during training,
  so these figures carry a mild optimistic bias. Scoring the final weights on the test
  split is the top open item (see [Roadmap](#roadmap)).
- **GFLOPs.** YOLO11l is 25.3M params / 86.6 GFLOPs, where 86.6 is Ultralytics' standard
  640px reference figure. Actual compute at the 1024×1024 training resolution is ~2.5×
  higher (~220 GFLOPs). Don't quote "86.6 GFLOPs at 1024".

---

## Approach

- **Detector:** YOLO11l (25,280,083 params), trained at 1024×1024.
- **Optimization:** AdamW, cosine LR schedule (`lr0=2e-4`, `lrf=5e-3`), AMP, staged
  fine-tuning (832px → 1024px → 1024px continued).
- **Augmentation:** Mosaic (+ `close_mosaic`), Copy-Paste, MixUp, HSV, scaling,
  rotation, horizontal flip, random erasing. Exact values in
  [`configs/train.yaml`](configs/train.yaml).
- **Small-object inference:** SAHI 512×512 tiled prediction (0.2 overlap) — explored,
  shelved.
- **Evaluation:** custom IoU-based matching, 101-point AP (mAP@50 and mAP@50–95),
  confidence-threshold sweep, and size-wise (small/medium/large) recall. See
  [`src/metrics.py`](src/metrics.py).
- **Architecture comparison:** RT-DETR-l trained under a matched 40-epoch budget as a
  transformer baseline; it underperformed the YOLO model.

---

## Repository layout

```
citylens/
├── notebooks/
│   ├── 01_train_eval_yolo11l.ipynb      # training + val + SAHI (main pipeline)
│   └── 02_rtdetr_vs_yolo_compare.ipynb  # RT-DETR arm, metrics module, WBF/complementarity scaffold
├── src/
│   └── metrics.py                       # reusable eval: IoU match, AP, conf sweep, recall-by-area
├── configs/
│   └── train.yaml                       # final training hyperparameters
├── data/
│   ├── data.yaml                        # YOLO dataset config (nc=1, pothole)
│   └── README.md                        # dataset provenance + curation TODO
├── results/
│   └── metrics.md                       # results table (source of truth for the README)
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Quickstart

```bash
git clone https://github.com/<you>/citylens.git
cd citylens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Point `data/data.yaml` at your dataset, then run `notebooks/01_train_eval_yolo11l.ipynb`.
Weights are **not** committed (see [`.gitignore`](.gitignore)); attach `best.pt` to a
GitHub Release or track it with Git LFS.

---

## Roadmap

- [ ] Score the final weights on the **test split** and report those as the headline.
- [ ] Commit the **dataset curation pipeline** (see `data/README.md`) so the multi-source
      claim is reproducible, not asserted.
- [ ] Publish `best.pt` via a GitHub Release (or Git LFS).
- [ ] Add example inference script + sample predictions to `docs/`.
- [ ] Log the confidence-threshold sweep and size-wise recall for the final model.

---

## License

MIT — see [LICENSE](LICENSE).
