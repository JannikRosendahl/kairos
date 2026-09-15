# KAIROS — metrics, splits & reported results

Fills the three paragraphs the thesis section left empty (metrics + imbalance, data &
split, reported results) and closes the last open item in §4 of
[`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md):
**how the "adjusted" results table is produced**.

Analysed at commit `fdac2948f408bf6bea894bc75fca528f1b0e3bc9` (2026-09-14).
Scripted pipeline is `DARPA/CADETS_E3/`; line references are to that directory unless
stated. Paper text: `resources/KAIROS.txt`.

**Provenance convention.** `[read]` = verified by reading the cited code. `[paper]` =
claim from the paper. `[measured]` = recomputed here from the paper's own counts by a
deterministic script. `[unverified]` = inference or argument.

---

## 1. Metrics

### 1.1 What is reported, and what is computed

`[paper]` Table 4 reports **TP, TN, FP, FN, precision, recall, accuracy and AUC**, all at
time-window granularity (`resources/KAIROS.txt:1026-1110`, `:1248-1250`).

`[read]` `classifier_evaluation` (`evaluation.py:19-36`) computes exactly those, plus an
F-score the paper does not print:

```python
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
precision = tp/(tp+fp)
recall    = tp/(tp+fn)
accuracy  = (tp+tn)/(tp+tn+fp+fn)
fscore    = 2*(precision*recall)/(precision+recall)
auc_val   = roc_auc_score(y_test, y_test_pred)
```

So the published table *is* reproducible from this function — conditional on the ground
truth matching the generated window filenames, which is fragile
(see [`detection-thresholding-and-ground-truth.md`](detection-thresholding-and-ground-truth.md) §3.2).

### 1.2 The reported "AUC" is not an AUC

`[read]` `roc_auc_score(y_test, y_test_pred)` at `evaluation.py:30` is called on
`y_pred`, which is a vector of **hard 0/1 labels** built at `evaluation.py:123-167` — not
on scores. ROC-AUC over a binary predictor degenerates to a single operating point, and
equals balanced accuracy, `(TPR + TNR)/2`.

`[measured]` Recomputed from the paper's own TP/TN/FP/FN, every row of Table 4 matches to
the printed three decimals:

| Dataset | windows | pos. rate | (TPR+TNR)/2 | paper AUC |
| --- | ---: | ---: | ---: | ---: |
| Manzoor et al. | 475 | 21.1% | 1.0000 | 1.000 |
| E3-THEIA | 227 | 4.0% | 0.9954 | 0.995 |
| **E3-CADETS** | **179** | **2.2%** | **0.9971** | **0.997** |
| E3-ClearScope | 119 | 4.2% | 0.9912 | 0.991 |
| E5-THEIA | 176 | 1.1% | 0.9971 | 0.997 |
| E5-CADETS | 254 | 2.8% | 0.9818 | 0.982 |
| E5-ClearScope | 232 | 4.3% | 0.9887 | 0.989 |
| OpTC | 1248 | 1.8% | 0.9935 | 0.993 |

All eight rows match. The figure carries no threshold-free ranking information; it is a
rescaling of the same confusion matrix already printed beside it. The same
`roc_auc_score(y_test, y_test_pred)` call appears in every dataset notebook
(e.g. `OpTC/optc_graph_learning.ipynb:1829`,
`CLEARSCOPE_E5/clearscope5_graph_learning.ipynb:3105`), so this is repo-wide `[read]`.

This also explains the hyperparameter study: Fig. 3 and Fig. 7 sweep hyperparameters
against "AUC" (`resources/KAIROS.txt:1623`, `:2594`) `[paper]`, which is therefore a
sweep of balanced accuracy over a fixed β, not a threshold-independent comparison.

### 1.3 No imbalance-aware metric is used

`[read]` Grep across `DARPA/` for `matthews`, `MCC`, `AUPRC`: no hits.
`average_precision_score` is **imported and never called** (`kairos_utils.py:12`, and the
same dead import at the head of every notebook). Nothing computes MCC, AP/AUPRC or ADP.

`[measured]` MCC recomputed from the paper's counts, alongside what the paper prints:

| Dataset | precision | recall | accuracy | "AUC" | **MCC** |
| --- | ---: | ---: | ---: | ---: | ---: |
| E3-THEIA | 0.818 | 1.000 | 0.991 | 0.995 | 0.900 |
| **E3-CADETS** | **0.800** | **1.000** | **0.994** | **0.997** | **0.892** |
| E3-ClearScope | 0.714 | 1.000 | 0.983 | 0.991 | 0.838 |
| E5-THEIA | 0.667 | 1.000 | 0.994 | 0.997 | 0.814 |
| E5-CADETS | 0.438 | 1.000 | 0.965 | 0.982 | 0.649 |
| E5-ClearScope | 0.667 | 1.000 | 0.978 | 0.989 | 0.807 |
| OpTC | 0.579 | 1.000 | 0.987 | 0.993 | 0.756 |

Accuracy and "AUC" sit in the high .9s across the board while MCC ranges 0.65–0.90. On
E5-CADETS the gap is 0.982 vs 0.649.

### 1.4 The imbalance, quantified

`[measured]` E3-CADETS has **4 positive windows out of 179** (2.2%). The consequence of
that base rate, holding TP and FN fixed and moving one window from TN to FP:

| FP | precision | accuracy | MCC |
| ---: | ---: | ---: | ---: |
| 1 (published) | 0.800 | 0.9944 | 0.892 |
| 2 | 0.667 | 0.9888 | 0.812 |
| 3 | 0.571 | 0.9832 | 0.749 |
| 4 | 0.500 | 0.9777 | 0.699 |

**One additional false-positive window costs 13.3 points of precision and 0.6 points of
accuracy.** Precision is a 5-bit quantity here; accuracy is nearly constant by
construction. Reporting both without an imbalance-aware summary lets the near-1.0 accuracy
and "AUC" set the reader's impression.

`[unverified, argued]` Recall is the weaker half still: with 4 positives, recall takes one
of five values, and KAIROS reports 1.000 on **every dataset** in both tables. A metric
that cannot distinguish between models is not evidence about them.

---

## 2. Data & split

### 2.1 Datasets

`[paper]` DARPA TC E3 and E5 (THEIA, CADETS, ClearScope each), DARPA OpTC, plus the
non-DARPA Manzoor et al. StreamSpot data (`resources/KAIROS.txt:992-1020`, table rows at
`:1026-1040`). The draft's sentence is correct; StreamSpot is the omission — the repo has
a top-level `StreamSpot/` directory `[read]`.

Repo coverage `[read]`: `DARPA/{CADETS_E3, THEIA_E3, CLEARSCOPE_E3, CADETS_E5, THEIA_E5,
CLEARSCOPE_E5, OpTC}`. **Only CADETS E3 is a scripted pipeline**
(`create_database.py`, `embedding.py`, `train.py`, `test.py`,
`anomalous_queue_construction.py`, `evaluation.py`, `attack_investigation.py`, `Makefile`);
every other dataset ships as two Jupyter notebooks with hyperparameters inlined.

### 2.2 CADETS E3 split — paper vs code

`[paper]` Table 12 (`resources/KAIROS.txt:2514-2575`): E3-CADETS train
`2018-04-02/03/04`, validation `2018-04-05`, test `2018-04-06` and `2018-04-07`.

`[read]` The code agrees:

| Role | Days | Location |
| --- | --- | --- |
| Train | `graph_4_2`, `graph_4_3`, `graph_4_4` | `train.py:73-76` |
| IDF seeding | `graph_4_3`, `graph_4_4`, `graph_4_5` | `test.py:148-150`, `anomalous_queue_construction.py:48-62` |
| Validation (queues) | `graph_4_5` | `anomalous_queue_construction.py:198-203` |
| Test | `graph_4_6`, `graph_4_7` | `evaluation.py:125-131` |

**Chronological** — unlike ORTHRUS, which tests on day 6 while training on days 3–10
(`orthrus/src/config.py:202-221`). Say so; it is a point in KAIROS's favour.

Two qualifications `[read]`:

- The IDF corpus spans days 3–5, of which **days 3 and 4 are training days**. The rareness
  statistic that gates queue construction is measured partly on fitted data.
- The validation day sets nothing: β is the hardcoded constant 100 and the validation score
  is only logged (`config.py:144-145`, `evaluation.py:119`). See
  [`detection-thresholding-and-ground-truth.md`](detection-thresholding-and-ground-truth.md) §2.4.

`[read]` Day graphs are generated for days 2–13 (`embedding.py:90`), so days 8–13 are built
and never used by the scripted pipeline. E3-CADETS test coverage is two days.

### 2.3 Per-dataset hyperparameter drift

`[read]` `config.py` has no argparse and no YAML; every script imports module-level
constants, and each notebook inlines its own. Known divergences:

| Parameter | CADETS E3 | Other E3/E5 notebooks | OpTC |
| --- | --- | --- | --- |
| `epoch_num` | 50 (`config.py:131`) | 30 `[import-note]` | 10 `[import-note]` |
| subject key | `exec` | THEIA: `cmdLine,tgid,path` `[import-note]` | — |
| edge types | 7 (`config.py:67-75`) | dataset-dependent | — |

The paper states one hyperparameter set "in all the experiments"
(`resources/KAIROS.txt:1745-1755`): |Φ| = 16, |s(v)| = 100, |N| = 20, |z| = 200,
|tw| = 15 min `[paper]`. Epoch count is not among them, and the code does not hold it
constant.

---

## 3. Reported results

### 3.1 Table 4 — as published

`[paper]` `resources/KAIROS.txt:1026-1110`. Counts first; at these magnitudes the counts
are the informative part.

| Dataset | TP | TN | FP | FN | Precision | Recall | Accuracy | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Manzoor et al. | 100 | 375 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| E3-THEIA | 9 | 216 | 2 | 0 | 0.818 | 1.000 | 0.991 | 0.995 |
| **E3-CADETS** | **4** | **174** | **1** | **0** | **0.800** | **1.000** | **0.994** | **0.997** |
| E3-ClearScope | 5 | 112 | 2 | 0 | 0.714 | 1.000 | 0.983 | 0.991 |
| E5-THEIA | 2 | 173 | 1 | 0 | 0.667 | 1.000 | 0.994 | 0.997 |
| E5-CADETS | 7 | 238 | 9 | 0 | 0.438 | 1.000 | 0.965 | 0.982 |
| E5-ClearScope | 10 | 217 | 5 | 0 | 0.667 | 1.000 | 0.978 | 0.989 |
| OpTC | 22 | 1210 | 16 | 0 | 0.579 | 1.000 | 0.987 | 0.993 |

**FN = 0 everywhere; recall = 1.000 everywhere.** The entire spread in the table comes
from false positives.

### 3.2 Table 5 — "adjusted", and exactly what it adjusts

This closes the open item. `[paper]` The justification is stated plainly
(`resources/KAIROS.txt:1262-1281`):

> KAIROS continues to assign high reconstruction errors to edges whose nodes were under
> the attacker's influence even after the attacker stops actively manipulating them. …
> However, in the ground truth, entities that remain active after the attack are often
> dismissed… **Any entity, once compromised by an attacker, should be considered
> problematic. We manually identify these "fake" FPs** … **and we show the adjusted
> results in Table 5.**

So the adjustment is a **hand relabelling of the ground truth**, performed by the authors,
on the basis that their detector's disagreement with the GT reflects a GT deficiency.

`[measured]` Comparing Table 4 with Table 5 (`resources/KAIROS.txt:1116-1200`) confirms
mechanically that **only labels move, never predictions**:

| Dataset | TP | FP | TN unchanged | TP+FP invariant | precision |
| --- | --- | --- | --- | --- | --- |
| E3-THEIA | 9→10 | 2→1 | yes | yes | 0.818→0.909 |
| **E3-CADETS** | **4→4** | **1→1** | **yes** | **yes** | **0.800→0.800** |
| E3-ClearScope | 5→5 | 2→2 | yes | yes | 0.714→0.714 |
| E5-THEIA | 2→2 | 1→1 | yes | yes | 0.667→0.667 |
| E5-CADETS | 7→16 | 9→0 | yes | yes | 0.438→**1.000** |
| E5-ClearScope | 10→10 | 5→5 | yes | yes | 0.667→0.667 |
| OpTC | 22→32 | 16→6 | yes | yes | 0.579→0.842 |

In every row TN is identical and TP+FP is conserved: the set of windows KAIROS alarmed on
does not change at all. Windows are simply reclassified from FP to TP. Three datasets are
affected — E3-THEIA, E5-CADETS and OpTC — matching the paper's own summary, "Notice the
significant improvement for E3-THEIA, E5-CADETS, and OpTC" (`:1279-1281`) `[paper]`.

Two consequences worth stating in the chapter:

- **E3-CADETS is unaffected.** For the dataset this thesis uses as primary, Table 4 and
  Table 5 are identical, so the adjustment does not bear on the headline CADETS number.
- **E5-CADETS goes from 0.438 to perfect precision by relabelling 9 alarms.** That is the
  row that carries the argument, and it is the one produced entirely by hand.

`[read]` **No code in the repository produces Table 5.** `evaluation.py` computes a single
confusion matrix from `ground_truth_label()`, whose attack list is the four fixed windows;
there is no second GT, no "adjusted" path, no exclusion list.

`[read]` The supplementary material (`supplementary-material.pdf`) does not supply the
derivation either. Its §1 announces only "attack descriptions … and the associated attack
summary graphs" plus "representative false positive samples"; a text extraction of the
whole document contains the string "adjust" **zero** times, and "false positive" only in
that §1 sentence. §3 shows FP examples as figures without a reclassification rule.

So the adjusted table is **not reproducible from the artifact** — this is the finding,
stated as such. The per-dataset FP narratives in the paper (`:1262-1300`) and the
supplementary's §3 figures are the only account of which windows were relabelled, and
neither enumerates them.

`[unverified, argued]` The methodological objection is worth one sentence and no more:
"any entity, once compromised, should be considered problematic" is a defensible security
position, but applied *after* seeing which windows the detector flagged, it makes the GT a
function of the predictions. Table 5 cannot be used to compare KAIROS to any other system.

### 3.3 Cross-granularity comparison inside the paper

`[paper]` Table 8's ThreaTrace comparison abandons window granularity: "For DARPA
datasets, we adopt ThreaTrace's way of computing metrics and use anomalous nodes in
suspicious time windows to compute precision, recall, and accuracy"
(`resources/KAIROS.txt:1561-1565`). KAIROS's own numbers therefore span two different units
across tables and are not internally comparable.

### 3.4 Third-party re-evaluation

`[unverified]` Not run down in this pass. ORTHRUS reports KAIROS numbers
(`resources/ORTHRUS.txt`) and VELOX/THESEUS re-evaluate prior systems under a shared
framework; GRASP states it could not reproduce ORTHRUS/VELOX detection results at all
(`resources/GRASP.txt:368-378`) `[paper]`. Collecting the re-reported KAIROS numbers and
contrasting them with Table 4 is a discrete follow-up — it belongs in the synthesis
section rather than the KAIROS section, since it is a statement about the corpus, not
about KAIROS's code.

---

## 4. Open items

- Third-party re-evaluated KAIROS numbers gathered into one table (§3.4).
- Whether the E5/OpTC notebooks differ from the CADETS E3 scripts in *model* terms
  (carried over from the model note; the import note covers their preprocessing).
- Realised window count per test day recomputed from our own import, to check the 179
  figure independently of the paper.
