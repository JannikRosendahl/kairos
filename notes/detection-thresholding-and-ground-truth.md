# KAIROS — detection granularity, thresholding & ground truth

Extends §4 of [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md),
which established the window-level scoring and the keyword denylist. This note answers
the three questions the thesis section left open: **what unit is actually scored and how
many of them there are**, **every threshold in the pipeline**, and **what the ground
truth is and where it comes from**.

Analysed at commit `fdac2948f408bf6bea894bc75fca528f1b0e3bc9` (2026-09-14).
Scripted pipeline is `DARPA/CADETS_E3/`; line references are to that directory unless
stated. Paper text: `resources/KAIROS.txt`.

**Provenance convention.** `[read]` = verified by reading the cited code. `[paper]` =
claim from the paper. `[import-note]` = from a companion note. `[unverified]` = inference
or argument, not confirmed by execution.

---

## 1. Detection granularity

### 1.1 The pipeline, end to end

`[read]`

| Stage | Produces | Location |
| --- | --- | --- |
| 1. Per-edge score | `-log p(true edge type)` per edge | `model.py:60-64`, `test.py:90` |
| 2. Window checkpoint | one `.txt` per window, all edge records sorted by loss desc. | `test.py:115-142` |
| 3. Anomalous nodes | endpoints of edges with loss > `mean + 1.5·std` | `anomalous_queue_construction.py:18-43` |
| 4. Queues | windows chained by shared *rare* anomalous nodes | `anomalous_queue_construction.py:138-189` |
| 5. Queue score vs β | queue flagged; **every window in it** gets label 1 | `evaluation.py:133-167` |
| 6. Metrics | `confusion_matrix` over windows | `evaluation.py:19-36, 170-176` |

**The scored unit is the time window.** Nodes and edges are intermediates: node sets exist
only to chain windows into queues (stage 4) and to seed the attack-summary graph. No
node- or edge-level verdict is ever compared against a label in this repo `[read]`.

### 1.2 How many scored units — the "4 test graphs" question, resolved

The current draft's "4 test graphs" conflates three different fours. Precisely `[read]`:

- **`test()` is called 5 times**, not 4 (`test.py:173-211`): on day-graphs `graph_4_3`,
  `graph_4_4`, `graph_4_5`, `graph_4_6`, `graph_4_7`.
- **Only 2 of those are test days.** Days 3, 4, 5 are replayed to seed the node-IDF
  statistics (`test.py:147` comment "graph_4_3 - graph_4_5 will be used to initialize node
  IDF scores"; consumed at `anomalous_queue_construction.py:48-62`). Days 6 and 7 are the
  evaluated ones (`evaluation.py:40-45, 125-131`).
- **Each day-graph is cut into many windows**, not one. `test.py:115` checkpoints whenever
  `t[-1] > start_time + time_window_size`, evaluated at 1024-event batch boundaries. One
  `.txt` file per window.
- **179 windows are scored** across days 6 and 7 — recoverable from the paper's own
  E3-CADETS row, TP 4 + TN 174 + FP 1 + FN 0 = 179 (`resources/KAIROS.txt:1040-1110`)
  `[paper]`. The code produces the window files; the count is not asserted anywhere in
  the repo.
- **The four ~15-minute spans are the four *attack* windows**, hardcoded in the ground
  truth (`evaluation.py:47-52`), all on day 6.

So: **~179 predictions, not 4.** The "almost 2 hours" observation in the draft is real but
belongs to the fourth *attack window*, not to a test graph: the GT entry
`2018-04-06 12:03:50.186115455~2018-04-06 14:01:32.489584227` spans 1 h 57 m 42 s
(`evaluation.py:51`) `[read]`. The preceding three span 15 m 09 s, 15 m 07 s and 15 m 08 s.

`[unverified, argued]` Why one window is eight times longer: the checkpoint only fires at
a batch boundary *after* the 15-minute mark, so a quiet period with few events produces a
long window. Low event rate → long window. That is the opposite of what a detector wants:
sparse periods, where individual anomalous edges stand out least against the
window-internal `mean + 1.5·std`, are exactly the periods given the longest windows.

### 1.3 Each day's tail is silently dropped

`[read]` The checkpoint at `test.py:115-142` is the only writer of window files, and it
runs *inside* the batch loop. When the loop ends, `test()` returns at `:144` without a
final flush. Every event after the last checkpoint — up to one whole window plus a batch —
is recorded in no file and therefore never scored, never contributes a node to any queue,
and cannot be a TP or FP.

Same class of defect as ORTHRUS's dropped day tails
(`orthrus/src/graph_construction/build_orthrus_graphs.py:107-146`). Worth pairing them in
the chapter: both systems quietly discard the end of every day.

### 1.4 The attack summary graph is not automatic

`[read]` `attack_investigation.py:36-42` hardcodes the same four window paths under the
comment:

```python
# Users should manually put the detected anomalous time windows here
attack_list = [ ... ]
```

and the node colouring uses a 9-entry attack-indicator list whose own comment concedes it
is cosmetic (`attack_investigation.py:99-117`): "They are **only be used to plot the
colors of attack nodes and edges**. They won't change the detection results."

The paper states KAIROS "creates compact summary graphs that highlight possible attack
footprints, all without any human intervention" (`resources/KAIROS.txt:1558-1561`)
`[paper]`. As shipped, the investigation stage is seeded by a hand-pasted window list.
The thesis does not evaluate summary graphs, so this is a footnote — but it is a real
paper↔code gap.

---

## 2. Thresholding

Four thresholds, in pipeline order.

### 2.1 σT — reconstruction threshold (per window)

`[read]` `thr = loss_mean + 1.5 * loss_std` (`anomalous_queue_construction.py:29`), over
the edges of that window only. Matches the paper: "σT is 1.5 standard deviations (SDs)
above the mean of all reconstruction errors in a time window"
(`resources/KAIROS.txt:659-664`) `[paper]`. **No discrepancy.**

`[unverified, argued]` It is *relative to the window*, so it self-calibrates: a window in
which everything is anomalous has a high mean and flags proportionally few edges. A window
of pure routine has a low mean and flags its most unusual edges regardless. There is no
absolute notion of "anomalous" anywhere in the pipeline.

### 2.2 α — rareness threshold on node IDF

`[read]` `if IDF > math.log(len(tw_list) * 0.9)` (`anomalous_queue_construction.py:133`).
Matches the paper's α (`resources/KAIROS.txt:678-681`) `[paper]`, which does not give it a
numeric value — so this is a **code-only** constant.

Three facts about the IDF that the paper does not state `[read]`:

- **Computed over days 3, 4 and 5** (`anomalous_queue_construction.py:48-62`). Days 3 and
  4 are *training* days (`train.py:73-76`). The rareness baseline is therefore measured on
  data the model was fitted to.
- **Netflow nodes are excluded** from the IDF corpus entirely
  (`anomalous_queue_construction.py:71,76`).
- **Unseen nodes get the maximum IDF**, `math.log(len(tw_list)/1)`
  (`anomalous_queue_construction.py:130`) — anything absent from days 3–5 is automatically
  "rare", which is the intended behaviour but means novelty alone, not maliciousness,
  drives queue linking.

### 2.3 The keyword denylist — test-derived, and broader than it looks

`[read]` `is_include_key_word` (`anomalous_queue_construction.py:93-117`) forces the IDF of
matching nodes to `log(len(tw_list)/(1+len(tw_list)))`, which is negative and so always
fails the α test at `:133`. Eleven entries:

```
'netflow', '/home/george/Drafts', 'usr', 'proc', 'var', 'cadet',
'/var/log/debug.log', '/var/log/cron', '/home/charles/Drafts',
'/etc/ssl/cert.pem', '/tmp/.31.3022e'
```

The authors' own comment concedes the provenance (`:94-99`):

> The following common nodes don't exist in the training/validation data, but will have
> the influences to the construction of anomalous queue (i.e. noise). **These nodes
> frequently exist in the testing data** but don't contribute much to the detection…

Two aggravating details `[read]`:

- Matching is plain substring containment (`if i in s`, `:114-116`), and `usr`, `var`,
  `proc`, `cadet` are short substrings. Anything under `/usr/…`, `/var/…`, `/proc/…` is
  suppressed — the majority of a Unix filesystem. `'netflow'` matches the type prefix of
  *every* socket node.
- `cal_set_rel` **mutates `node_IDF` in place** (`:123`), so once a node is suppressed it
  stays suppressed for all later windows in the run — a persistent side effect of what
  reads like a per-call filter.

This is the strongest single instance of test-set leakage in the pipeline, and it is
self-documented. It directly qualifies the paper's assurance that "the ground truth is
used only by us to verify KAIROS' efficacy; KAIROS does not leverage any attack knowledge
in its own analysis" (`resources/KAIROS.txt:1221-1225`) `[paper]`: the denylist is not
attack knowledge, but it *is* test-data knowledge.

### 2.4 β — the queue anomaly threshold

Two discrepancies here, both material.

**(a) The score formula differs from the paper.** `[read]` The code multiplies
`(loss + 1)` per window (`evaluation.py:104-110`, and identically at `:136-140`, `:154-158`):

```python
if anomaly_score == 0:
    # Plus 1 to ensure anomaly score is monotonically increasing
    anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
else:
    anomaly_score = (anomaly_score) * (hq['loss'] + 1)
```

The paper defines the queue score as the plain product of window anomaly scores,
`AnomalyScore(q) = Π AnomalyScore(Ti)` (`resources/KAIROS.txt:707-712`) `[paper]`. The
`+1` is code-only.

`[unverified, argued]` The `+1` changes the statistic's meaning. Every factor is ≥ 1, so
the score is **monotonically increasing in queue length** — a long queue of unremarkable
windows can exceed β purely by being long. With β = 100, a queue of windows each scoring
loss ≈ 0.2 needs about 25 windows to alarm on length alone. Detection is therefore partly
a function of how aggressively the queue-linking chains windows together (§2.2–2.3), not
only of reconstruction error.

**(b) β is hardcoded, not fitted.** `[read]` `config.py:144-145`:

```python
beta_day6 = 100
beta_day7 = 100
```

The paper says "KAIROS uses benign validation data to set β after model training"
(`resources/KAIROS.txt:719-722`) `[paper]`. In the code, the validation day (04-05) is
processed only to **log** its maximum queue score — `logger.info(f"The largest anomaly
score in validation set is: {max(anomalous_queue_scores)}")` (`evaluation.py:119`) — and
that value is never assigned to anything. Confirms §3.4 of the model note, independently.

Note also that the code carries **one β per test day**, whereas the paper describes a
single β. Both happen to be 100, so nothing differs numerically, but the structure permits
per-day tuning against test days.

### 2.5 No max-vs-percentile lever

`[read]` KAIROS has no alternative thresholding mode: σT is fixed at `mean + 1.5·std`,
α at `log(0.9·N)`, β at a constant. GRASP's observation that ORTHRUS/VELOX results are
sensitive to max-vs-percentile thresholding (`resources/GRASP.txt:368-378`) has no direct
analogue here — the corresponding sensitivity in KAIROS would be to the 1.5 SD multiplier
and to β, neither of which is swept in the released code.

---

## 3. Labelling convention

### 3.1 What the ground truth is

`[read]` `ground_truth_label()` (`evaluation.py:38-56`): label every window file in the
day-6 and day-7 output directories 0, then set four hardcoded filenames to 1:

```python
attack_list = [
    '2018-04-06 11:18:26.126177915~2018-04-06 11:33:35.116170745.txt',
    '2018-04-06 11:33:35.116170745~2018-04-06 11:48:42.606135188.txt',
    '2018-04-06 11:48:42.606135188~2018-04-06 12:03:50.186115455.txt',
    '2018-04-06 12:03:50.186115455~2018-04-06 14:01:32.489584227.txt',
]
```

**Unit = time window.** Four positives; day 7 contributes only negatives.

A second hardcoded list of nine attack indicators exists in `calc_attack_edges`
(`evaluation.py:58-95`) — `vUgefal`, `/var/log/devc`, `nginx`, and six IPs — but it only
logs a count (`:95`) and feeds no metric `[read]`.

### 3.2 The ground truth is keyed on filenames produced by a particular run

`[read]` The labels are dictionary keys matching window filenames *to the nanosecond*
(`evaluation.py:40-54`), and those filenames are generated from the realised window
boundaries (`test.py:117`), which depend on the 1024-event batch quantisation, which
depends on the exact event set that survived preprocessing.

`[unverified, argued]` Consequence: any change that shifts an event — a different edge-type
filter, a fixed parser, a different batch size — renames every window and silently zeroes
out the four positives, since `labels[i] = 1` would be writing keys that no longer exist
while `pred_label` is built by `os.listdir`. The evaluation would then report a confusion
matrix over 179 negatives with no error raised. This makes the published number
**brittle to reproduce and impossible to transplant** to another preprocessing pipeline
without re-deriving the GT by hand.

### 3.3 The paper's account

`[paper]` The paper is explicit that DARPA's report is the source and that the projection
to windows is manual (`resources/KAIROS.txt:1216-1235`):

> DARPA provides attack ground truth, which enables us to label individual nodes and edges
> related to the attack. … In both TC and OpTC, attack activity occurred only in a subset
> of time windows within an attack day. … As such, we mark the time window that includes
> the Firefox event as an attack time window. Since each time window is 15-minute long in
> our experiments, the next several time windows are therefore benign time windows, until
> the attack activity resumes.

and at `:1248-1252`: "We compute these metrics based on time windows. … we manually label
each time window in a provenance graph as either benign or attack according to the ground
truth."

So node-level GT exists upstream and is deliberately projected down to windows.

`[unverified, argued]` That projection is generous in a specific way: a window is positive
if it contains *any* attack event, and a correct alarm on that window counts as a full TP
regardless of whether the detector's reason for alarming had anything to do with the
attack. With four positives on the whole dataset, recall is a 4-bit quantity.

### 3.4 Is it a predecessor of the ORTHRUS / PIDSMaker ground truth? No.

Checked directly against the other repos `[read]`:

| | KAIROS | ORTHRUS / PIDSMaker |
| --- | --- | --- |
| Unit | time window | node (UUID) |
| Storage | 4 filenames hardcoded in `evaluation.py:47-52` | per-attack CSVs, e.g. `PIDSMaker/Ground_Truth/orthrus/E3-CADETS/node_Nginx_Backdoor_{06,11,12,13}.csv` |
| Scope | day 6 only | days 6, 11, 12, 13 |
| Size (E3-CADETS) | 4 windows | node lists per attack; ThreaTrace variant is a flat list of 12,858 UUIDs |
| Selectable? | n/a | `cfg.evaluation.ground_truth_version` ∈ {`orthrus`, `reapr`, `threatrace`} (`PIDSMaker/pidsmaker/utils/labelling.py:8-24`, `Ground_Truth/`) |

**PIDSMaker's ground-truth registry has no `kairos` option.** The two conventions are
independent projections of the same upstream source — the DARPA report, shipped in both
repos as `TC_Ground_Truth_Report_E3_Update.pdf` — not one derived from the other.

`[unverified, argued]` One suggestive link, worth a single sentence and no more: ORTHRUS's
GT CSVs carry node labels in the form `{'subject': 'None nginx'}`,
`{'netflow': '128.55.12.73:80->81.49.200.166:43530'}` — the same `{type: label}` dict
shape KAIROS builds in `gen_nodeid2msg` (`kairos_utils.py:117-126`) and writes into its
per-edge records (`test.py:96-97`). Shared tooling ancestry is plausible; shared labelling
convention is not.

### 3.5 Consequences for comparability

`[unverified, argued]` unless marked.

- Window-level GT cannot be compared to node- or edge-level GT without recomputation. A
  system that flags 3 of 4 attack windows and one that flags 3,000 of 4,000 attack nodes
  are not on the same axis.
- KAIROS itself uses **two different granularities in one paper**: Table 4 is window-level,
  while the ThreaTrace comparison adopts "ThreaTrace's way of computing metrics and use
  anomalous nodes in suspicious time windows" (`resources/KAIROS.txt:1561-1565`) `[paper]`.
  Its own numbers are therefore not internally comparable across tables.
- The paper is itself critical of ThreaTrace's convention — 2-hop neighbours labelled
  anomalous, and FPs excused if any 2-hop neighbour is labelled, so "a benign node as far
  as 4 hops away … can be misclassified by ThreaTrace but not reported as a FP"
  (`:1566-1580`) `[paper]`. Good material for the chapter: KAIROS diagnoses the
  convention problem accurately and then introduces its own.
- With 4 positives, recall is either 1.000 or a coarse fraction; every published KAIROS
  recall on E3-CADETS is 1.000 (§ see [`metrics-splits-and-reported-results.md`](metrics-splits-and-reported-results.md)).

---

## 4. Open items

- Size of the dropped day tail (§1.3) — needs the data; one number per test day.
- How many of the 179 windows fall below the length KAIROS's own hyperparameter study
  assumes, and the realised window-length distribution on CADETS E3 (THEIA E3 figures
  exist `[import-note]`).
