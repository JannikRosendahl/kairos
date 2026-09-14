# KAIROS — model, methodology & architecture analysis

Companion to [`kairos-data-import-analysis.md`](kairos-data-import-analysis.md), which
covers record selection, featurisation and the exclusion catalogue. This note covers the
**model**: what the encoder computes, how memory is scoped, and where the paper and the
code disagree.

Analysed at commit `68685ceb59bc15db81b8b5ca3057411c198bd708` (2026-09-08);
training/evaluation granularity pass (§3.2 addendum, §3.4, §3.5) added 2026-09-14.
Scripted pipeline is `DARPA/CADETS_E3/`; all line references are to that directory
unless stated. Paper text: `resources/KAIROS.txt`.

**Provenance convention.** `[read]` = verified by reading the cited code during this
analysis. `[import-note]` = established in the companion import analysis. `[paper]` =
claim from the paper. `[unverified]` = not yet checked; do not cite until confirmed.

---

## 1. Architecture

KAIROS is a genuine TGN. Three modules, jointly optimised `[read]` (`train.py:76-104`):

```python
memory = TGNMemory(max_node_num, node_feat_size, node_state_dim, time_dim,
                   message_module=IdentityMessage(node_feat_size, node_state_dim, time_dim),
                   aggregator_module=LastAggregator())
gnn = GraphAttentionEmbedding(in_channels=node_state_dim, out_channels=edge_dim,
                              msg_dim=node_feat_size, time_enc=memory.time_enc)
link_pred = LinkPredictor(in_channels=edge_dim, out_channels=len(include_edge_type))
```

| Component | What it is | Location |
| --- | --- | --- |
| Memory | PyG `TGNMemory`, GRU-updated per-node state, dim 100 | `train.py:79` `[read]` |
| Message fn | `IdentityMessage` | `train.py:83` `[read]` |
| Aggregator | `LastAggregator` ("last wins") | `train.py:84` `[read]` |
| Embedding | `GraphAttentionEmbedding` = 2× `TransformerConv`, 8 heads then 1 | `model.py:14-33` `[read]` |
| Time encoder | `memory.time_enc`, PyG Fourier-style, **shared** with the GNN | `train.py:89` `[read]` |
| Decoder | `LinkPredictor` MLP → edge-type logits | `model.py:35-58` `[read]` |
| Neighbour loader | `LastNeighborLoader`, size 20 | `train.py:102` `[read]` |

### Resolves the "UniMP is not a TGN?" question

The draft chapter asked whether §4.2's UniMP model contradicts the TGN claim. It does
not — **UniMP is the GNN inside the TGN**. `GraphAttentionEmbedding` is a
`TransformerConv` stack (i.e. UniMP), and it consumes the memory state `z` produced by
`TGNMemory`. The two are layers of one architecture, not alternatives.

### Time genuinely enters the model

Unlike ORTHRUS, KAIROS encodes elapsed time explicitly `[read]` (`model.py:23-30`):

```python
def forward(self, x, last_update, edge_index, t, msg):
    rel_t = last_update[edge_index[0]] - t
    rel_t_enc = self.time_enc(rel_t.to(x.dtype))
    edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
```

`rel_t` is the gap between a neighbour's last memory update and the current event time,
Fourier-encoded and concatenated onto the edge attributes before attention. (The
`last_update - t` ordering, rather than `t - last_update`, matches the upstream PyG TGN
example — not a bug.)

**This is the fact that reframes the ORTHRUS comparison**: KAIROS has both memory *and*
a time encoder; ORTHRUS has neither, and wins. See the ORTHRUS note §1.

---

## 2. Memory scope — narrower than the paper implies

`[read]` The training loop is:

```python
# train.py:117-126
for epoch in range(1, epoch_num+1):
    for g in train_data:              # one call per DAY-GRAPH
        loss = train(train_data=g, memory=memory, ...)
```

and `train()` opens with `memory.reset_state()` (`train.py:33`) `[read]`. `test()` does
the same (`test.py:40`) `[import-note]`.

**Therefore the memory is wiped at every day boundary.** It never carries across days;
within a day it carries across the 15-minute analysis windows. Any claim KAIROS makes
about capturing long-range dependencies is bounded by a single day — worth stating
whenever the paper's "long-range" language is quoted.

Ordering within a day is chronological and non-shuffled (`train_data.seq_batches`,
`train.py:37`) `[read]`.

### No label leakage into the predicted edge

`[import-note]`, re-checked here `[read]`. `msg` carries the edge-type one-hot, and the
target is decoded from that same space, which looks like leakage. It is not, for the
edge being predicted:

- `link_pred` sees only `z[assoc[src]], z[assoc[dst]]` — never the current edge's `msg`
  (`train.py:49`).
- `memory.update_state(src, pos_dst, t, msg)` runs **after** `pos_out` is computed
  (`train.py:63`).
- `gnn(...)` receives `train_data.msg[e_id]`, i.e. messages of *neighbour* edges already
  in the buffer.

The current edge's type reaches the model only through memory and neighbour state
written by *earlier* edges. Worth stating explicitly, because the code shape argues the
opposite.

---

## 3. Paper ↔ code discrepancies

### 3.1 The edge-type space is 7, not 9

`[paper]` The decoder target is described as a categorical feature of **dimension 9**.

`[read]` The code uses **7** for CADETS E3. `include_edge_type` (`config.py:67-75`) is
exactly `EVENT_WRITE, EVENT_READ, EVENT_CLOSE, EVENT_OPEN, EVENT_EXECUTE, EVENT_SENDTO,
EVENT_RECVFROM`, and the decoder's output width is `len(include_edge_type)`
(`train.py:96`). Everything outside that list is dropped at graph construction
(`embedding.py:105`) `[import-note]`.

This resolves the open `\todo` in `rel_work_kairos.tex`. The class count differs by
dataset — check per experiment before quoting a single number.

### 3.2 The window size *is* stated — the earlier note was wrong

The claim that KAIROS hides its window size does **not** survive checking. `[paper]`
`resources/KAIROS.txt:843` gives "time window length |tw| = 15 minutes", and `:1755`
adds "We find |tw| = 15 minutes to be ideal among all datasets". `[read]` The code
agrees: `config.py:133-135`, `time_window_size = 60000000000 * 15`.

The genuine code-only findings are §2 (per-day memory reset), §3.3 (keying), §3.4 (β),
§3.5 (IDF seeding) and §4 (filters) — not the window size.

What the paper does *not* state is that the window is inert during training and that the
realised windows are not 15 minutes. `time_window_size` has exactly one reader,
`test.py:115`; `train.py` never mentions it. And because the boundary is only tested once
per 1024-event batch with `start_time = t[-1]` on flush (`test.py:140`), windows are
≥15 min with accumulating drift — median 15.48 min, max 28.9 min over the 454 logged
THEIA E3 windows, and one CADETS E3 ground-truth attack window spans 1 h 58 m
(`evaluation.py:51`). `[import-note]` §1.6.1. So "15-minute time window" is a nominal
setting, not the realised evaluation unit.

### 3.3 Node identity is `sha256(label)`, not the UUID

`[import-note]` The `node2id` key is a hash of the node *label*. CADETS E3: 613,713 uuid
rows collapse to 268,242 entities, ≈56% (`create_database.py:235-251`).

`[read]` Independently corroborated by a hardcoded constant in the model file:

```python
# model.py:6
max_node_num = 268243  # the number of nodes in node2id table +1
```

Consequence: **every process that ever ran `/usr/sbin/sshd` is one node, sharing one
memory slot and one neighbour list.** Process instance identity, pid lineage and
lifetime are gone before the model sees anything.

Any KAIROS statement about "stateful node memory" is a statement about memory *per label
class*, not per entity. The collapse rate is dataset-dependent (THEIA E3 keys on
`cmdLine,tgid,path` and retains most instance identity; OpTC keys on uuid and does not
collapse at all), which makes cross-dataset comparison of the memory's contribution
unsound as published. This is the deepest single finding about KAIROS.

---

### 3.4 β is hardcoded, not set from validation

`[paper]` `resources/KAIROS.txt:719-722`: "KAIROS uses benign validation data to set β
after model training."

`[read]` The released code does not do this. `config.py:144-145` sets
`beta_day6 = 100` and `beta_day7 = 100` as literals. `evaluation.py:100-119` *does*
compute the maximum queue anomaly score over the validation day (`graph_4_5`) — and then
only logs it (`:119`); the value is never assigned to β. The test-day comparisons at
`evaluation.py:142` and `:160` read the hardcoded constants.

Two consequences. The threshold is a **per-test-day** parameter — `beta_day6` and
`beta_day7` are independent knobs that merely happen to both be 100 — which is a degree
of freedom the paper does not describe. And the validation day does no threshold-setting
work at all in the artifact, so the reported numbers cannot be reproduced *via* the
documented procedure, only via the constants.

### 3.5 The node-IDF statistics are seeded on training days

`[read]` `test.py:146-156` replays 04-03, 04-04 and 04-05 before the test days, commented
"graph_4_3 - graph_4_5 will be used to initialize node IDF scores". Two of those three
(04-03, 04-04) are **training** days (`train.py:73-76` loads 04-02…04-04).

The IDF denominator is `len(file_list)`, the count of *those* windows
(`anomalous_queue_construction.py:83`), and the rareness test is
`IDF > log(len(tw_list) * 0.9)` (`:132`). The rareness threshold that gates suspicious
nodes is therefore calibrated on windows the model was trained on. `[paper]` Table 12
(`resources/KAIROS.txt:2514-2555`) lists 04-05 alone as validation for E3-CADETS and says
nothing about replaying training days.

---

## 4. Detection, thresholding and evaluation

`[import-note]` unless marked.

- 15-minute windows are applied at **test time only** (`test.py:115`, the sole reader of
  `time_window_size`; `train.py` has none); graphs are built per calendar day at
  US/Eastern midnight. Realised windows are ≥15 min and drift — median 15.48 min, max
  28.9 min across 454 logged THEIA E3 windows `[import-note]` §1.6.1.
- **179 windows** are scored for E3-CADETS (paper Table 4: TP 4 / TN 174 / FP 1 / FN 0)
  across two test days, against 192 for an exact 15-min grid.
- Anomaly threshold per window: `mean + 1.5·std`
  (`anomalous_queue_construction.py:29`).
- **Keyword denylist chosen from test data.** `cal_set_rel` zeroes the IDF of nodes
  matching a hardcoded list (`anomalous_queue_construction.py:93-113`), with the comment
  conceding: *"These nodes frequently exist in the testing data but don't contribute
  much to the detection."* Several entries are broad substrings (`usr`, `var`, `proc`,
  `cadet`), so suppression is far wider than the specific paths suggest. Two variants
  exist and are not equivalent — the CADETS E3 script floors the IDF while several
  notebooks skip matching nodes outright.
- **Ground truth is a hand-listed set of four 15-minute windows** for CADETS E3
  (`evaluation.py:47-52`), plus a keyword-based attack-edge counter over nine hardcoded
  indicators (`:60-70`). Detection is therefore scored at **time-window granularity**,
  not node or edge — this alone makes KAIROS's numbers non-comparable with node- or
  edge-level systems without recomputation.
- Loss-accumulation bug at `test.py:128-131`: the reported per-window loss is
  `(last_batch_loss + Σ edge losses) / event_count`. Affects the logged window loss, not
  the per-edge losses that drive detection.
- `[unverified]` How the paper's "adjusted" results table is produced. This is the one
  remaining material paper↔code delta not yet run down.

### Splits (CADETS E3) `[import-note]`

Train `graph_4_2..4_4`; IDF/validation `graph_4_3..4_5`; test `graph_4_6`, `graph_4_7`.
Chronological, unlike ORTHRUS — but note validation overlaps training days 3–4 (§3.5),
and the validation day never sets a threshold anyway (§3.4).

Training consumes **3 whole-day graphs**, not windows. First-party sizes exist only for
THEIA E3 `[import-note]` §3.3.1: 8.23M + 4.93M + 1.49M = **14.65M edges**, ≈14,300
batches of 1024 per epoch, ×50 epochs. Epoch count is *not* uniform across experiments —
50 for CADETS/THEIA E3, 30 for the other E3/E5 notebooks, 10 for OpTC (§3.3.2).

---

## 5. Hyperparameters `[read]` (`config.py`)

| Name | Value | Line |
| --- | --- | --- |
| `node_embedding_dim` | 16 | 102 |
| `node_state_dim` (memory dim) | 100 | 105 |
| `neighbor_size` | 20 | 108 |
| `edge_dim` | 100 | 111 |
| `time_dim` | 100 | 114 |
| `BATCH` | 1024 | 124 |
| `lr` | 0.00005 | 127 |
| `epoch_num` | 50 | 131 |
| `time_window_size` | 15 min (ns) | 133-135 |
| `beta_day6` / `beta_day7` | 100 / 100 | 144-145 |

No argparse, no YAML — every script reads these module-level constants directly.
`epoch_num` and `time_window_size` are CADETS E3 values; the notebooks inline their own
(§3.3.2 of the import note). `beta_day6`/`beta_day7` are the detection thresholds the
paper claims are derived from validation data (§3.4).

---

## 6. Positioning on the thesis axes

| Axis | KAIROS |
| --- | --- |
| Node memory | **yes** — `TGNMemory`, GRU, dim 100, **reset every day** |
| Time representation | **yes** — Fourier Δt encoder shared between memory and GNN |
| Node keying | `sha256(label)` — **collapses ~56% of entities on CADETS E3** |
| Encoder | TGN memory + UniMP (`TransformerConv` ×2) |
| Proxy task | edge-type prediction, 7 classes on CADETS E3 (paper says 9) |
| Granularity | scored at **time-window** level |
| Training unit | **whole calendar day** — windows play no part in training |
| Window | 15 min *nominal*, test-time only; realised ≥15 min, median 15.5, max 28.9 |
| Thresholding | `mean + 1.5·std` per window, plus a test-derived keyword denylist |

---

## 7. Open items

- How the "adjusted" results table is produced (§4).
- Whether the E5 / OpTC notebooks differ materially from the CADETS E3 scripted
  pipeline in *model* terms (the import note already covers their preprocessing).
- Quantify the keying collapse on our own lossless import, and report the THEIA rate for
  contrast — cheap, and it is the strongest single number in the chapter.
