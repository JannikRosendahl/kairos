# KAIROS — objective function & memory lifecycle

Companion to [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md).
That note establishes *what* the modules are; this one answers **what is optimised**,
**what the memory carries across phase boundaries**, and **what the edge-type one-hot
inside the message means for the proxy task**.

Analysed at commit `fdac2948f408bf6bea894bc75fca528f1b0e3bc9` (2026-09-14).
Scripted pipeline is `DARPA/CADETS_E3/`; line references are to that directory unless
stated. Paper text: `resources/KAIROS.txt`.

**Provenance convention.** `[read]` = verified by reading the cited code. `[paper]` =
claim from the paper. `[import-note]` = from a companion note. `[unverified]` = inference
or argument, not confirmed by execution.

---

## 1. The objective

### 1.1 It is a 7-way edge-type classification, not link prediction

`[read]` The loss is plain multi-class cross-entropy over edge types:

```python
criterion = nn.CrossEntropyLoss()                                  # model.py:5
...
y_pred = torch.cat([pos_out], dim=0)                               # train.py:51
y_true = []
for m in msg:
    l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1   # train.py:54
    y_true.append(l)
loss = criterion(y_pred, y_true)                                   # train.py:60
```

- `out_channels = len(include_edge_type)` = 7 (`train.py:95`, `config.py:67-75`).
- The label is recovered by slicing the **middle 7 dimensions of the message vector**,
  i.e. the edge-type one-hot that was concatenated into `msg` at graph-build time
  (`embedding.py:116-117`). See
  [`graph-transformation-and-featurisation.md`](graph-transformation-and-featurisation.md) §3.5.

**No negative sampling exists.** `torch.cat([pos_out], dim=0)` at `train.py:51` is a
concatenation of a single tensor — a vestige of PyG's `examples/tgn.py`, which the header
comment cites as the source (`train.py:1-4`) and which builds `torch.cat([pos_out,
neg_out])` for binary link prediction. The class name `LinkPredictor` (`model.py:35`) is
vestigial for the same reason; it is an edge-type classifier `[read]`.

This is a material deviation from the TGN paper, whose self-supervised objective is
future-link prediction against sampled negatives (`resources/TGN.txt`). KAIROS keeps
TGN's *encoder* and replaces its *objective*. Worth saying plainly: comparisons that
treat KAIROS as "TGN applied to provenance" understate how much was changed.

### 1.2 Reconstruction error = per-edge cross-entropy

`[read]` At test time the same quantity is recomputed per edge rather than batch-averaged
(`model.py:60-64`):

```python
def cal_pos_edges_loss_multiclass(link_pred_ratio, labels):
    loss = []
    for i in range(len(link_pred_ratio)):
        loss.append(criterion(link_pred_ratio[i].reshape(1,-1), labels[i].reshape(-1)))
    return torch.tensor(loss)
```

called at `test.py:90`; each edge's loss is stored in a dict with its endpoints, type and
timestamp (`test.py:103-112`) and written out sorted descending (`test.py:134-137`).
So the paper's "reconstruction error" is exactly `-log p(true edge type)` `[read]`.

`[unverified, argued]` This bounds what an anomaly score can express. The score is high
only when the model assigns low probability to the observed *edge type*. An attack that
uses entirely ordinary operations — reads, writes, opens — on unusual targets produces a
low score, because the type is still predictable. The featurisation carries the target
identity into the embedding, so it is not strictly blind to it, but the *supervision
signal* is 7-way and type-only.

### 1.3 The task's own labels are in the model's input context

This is the sharpest finding in this note. `[read]`

- `msg` = `[feat(src) 16 | edge-type one-hot 7 | feat(dst) 16]`, 39 dims
  (`embedding.py:116-117`).
- The memory write takes the whole message: `memory.update_state(src, pos_dst, t, msg)`
  (`train.py:63`, `test.py:86`). With `IdentityMessage` (`train.py:84`) the raw message is
  passed through unchanged, so the GRU state absorbs the edge-type one-hot of every event
  the node participates in.
- The GNN's edge attributes are built from the *historical* neighbour messages:
  `edge_attr = torch.cat([rel_t_enc, msg], dim=-1)` (`model.py:30`), where `msg` is
  `train_data.msg[e_id]` — the messages of the last-20 neighbour edges (`train.py:48`,
  `test.py:69`).

So the decoder predicts the type of edge *e* from a context that explicitly contains the
types of the 20 most recent edges around both endpoints, plus a GRU summary of every
earlier edge type.

**The current edge's own label does not leak.** The ordering is strict: loss is computed
at `train.py:60` / `test.py:82`, and `memory.update_state` / `neighbor_loader.insert` run
*after*, at `train.py:63-64` / `test.py:86-87` `[read]`. This confirms §2 of the model
note.

`[unverified, argued]` But the *task* is still substantially easier than it looks.
Provenance edge types are strongly autocorrelated — a process reading a file emits long
runs of `EVENT_READ`; an open is followed by reads and a close. Predicting the next edge
type from the last twenty edge types is close to a Markov baseline on type sequences.
This matters for the thesis in two ways: it is a plausible reason why memory contributes
little in published ablations (the shortcut is available with or without it), and it is
an argument for the executable-classification proxy task, whose label is *not* present in
the message.

`[unverified, argued]` Combined with label-class keying, the interaction is worse: since
all processes named `sshd` share one memory slot
([`graph-transformation-and-featurisation.md`](graph-transformation-and-featurisation.md) §2.5),
that slot's GRU state is a running summary of *the edge types that anything named sshd
recently performed*. The "node state" is closer to a per-executable edge-type frequency
profile than to a per-entity behavioural history.

### 1.4 Class imbalance is not handled

`[read]` `nn.CrossEntropyLoss()` is constructed with no `weight` and default
`reduction='mean'` (`model.py:5`). There is no resampling, no class weighting, no focal
loss anywhere in the pipeline.

`[unverified]` The realised 7-class distribution cannot be derived from the code; it needs
the data. The consequence is directional and safe to state: with an unweighted mean CE,
the model is optimised towards the dominant types, and a rare-but-benign edge type yields
a high reconstruction error purely by being rare. Since the anomaly score *is* that error
(§1.2), rare edge types are structurally biased towards being flagged.

### 1.5 Optimisation details

| Item | Value | Location |
| --- | --- | --- |
| Optimiser | Adam over `memory ∪ gnn ∪ link_pred` parameters | `train.py:98-100` `[read]` |
| lr / eps / weight_decay | 5e-5 / 1e-8 / 0.01 | `config.py:127-129` `[read]` |
| Epochs | 50 | `config.py:131` `[read]` |
| Batch | 1024 events, `seq_batches` (chronological) | `config.py:124`, `train.py:37` `[read]` |
| BPTT | truncated per batch — `memory.detach()` | `train.py:68` `[read]` |

The memory module *is* trained: its GRU parameters are in the optimiser set. What is not
carried is its **state** (§2).

### 1.6 Two counting bugs in the test loop

`[read]`, both cosmetic for detection but worth knowing before quoting any logged number:

- `test.py:128-131` — the reported per-window loss is
  `(last_batch_loss + Σ per-edge losses) / event_count`. The variable `loss` still holds
  the final batch's mean CE from `:82` when the accumulation loop starts. `total_loss`,
  accumulated correctly at `:83`, is computed and then never used. Affects only the
  logged window loss, **not** the per-edge losses that drive detection.
- `test.py:62` — `total_edges += BATCH` adds the nominal batch size regardless of the
  actual batch length, so the logged edge count overshoots on the final short batch.

---

## 2. Memory lifecycle

### 2.1 Reset granularity

`[read]`

| Phase | Reset point | Effect |
| --- | --- | --- |
| Training | `memory.reset_state()` at `train.py:33`, top of `train()` | `train()` is called once per day-graph per epoch (`train.py:117-126`), so memory is zeroed **3 × 50 = 150 times** on CADETS E3 |
| Testing | `memory.reset_state()` at `test.py:40`, top of `test()` | once per day-graph, i.e. 5 times (`test.py:173-211`) |

`neighbor_loader.reset_state()` sits alongside both (`train.py:34`, `test.py:41`), so the
last-20-neighbour buffer also starts empty at every day boundary `[read]`.

**The memory never spans a calendar day, in either phase.** Combined with the fact that
training uses whole-day graphs and never windows, the 15-minute window plays no role
whatsoever in what the memory can accumulate.

### 2.2 Training state is saved, then discarded

`[read]` Training persists the module objects — which include `TGNMemory`'s state buffers:

```python
model = [memory, gnn, link_pred, neighbor_loader]      # train.py:130
torch.save(model, f"{models_dir}/models.pt")           # train.py:133
```

Testing loads them (`test.py:170`) and then immediately zeroes the state at `test.py:40-41`.

So the answer to the thesis's question is precise and slightly sharper than "KAIROS does
not consider carrying memory over": **the trained memory state is serialised to disk and
then thrown away on the first line of every test run.** Only the GRU *weights*, the
attention weights and the decoder transfer. The final state of training day 4 is
recoverable from `models.pt` but unused.

`[unverified]` Whether that state would even be meaningful is doubtful — it is the state
at the end of the last batch of day 4 of the 50th epoch, and epochs replay the same days,
so it encodes a day the test phase does not continue from.

### 2.3 Cold start

`[paper]` "When a new node appears in the graph, its state is initialized to a feature
vector with all zeros, because there is no historical information on the node"
(`resources/KAIROS.txt:561-564`). PyG's `TGNMemory.reset_state()` zeroes the memory matrix
and `last_update` for all `max_node_num` slots `[unverified]` — asserted from the API
contract, not read from an installed PyG here.

`[unverified, argued]` Every test day therefore begins with *every* node memoryless and an
*empty* neighbour buffer. A node needs at least one prior event that day before its state
is anything but zero, and up to 20 before the attention module has a full neighbourhood.
The CADETS E3 attack windows start at 11:18 (`evaluation.py:48`), roughly 11 hours into
the day-graph, so for the attack itself the warm-up is not the binding constraint — but
for the early windows of each test day the model is operating at or near its stateless
limit, and those windows are scored on the same footing as the rest.

### 2.4 Memory keeps updating at inference

`[read]` `test.py:86` calls `memory.update_state(...)` inside a `@torch.no_grad()`
function (`test.py:22`). Inference is stateful: the state evolves across the test day even
though no gradients flow. This is faithful to TGN and worth stating, because it means
KAIROS's test-time behaviour is order-dependent — replaying the same day's events in a
different order would produce different scores.

### 2.5 What this does to the "long-range dependency" claim

`[paper]` The paper's motivation for node state is that "Each node state is a feature
vector that describes the history of graph changes involving the node"
(`resources/KAIROS.txt:561-563`), and the FP discussion leans on it explicitly: "KAIROS
still considers these entities to be compromised, because KAIROS remembers the history of
their states" (`:1268-1271`).

`[unverified, argued]` Three code facts bound that claim, and they compose:

1. **Horizon ≤ one calendar day** — reset at every day-graph (§2.1), in both phases.
2. **Not per entity** — the state belongs to a label class, not an instance
   (§1.3, and `graph-transformation-and-featurisation.md` §2.5).
3. **Not carried into deployment** — training's accumulated state is discarded (§2.2).

What remains is a within-day, per-executable-class running summary. That is a real
mechanism and genuinely more temporal than ORTHRUS or GRASP — but it is not the
long-range, per-entity memory the prose implies, and the cross-window correlation the
paper relies on for multi-stage attacks is done *outside* the model, by the queue
construction ([`detection-thresholding-and-ground-truth.md`](detection-thresholding-and-ground-truth.md) §2).

---

## 3. Open items

- The realised 7-class edge-type distribution (§1.4) — needs data.
- Whether a Markov-on-edge-types baseline matches KAIROS's per-edge scores (§1.3). This
  would be the decisive test of the shortcut argument and is cheap on our own import;
  recorded, not scheduled here.
- PyG `TGNMemory.reset_state()` semantics read from an installed copy rather than assumed
  (§2.3).
