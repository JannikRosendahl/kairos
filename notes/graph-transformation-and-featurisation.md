# KAIROS — graph transformation & featurisation

Companion to [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md)
(the model) and [`kairos-data-import-analysis.md`](kairos-data-import-analysis.md)
(record selection). This note answers two questions the thesis section left open:
**what transformation is applied between raw records and the tensor the model sees**,
and **what the node feature actually is**.

Analysed at commit `fdac2948f408bf6bea894bc75fca528f1b0e3bc9` (2026-09-14).
Scripted pipeline is `DARPA/CADETS_E3/`; all line references are to that directory
unless stated. Paper text: `resources/KAIROS.txt`.

**Provenance convention.** `[read]` = verified by reading the cited code during this
analysis. `[measured]` = produced by running a deterministic probe (recorded below).
`[import-note]` = established in the companion import analysis. `[paper]` = claim from
the paper. `[unverified]` = inference or argument, not confirmed by execution.

---

## 1. Graph transformation — the short answer

**There is no edge fusion, no deduplication, and no synthetic node or edge anywhere in
the CADETS E3 pipeline.** Every surviving event record becomes exactly one edge. This is
the opposite of ORTHRUS, which collapses runs of identical operations unconditionally
(`orthrus/src/graph_construction/build_orthrus_graphs.py:176-203`).

Evidence for the negative `[read]`:

- `store_event` appends one row per matching log line and bulk-inserts them with
  `ex.execute_values(cur, sql, datalist, page_size=10000)` — a plain
  `insert into event_table values %s`, no `ON CONFLICT`, no `DISTINCT`
  (`create_database.py:198-229`).
- `gen_vectorized_graphs` selects `select * from event_table where timestamp_rec > … and
  < … ORDER BY timestamp_rec` and appends one `edge_temp` per row
  (`embedding.py:93-107`). No grouping, no run detection.
- Grep across `DARPA/CADETS_E3/` for `groupby`, `drop_duplicates`, `unique(` on edges,
  or any fusion/merge helper returns nothing relevant.

So KAIROS preserves event multiplicity and burst structure. A process reading a file
10,000 times produces 10,000 edges. **This is a genuine strength relative to ORTHRUS and
worth stating as such** — the temporal signal a memory module would exploit is still
present in the data at this stage.

## 2. What IS applied

| Transformation | Effect | Location |
| --- | --- | --- |
| Record-type filter | `EVENT_FLOWS_TO` dropped at parse time | `create_database.py:203` `[read]` |
| Endpoint filter | event kept only if subject is a known subject **and** object is a known file or netflow | `create_database.py:207` `[read]` |
| Edge-type allowlist | only 7 of the parsed types enter the graph | `embedding.py:105`, `config.py:67-75` `[read]` |
| Direction reversal | 3 types have src/dst swapped | `create_database.py:216-223`, `config.py:59-63` `[read]` |
| Day partition | one `TemporalData` per calendar day, US/Eastern | `embedding.py:90-92` `[read]` |
| Node identity collapse | uuid → `sha256(label)` | `create_database.py:48-52,93,128` `[read]` |

### 2.1 Process→process edges do not exist

`create_database.py:207` requires the predicate object to be in `file_uuid2hash` **or**
`net_uuid2hash`:

```python
if subject_uuid[0] in subject_uuid2hash and (predicateObject_uuid[0] in file_uuid2hash
                                             or predicateObject_uuid[0] in net_uuid2hash):
```

A subject is never a valid destination. Every `EVENT_EXECUTE`, fork- or clone-like event
whose object is another *process* is therefore silently dropped `[read]`. The CADETS E3
graph is bipartite-ish: subjects on one side, files and sockets on the other.

Consequence `[unverified, argued]`: process lineage — the parent/child chain that
carries the clearest signal of an intrusion spreading — is **not representable** in
KAIROS's CADETS E3 graph. The model can only see a process through the files and sockets
it touches. This also caps what node memory can encode: a compromised process cannot
propagate state to a child process, because no edge connects them.

### 2.2 Direction is semantic, not syntactic

`edge_reversed = ["EVENT_ACCEPT", "EVENT_RECVFROM", "EVENT_RECVMSG"]` (`config.py:59-63`)
and the swap at `create_database.py:216-223` put the socket on the source side for
inbound events, so edge direction tracks data flow rather than the CDM record's
subject/object roles `[read]`. Note `EVENT_ACCEPT` and `EVENT_RECVMSG` are in
`edge_reversed` but **not** in `include_edge_type`, so of the three reversals only
`EVENT_RECVFROM` ever reaches the graph `[read]`.

The graph is directed; no reverse edges are added (contrast GRASP, which is undirected).

### 2.3 The 7-type allowlist

`include_edge_type` (`config.py:67-75`) admits `EVENT_WRITE, EVENT_READ, EVENT_CLOSE,
EVENT_OPEN, EVENT_EXECUTE, EVENT_SENDTO, EVENT_RECVFROM` — seven types. The filter is
applied when the day graph is built (`embedding.py:105`), *after* the events were
inserted into Postgres, so `event_table` holds more types than the model ever sees.
`out_channels = len(include_edge_type)` = 7 (`train.py:95`).

The paper states the decoder predicts "each of the **nine** possible types"
(`resources/KAIROS.txt:540-542`) `[paper]`. Corroborates F5 in `PLAN_RW.md`.

### 2.4 Windowing and batching are not transformations of the graph

Day graphs are built with strict inequalities on the day boundary
(`embedding.py:96`), so an event landing exactly on midnight is dropped — negligible in
practice `[read]`. Windows are carved only at test time and only at 1024-event batch
boundaries; see [`detection-thresholding-and-ground-truth.md`](detection-thresholding-and-ground-truth.md) §1.

### 2.5 Node counts, and what 613,713 → 268,242 actually means

The three node tables are keyed by **uuid**; `node2id` is keyed by the **label hash**.
The authors' own inline comments give the counts `[read]`:

| Table | Rows | Key | Comment |
| --- | --- | --- | --- |
| `netflow_node_table` | 155,322 | uuid | `create_database.py:235` |
| `subject_node_table` | 224,146 | uuid | `create_database.py:239` |
| `file_node_table` | 234,245 | uuid | `create_database.py:243` |
| **sum** | **613,713** | | |
| `node2id` | **268,242** | `sha256(label)` | `create_database.py:247` |

So: **613,713 is the number of uuid-keyed node rows across the three type tables;
268,242 is the number of distinct label-hash entities they collapse into** — a 56.3%
reduction `[read]`. Corroborated by the hardcoded `max_node_num = 268243`
(`model.py:7`, comment "the number of nodes in node2id table +1").

Per-type keys, and therefore per-type collapse behaviour `[read]`:

| Type | Memory keyed on | Location |
| --- | --- | --- |
| netflow | `sha256(srcaddr,srcport,dstaddr,dstport)` | `create_database.py:48-52` |
| subject | `sha256(exec)` — the executable name only | `create_database.py:78-81, 93` |
| file | `sha256(path)` | `create_database.py:122-123, 128` |

A per-type breakdown of the collapse cannot be derived from the code alone — the counts
above are totals, and `node2id` merges the three namespaces before counting. Deriving it
needs the data: count `DISTINCT hash` per table. Cheap on our own lossless import;
recorded as an open item, not done here.

### 2.6 The `"null"` subject sink — an unreported collapse

`store_subject` extracts the `exec` field by regex over `Event` records. When the regex
fails, the subject's label is set to the literal string `"null"`
(`create_database.py:84-87`) `[read]`:

```python
except:
    try:
        subject_obj2hash[subject_uuid[0][0]] = "null"
    except:
        pass
    fail_count += 1
```

Because identity is `sha256(label)`, **every subject whose `exec` could not be parsed
collapses into one single entity** — one shared memory slot for an arbitrary set of
unrelated processes. `fail_count` is counted but never logged or asserted on
(`create_database.py:70,88`), so the size of this bucket is unknown from the code alone
`[read]`. Quantifying it needs the data.

`[unverified, argued]` This is the worst case of the keying problem: it is not just
"all `sshd` processes share a slot" but "all *unidentifiable* processes share a slot",
and the GRU state of that slot is a blend of whatever those processes did.

Files are treated differently: a file event whose `predicateObjectPath` is `null` or
contains `<unknown>` is **dropped entirely** rather than bucketed
(`create_database.py:121`) `[read]`.

---

## 3. Featurisation

### 3.1 What the node feature is

`gen_feature` (`embedding.py:48-77`) `[read]`:

1. Build the hierarchical prefix list — `path2higlist` splits on `/`, `ip2higlist` on `.`
   (`embedding.py:22-40`), and the node **type string** is prepended as a literal
   `'netflow'` / `'file'` / `'subject'` (`:56-66`).
2. Concatenate the whole list into one string, `list2str` (`:42-46`, called at `:67`).
3. `FH_string = FeatureHasher(n_features=node_embedding_dim, input_type="string")` with
   `node_embedding_dim = 16` (`:70`, `config.py:102`), then
   `vec = FH_string.transform([i]).toarray()` (`:73`).

### 3.2 It hashes CHARACTERS — and this matches the paper

`transform([i])` passes a **single string** as one sample. scikit-learn's
`input_type="string"` path iterates each sample, so iterating a `str` yields its
*characters*. The node feature is therefore a **signed 16-bin character histogram**.

This is not a bug. The paper specifies exactly this `[paper]`
(`resources/KAIROS.txt:466-476`):

> The i-th dimension of s' feature vector is computed by ϕi(s) = Σ_{j:h(sj)=i} H(sj)
> where **sj is a character in the substring**, h is a hash function that maps **each
> character** to one of the dimensions in the feature space, and H is another hash
> function that hashes a character to {±1}.

Verified `[measured]`, under KAIROS's own pinned `scikit-learn==1.2.0`
(`DARPA/settings/requirements.txt`, python 3.9):

```
concat str  : 'subject/usr/usr/sbin/usr/sbin/sshd'
as-written  : [ 4  2  0  0 -2 -2  0  0  0  0  0  0  0 -4  0 -2]
char-level  : [ 4  2  0  0 -2 -2  0  0  0  0  0  0  0 -4  0 -2]
token-level : [ 0 -1  0  0  0  0  0  0  0  1  0  0  0  0  1  0]
as-written == char-level : True
as-written == token-level: False
```

The paper defines Φ(a) as the **sum of the per-substring vectors** (`:477-479`) while the
code hashes the **concatenation**. These are mathematically identical for character-level
signed hashing, since each character contributes independently and addition commutes.
Verified `[measured]`:

```
file     /home/admin/clean    concat==sum_of_substrings: True
subject  /usr/sbin/sshd       concat==sum_of_substrings: True
netflow  128.55.12.233:80     concat==sum_of_substrings: True
```

**So featurisation is one of the places where code and paper agree.** Say so; the
chapter's contribution is the delta, and this is not one.

### 3.3 Consequences of a 16-bin character histogram

`[measured]` unless marked.

- **Order-insensitive.** Any two labels that are character anagrams receive *identical*
  vectors:
  ```
  /usr/bin/abc     vs /usr/bin/cba     -> identical: True
  /etc/passwd      vs /etc/sswdap      -> identical: True
  /tmp/vUgefal     vs /tmp/laefgUv     -> identical: True
  ```
  The third pair matters: `/tmp/vUgefal` is the actual CADETS E3 attack payload
  (`evaluation.py:61`). A renamed payload using the same letters is, at featurisation
  time, indistinguishable from the original.
- **Magnitude encodes path depth, not just content.** Because the hierarchy repeats every
  prefix, components are counted once per level:
  ```
  /a             L1=  6   max|component|= 1
  /a/b           L1= 10   max|component|= 3
  /a/b/c         L1= 16   max|component|= 6
  /a/b/c/d       L1= 24   max|component|=10
  /a/b/c/d/e     L1= 34   max|component|=15
  ```
  Deeper paths produce larger-magnitude vectors, and early path components are weighted
  most heavily (they appear in every prefix).
- **Node type is not a separable marker.** The literal `'subject'` / `'file'` / `'netflow'`
  prefix is mixed into the same 16 bins as the path characters; the same path under two
  type prefixes differs by only L1 = 9. The model cannot cleanly read node type off the
  feature — it has to infer it.
- **Exact collisions are rarer than the 16 dimensions suggest.** On a synthetic
  300-path Unix vocabulary, 300 distinct labels produced 300 distinct vectors (0%
  collision). The components are unbounded integer counts, not bits, so the space is
  larger than "16 dimensions" implies. The real loss is *structure*, not *capacity*.
- `[unverified, argued]` The representation is purely lexical: it carries no notion that
  `/usr/bin` is a directory or that `.166` is an octet. Combined with THESEUS's finding
  that a training-executable-name allowlist matches learned baselines
  (`resources/THESEUS.txt:15-40`), a lexical input feature is exactly what one would
  expect to make a model look good for non-behavioural reasons.
- `[unverified, argued]` The features are **non-learnable**: `node2higvec` is computed
  once and saved (`embedding.py:76`), never a `nn.Parameter`. The paper's answer to
  adversarial label manipulation is that "KAIROS' graph learning will update these
  initial feature vectors based on temporal and structural equivalence"
  (`resources/KAIROS.txt:484-489`) `[paper]` — but the *features* are never updated. What
  is learned is the memory state and the attention, which sit downstream of a frozen
  input. The distinction is worth making precisely rather than accepting the paper's
  phrasing.

### 3.4 Paper↔code note on |Φ| and |z|

- The paper's hyperparameter study justifies |Φ| = 16 by trading feature sparsity against
  "the probability of hash collision in hierarchical feature hashing"
  (`resources/KAIROS.txt:1640-1647`) `[paper]`. Consistent with the character scheme.
- The paper lists **edge embedding dimension |z| = 200** (`resources/KAIROS.txt:841-843`)
  `[paper]`, while `config.py:111` sets `edge_dim = 100` and
  `GraphAttentionEmbedding(out_channels=edge_dim)` makes `z` 100-dimensional
  (`model.py:19-22`) `[read]`. The 200 is recoverable only if |z| is read as the
  decoder's concatenated `lin_src`/`lin_dst` width (`model.py:38-39`, 100→200 each)
  `[unverified]`. Flagging as a probable notation mismatch, not a confirmed discrepancy.

### 3.5 Edge features

Edge type only, one-hot over the 7 admitted types (`embedding.py:79-87`) `[read]`. The
per-edge message is the concatenation

```
msg = [ node2higvec[src] (16) , rel2vec[type] (7) , node2higvec[dst] (16) ]   # 39 dims
```

(`embedding.py:116-117`) `[read]`. The paper describes the same construction: "Each edge
is encoded as a concatenation of the source and destination node's feature embedding
(§4.1) and the one-hot encoding of the edge type" (`resources/KAIROS.txt:573-576`)
`[paper]`.

**The edge-type one-hot inside `msg` is also the training label** — it is recovered by
slicing `msg[16:-16]` (`train.py:54`, `test.py:77`). What that implies for the objective
is analysed in [`objective-and-memory-lifecycle.md`](objective-and-memory-lifecycle.md) §1.3.

---

## 4. THEIA E3 contrast (brief)

THEIA E3 is notebook-based (`DARPA/THEIA_E3/theia3_datapreprocess.ipynb`), not scripted.
The material difference is the subject key: THEIA keys identity on
`cmdLine + "," + tgid + "," + path` rather than CADETS's bare `exec` `[import-note]`
(`kairos-data-import-analysis.md`). A command line including arguments and a thread-group
id is far closer to a per-instance identifier, so THEIA collapses much less than CADETS's
56.3%.

`[unverified, argued]` This makes the *meaning of node memory dataset-dependent within
one paper*: on CADETS a memory slot is an executable class, on THEIA it is close to a
process instance. Any cross-dataset statement about what memory contributes is therefore
comparing two different mechanisms.

---

## 5. Open items

- Per-node-type collapse rate (§2.5) and the size of the `"null"` subject bucket (§2.6) —
  both need the data; cheap on our own lossless import.
- Exact collision rate of the character histogram over the *real* CADETS E3 label
  vocabulary, as opposed to the synthetic vocabulary used in §3.3.
- Whether the E5/OpTC notebooks apply any fusion or dedup (not checked; CADETS E3 and
  THEIA E3 do not).
