# KAIROS — data import & preprocessing analysis

`RW7.P2` code pass over `related-work/kairos` (fork of `ProvenanceAnalytics/kairos`).
Scope: **what enters the model**, per experiment — node/edge construction, features,
exclusions, and any transformation applied on import that can move downstream results.

Provenance format: `.py` files as `path:line`; notebooks as `path` **cell [n]**
(n = index over *code* cells only, 0-based).

Experiments in the repo:

| Experiment | Driver | CDM / format |
| --- | --- | --- |
| CADETS E3 | `DARPA/CADETS_E3/*.py` (the only scripted, `make pipeline`-able one) | cdm18 |
| THEIA E3 | `DARPA/THEIA_E3/theia3_datapreprocess.ipynb` | cdm18 |
| CLEARSCOPE E3 | `DARPA/CLEARSCOPE_E3/clearscope3_datapreprocess.ipynb` | cdm18 |
| CADETS E5 | `DARPA/CADETS_E5/cadets5_datapreprocess.ipynb` | cdm20 |
| THEIA E5 | `DARPA/THEIA_E5/theia5_datapreprocess.ipynb` | cdm20 |
| CLEARSCOPE E5 | `DARPA/CLEARSCOPE_E5/clearscope5_datapreprocess.ipynb` | cdm20 |
| OpTC | `DARPA/OpTC/optc_datapreprocess.ipynb` | eCAR JSON |
| StreamSpot | `StreamSpot/src/preprocess.py` | TSV |

---

## 1. The shared DARPA pipeline

All six DARPA TC experiments run the same five stages. Only the gates, the label
fields and the edge-type lists differ.

1. **Select** a raw JSON line by a **substring test** (§1.1) — no JSON parsing, no
   schema validation.
2. **Extract** fields by positional **regex** (§1.3). Non-matching records are dropped
   silently (`except: pass`).
3. **Key** each node on `sha256(label)` — not on its UUID (§5.1).
4. **Index**: `node2id` assigns a dense `index_id` in insertion order, always
   files → subjects → netflows. That integer is the TGN memory slot.
5. **Vectorize** per calendar day into one `TemporalData`, filtering edges by type (§1.4).

### 1.1 Record selection — every gate, verbatim

The gate is a Python substring test against the raw line. Full set:

| Experiment | Target | Exact condition | Where |
| --- | --- | --- | --- |
| CADETS E3 | netflow | `"NetFlowObject" in line` | `create_database.py:36` |
| CADETS E3 | subject | `"Event" in line` | `create_database.py:77` |
| CADETS E3 | file (pass 1, uuids) | `"com.bbn.tc.schema.avro.cdm18.FileObject" in line` | `create_database.py:105` |
| CADETS E3 | file (pass 2, paths) | `'{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line` | `create_database.py:116` |
| CADETS E3 | event | `'{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line and "EVENT_FLOWS_TO" not in line` | `create_database.py:203` |
| THEIA E3 | netflow | `'{"datum":{"com.bbn.tc.schema.avro.cdm18.NetFlowObject"' in line` | cell [8] |
| THEIA E3 | subject | `'{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' in line` | cell [11] |
| THEIA E3 | file | `'{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject"' in line` | cell [19] |
| THEIA E3 | event | `'{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line and "EVENT_FLOWS_TO" not in line` | cell [27] |
| CLEARSCOPE E3 | netflow | `"NetFlowObject" in line` | cell [5] |
| CLEARSCOPE E3 | subject | `"schema.avro.cdm18.Subject" in line` | cell [8] |
| CLEARSCOPE E3 | file | `"com.bbn.tc.schema.avro.cdm18.FileObject" in line` | cell [10] |
| CLEARSCOPE E3 | event | `'{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line and "EVENT_FLOWS_TO" not in line` | cell [17] |
| CADETS E5 | netflow | `"avro.cdm20.NetFlowObject" in line` | cell [7] |
| CADETS E5 | subject + file (uuid registration) | `if "schema.avro.cdm20.Subject" in line: … elif "schema.avro.cdm20.FileObject" in line:` | cell [10] |
| CADETS E5 | subject label | `"schema.avro.cdm20.Event" in line` **and** `relation_type in include_edge_type` | cell [12] |
| CADETS E5 | file label | `"schema.avro.cdm20.Event" in line` **and** `relation_type in include_edge_type` | cell [16] |
| CADETS E5 | event | `'{"datum":{"com.bbn.tc.schema.avro.cdm20.Event"' in line` | cell [25] |
| THEIA E5 | netflow | `"NetFlowObject" in line` | cell [6] |
| THEIA E5 | subject | `"schema.avro.cdm20.Subject" in line` | cell [9] |
| THEIA E5 | file | `"avro.cdm20.FileObject" in line` | cell [12] |
| THEIA E5 | event | `'{"datum":{"com.bbn.tc.schema.avro.cdm20.Event"' in line` | cell [21] |
| CLEARSCOPE E5 | netflow | `"avro.cdm20.NetFlowObject" in line` | cell [5] |
| CLEARSCOPE E5 | subject | `"schema.avro.cdm20.Subject" in line` | cell [8] |
| CLEARSCOPE E5 | file | `"avro.cdm20.FileObject" in line` | cell [10] |
| CLEARSCOPE E5 | event | `'{"datum":{"com.bbn.tc.schema.avro.cdm20.Event"' in line` | cell [18] |
| OpTC | all | `temp_dic['object'] in node_type_used and is_selected_hosts(hostname)` (real `json.loads`) | cells [18], [22] |

Consequences of gating on substrings:

- **CADETS E3 subjects are scraped from `Event` records, not `Subject` records.** The gate
  is the bare string `"Event"`, which also matches any line where `Event` occurs in a
  path or argument. Process identity comes from the `"exec"` field carried *on the event*.
- **CADETS E3 file paths likewise come from `Event` records** (`predicateObjectPath`), not
  from the `FileObject` record. `FileObject` records are read once, only to collect the
  set of valid uuids.
- **`EVENT_FLOWS_TO` is excluded by a whole-line substring test** in the three E3
  pipelines — a record mentioning that string in any other field is also dropped. The
  three **E5** pipelines have no such guard.
- **CADETS E5 uses `elif`** (cell [10]): a line matching `…Subject` never reaches the
  `FileObject` branch.
- **Non-anchored gates** (`"NetFlowObject" in line`, `"schema.avro.cdm20.Subject" in line`)
  match the type name anywhere in the record, including inside a nested reference, not
  only in the datum position. The anchored form `'{"datum":{"com…X"' in line` is used
  only by THEIA E3 and for the event gates.

### 1.2 CDM record types → the three node tables

`DARPA/settings/database.md` defines exactly three node tables for every DARPA
experiment. Mapping:

| Table | CDM record type | Columns kept | CDM subtypes collapsed into it |
| --- | --- | --- | --- |
| `subject_node_table` | `Subject` (cdm18/20) — via `Event.subject` in CADETS E3/E5 | CADETS/CLEARSCOPE: `node_uuid, hash_id, exec`⁄`cmdLine`; THEIA E3: `node_uuid, hash_id, cmdLine, tgid, path` | `SUBJECT_PROCESS`, `SUBJECT_THREAD`, `SUBJECT_UNIT`, `SUBJECT_BASIC_BLOCK` — **`Subject.type` is never read** |
| `file_node_table` | `FileObject` | `node_uuid, hash_id, path` | `FILE_OBJECT_FILE`, `_DIR`, `_NAMED_PIPE`, `_UNIX_SOCKET`, `_PEFILE`, `_BLOCK`, `_CHAR`, `_LINK` — **`FileObject.type` is never read** |
| `netflow_node_table` | `NetFlowObject` | `node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port` | — |

- The node **type token** that reaches the model is therefore 3-valued
  (`file` / `subject` / `netflow`), coarser than CDM's own typing. A directory, a named
  pipe and a regular file are indistinguishable; so are a process, a thread and a unit.
- **CDM record types never imported** (present in the cdm18/cdm20 datum union):
  `Principal`, `MemoryObject`, `SrcSinkObject`, `UnnamedPipeObject`, `IpcObject`,
  `RegistryKeyObject`, `PacketSocketObject`, `ProvenanceTagNode`, `TagRunLengthTuple`,
  `Value`, `CryptographicHash`, `UnitDependency`, `Host`, `TimeMarker`,
  `StartMarker`/`EndMarker`.
- Consequence: any event whose `predicateObject` is one of those types fails the
  "both endpoints must resolve" test and is dropped along with it.

### 1.3 Field extraction — the regexes

**netflow** — identical shape everywhere; only the CDM value-wrapper differs.

- cdm18 (CADETS E3, THEIA E3, CLEARSCOPE E3):
  `'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),'`
- cdm20 (all E5): same, with `{"string":"…"}` / `{"int":…}` wrappers.
- Captures → `srcaddr, srcport, dstaddr, dstport`. Identity = `sha256("src,sport,dst,dport")`;
  **label written to `node2id` = `dst_addr:dst_port` only** (`create_database.py:170`).

**subject**

| Experiment | Regex | Label |
| --- | --- | --- |
| CADETS E3 | `'"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}(.*?)"exec":"(.*?)"'` | `exec` |
| THEIA E3 | `'Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"properties":{"map":{"tgid":"(.*?)"'` + `'"path":"(.*?)"'` | identity `cmdLine,tgid,path`; **feature `path`** |
| CLEARSCOPE E3 | `'Subject":{"uuid":"(.*?)",(.*?)"cmdLine":{"string":"(.*?)"}'` | `cmdLine` |
| CADETS E5 | `'"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"},(.*?)"exec":"(.*?)",'` | `exec` |
| THEIA E5 | `'avro.cdm20.Subject":{"uuid":"(.*?)",(.*?)"path":"(.*?)"'` | `path` |
| CLEARSCOPE E5 | `'avro.cdm20.Subject":{"uuid":"(.*?)",(.*?)"cmdLine":{"string":"(.*?)"}'` | `cmdLine` |

**file**

| Experiment | Regex | Label |
| --- | --- | --- |
| CADETS E3 | `'"predicateObjectPath":{"string":"(.*?)"'` (on Event lines) | `predicateObjectPath` |
| THEIA E3 | `'FileObject":{"uuid":"(.*?)"(.*?)"filename":"(.*?)"'` | `filename` |
| CLEARSCOPE E3 | `'FileObject":{"uuid":"(.*?)",(.*?)"path":"(.*?)"'` | `path` |
| CADETS E5 | `'"predicateObjectPath":{"string":"(.*?)"}'` (on Event lines) | `predicateObjectPath` |
| THEIA E5 | `'avro.cdm20.FileObject":{"uuid":"(.*?)",(.*?)"filename":"(.*?)"'` | `filename` |
| CLEARSCOPE E5 | `'cdm20.FileObject":{"uuid":"(.*?)",(.*?){"map":{"path":"(.*?)"'` | `properties.map.path` |

**event** — same four regexes in every experiment (cdm18/cdm20 differ only in the
namespace, and THEIA E3/E5 omit the trailing `}`):

- `'"subject":{"com.bbn.tc.schema.avro.cdmNN.UUID":"(.*?)"}'` → src uuid
- `'"predicateObject":{"com.bbn.tc.schema.avro.cdmNN.UUID":"(.*?)"}'` → dst uuid
- `'"type":"(.*?)"'` → edge type. **First match in the line**, not a keyed lookup.
- `'"timestampNanos":(.*?),'` → `t`

Everything else on the event record is discarded: `size`, `sequence`, `threadId`,
`programPoint`, `predicateObject2` (so the second endpoint of two-object events is lost),
`properties`, `hostId`, `location`, `name`, `parameters`.

### 1.4 The edge-type lists

Three separately-maintained lists, and they do not always agree:

- `edge_reversed` / `reverse` — types whose direction is flipped at **import** so data
  flows object → subject.
- `include_edge_type` — the type filter.
- `rel2id` — the one-hot vocabulary, and thus the **classifier's label set**.

Per experiment (see §3.2 for the full matrix):

- CADETS E3: filter = `include_edge_type` (7) at vectorization; `rel2id` also 7. ✔ consistent.
- THEIA E3, CLEARSCOPE E3, THEIA E5, CLEARSCOPE E5: filter = `e[2] in rel2id`, i.e. the
  one-hot vocabulary *is* the filter.
- CADETS E5: `include_edge_type` (9) is used **twice** — once to decide which events may
  supply node labels (§2.4), once as the graph filter; `rel2id` matches it.
- CLEARSCOPE E5: `filter_type` (10) is applied at **DB insert**; `include_edge_type` is
  dead code.
- THEIA E5: `include_edge_type` (cell [19]) is dead code; `rel2id` (cell [29]) is the
  real filter, and the two differ (`CLOSE` vs `CONNECT`).

### 1.5 Features — the complete inventory

Everything the model sees is in a `TemporalData` object with exactly four tensors:

| Tensor | Dim | Source | Construction | Meaning |
| --- | --- | --- | --- | --- |
| `src`, `dst` | 1 each, `int64` | `node2id.index_id` | dense index, insertion order | TGN memory slot + `LastNeighborLoader` index |
| `t` | 1, `int64` | `Event.timestampNanos` | verbatim | event time (ns; OpTC ms; StreamSpot row serial) |
| `msg[0:16]` | 16, `float` | `node2id.msg` of **src** | §1.5.1 | hashed src label |
| `msg[16:16+k]` | k, `float` | `event_table.operation` | `F.one_hot` over `rel2id` | edge type, k = 7…10 |
| `msg[-16:]` | 16, `float` | `node2id.msg` of **dst** | §1.5.1 | hashed dst label |

- **There is exactly one raw feature per node: its label string.** No numeric attribute
  of any entity is featurized.
- **Node features are static.** `node2higvec` is computed once over the whole corpus and
  indexed by `index_id`; a node's 16 dims are identical in every event, on every day. All
  temporal variation lives in the TGN memory, not in the inputs.
- Sizes come from `config.py`: `node_embedding_dim=16`, `node_state_dim=100` (memory),
  `time_dim=100`, `edge_dim=100` (GNN output), `neighbor_size=20`.

Derived at model time, not stored:

- `y_true` — the classification target, recovered from `msg[16:-16]` by
  `tensor_find(...) - 1` (`train.py:53-55`).
- `rel_t = last_update[src] - t`, fed to TGN's 100-d cosine time encoder (`model.py:28-29`).

#### 1.5.1 What a node's *label* is, and how it becomes sixteen numbers

**A node's label is its one and only raw attribute.** It is a single string held in
`node2id.msg` — column 3 of `node2id(hash_id, node_type, msg, index_id)`
(`DARPA/settings/database.md`) — written once by `create_node_list`
(`create_database.py:135-186`) and read back by `gen_nodeid2msg`
(`kairos_utils.py:117-126`) as `nodeid2msg[index_id] = {node_type: label}`. Everything
else the CDM record carried is discarded at this point (§1.5.4).

| Node type | What the label string *is* | Typical value | Source field |
| --- | --- | --- | --- |
| `subject` | the process's executable or command string | `sshd`, `/usr/bin/curl`, `com.android.chrome` | `exec` / `path` / `cmdLine`, per experiment (§1.3) |
| `file` | the object's path or file name | `/etc/passwd`, `libc.so.6` | `predicateObjectPath` / `path` / `filename`, per experiment (§1.3) |
| `netflow` | the **remote endpoint only**, `dst_addr + ":" + dst_port` | `128.55.12.10:80` | `remoteAddress`, `remotePort` (`create_database.py:170`) |

Three properties of the label that propagate into everything downstream:

- **The label is also the identity key.** `hash_id = sha256(label)`
  (`create_database.py:23-27`), and `node2id` is keyed on `hash_id`, so two entities with
  the same label *are the same node* — one row, one `index_id`, one TGN memory slot
  (§5.1). The label therefore fixes both *who* a node is and *what the model sees* of it.
- **Except in the two places where identity and label diverge.** netflow identity is
  `sha256("src,sport,dst,dport")` while the label drops `src` entirely; THEIA E3 subject
  identity is `sha256("cmdLine,tgid,path")` while the label is `path` alone (§5.6). In
  those two cases distinct nodes routinely carry byte-identical labels, hence
  byte-identical features.
- **A failed extraction becomes a literal string, not a missing value.** CADETS E3,
  CLEARSCOPE E3 and THEIA E5 write the four characters `null` when the subject regex does
  not match (`create_database.py:85`); every such process collapses into *one* node
  labelled `null` with one feature vector. CADETS E5 is the exception: it seeds
  `subject_uuid2path[uuid] = 'none'` and drops entries that never got overwritten
  (cell [14]).

**From label to vector — the six steps** (`embedding.py:48-77`; the notebooks inline the
same code: THEIA E3 cell [30], CLEARSCOPE E3 cell [21], CADETS E5 cell [30], THEIA E5
cell [26], CLEARSCOPE E5 cell [25]):

| # | Step | Input → output |
| --- | --- | --- |
| 1 | look up the label | `index_id` → `nodeid2msg[i]` → `{node_type: label}`; one `str` |
| 2 | seed with the type token | `higlist = [node_type]`, the **literal string** `'file'` / `'subject'` / `'netflow'`. The type is *not* a separate one-hot dimension — it is 4–7 characters prepended to the label |
| 3 | hierarchical prefix expansion | label → list of cumulative prefixes. `path2higlist` splits on `/`, `ip2higlist` on `.`, `subject2higlist` on `.` or `/` depending on the dataset (§1.5.3). `'/etc/passwd'` → `['', '/etc', '/etc/passwd']`; `'128.55.12.10:80'` → `['128', '128.55', '128.55.12', '128.55.12.10:80']` |
| 4 | `list2str` | the whole list concatenated into **one string with no separator**. Token boundaries are destroyed here: `['file', '', '/etc', '/etc/passwd']` → `'file/etc/etc/passwd'` |
| 5 | `FeatureHasher(n_features=16, input_type="string").transform([s])` | one string → a 16-wide sparse row. With the pinned scikit-learn this hashes **characters**, not tokens (§1.5.2) |
| 6 | `np.array(...).reshape([-1,16])` | one row per node, appended in `nodeid2msg` iteration order, which is ascending `index_id` because `gen_nodeid2msg` runs `ORDER BY index_id`. Saved as `node2higvec`, indexed positionally by `index_id` |

Worked examples (reproduced by `code/scratchpad/kairos_feature_hasher.py` in the thesis
umbrella repo, which reimplements steps 3–5 without sklearn):

| Label | `higlist` | `list2str` | `node2higvec[i]` |
| --- | --- | --- | --- |
| file `/etc/passwd` | `['file', '', '/etc', '/etc/passwd']` | `'file/etc/etc/passwd'` (19 ch) | `[0,-3,1,-1,0,-1,-1,3,0,0,0,1,1,0,0,1]` |
| subject `sshd` | `['subject', 'sshd']` | `'subjectsshd'` (11 ch) | `[1,-1,0,0,0,0,0,0,0,0,0,0,0,-2,0,-3]` |
| subject `/usr/bin/curl` | `['subject', '', '/usr', '/usr/bin', '/usr/bin/curl']` | `'subject/usr/usr/bin/usr/bin/curl'` (32 ch) | `[5,2,0,-1,-2,-2,0,0,0,0,0,0,1,-3,0,2]` |
| netflow `128.55.12.10:80` | `['netflow', '128', '128.55', '128.55.12', '128.55.12.10:80']` | `'netflow128128.55128.55.12128.55.12.10:80'` (40 ch) | `[0,-3,0,0,5,0,1,7,0,-5,0,1,1,-13,0,0]` |

Two step-3 details worth recording. The leading `''` that `path2higlist` emits for an
absolute path is **inert** — it contributes no characters in step 5, so the "hierarchy"
of `/etc/passwd` is effectively two elements, not three. And a label with no split
character (`sshd`, most CADETS `exec` values) expands to a one-element list, so the
hierarchical step is a no-op and the feature is just the character histogram of
`'subject' + label`.

**What the sixteen resulting features actually are.** They are *not* sixteen named
attributes. There is exactly one input — the expanded label string `s` from step 4 — and
feature `j` is the signed count of the characters of `s` whose hash lands in bin `j`:

```
f_j = Σ_{c ∈ chars(s)}  sign(h(c)) · [ |h(c)| mod 16 == j ],   h = MurmurHash3-x86-32(c, seed=0)
```

Because `h` is evaluated per character and the contributions are summed, the whole vector
is a **fixed linear map of the label's character histogram**: `f = M · count(s)`, where
`M ∈ {-1,0,+1}^{16×|alphabet|}` is a constant. The bin-and-sign assignment over the
characters that dominate real labels:

| Feature (bin) | Characters that feed it, with sign |
| --- | --- |
| 0 | +`6` +`u` |
| 1 | -`-` -`0` +`9` -`c` +`r` -`w` |
| 2 | +`a` |
| 3 | +`_` +`d` +`o` -`t` |
| 4 | +`3` +`5` -`n` |
| 5 | -`i` |
| 6 | +`:` -`p` +`y` |
| 7 | +`2` +`e` -`j` |
| 8 | -`4` +`7` -`q` |
| 9 | -`8` -`z` |
| 10 | -`g` |
| 11 | +`f` -`v` +`x` |
| 12 | +`l` +`m` |
| 13 | -`.` -`1` -`b` -`h` |
| 14 | — unreachable from `[a-z0-9/.:_-]`; only `A`, `O`, `U`, `'`, `=` reach it |
| 15 | +`/` -`k` -`s` |

So, concretely: *feature 15 is (count of `/`) − (count of `k`) − (count of `s`)*, feature
2 is the count of `a`, feature 13 is minus the count of `.`,`1`,`b`,`h`. Eight of the
fifteen reachable bins mix both signs, so characters inside a bin cancel; for
lowercase-only labels feature 14 is identically zero.

The node type enters as a constant offset — the characters of the type token, added once:

| Type token | Its contribution to every node of that type |
| --- | --- |
| `file` | `[0,0,0,0,0,-1,0,1,0,0,0,1,1,0,0,0]` |
| `subject` | `[1,-1,0,-1,0,0,0,0,0,0,0,0,0,-1,0,-1]` |
| `netflow` | `[0,-1,0,0,-1,0,0,1,0,0,0,1,1,0,0,0]` |

These are not orthogonal and are not separable from the label's own contribution, so the
model cannot read node type off the feature vector.

**Dtype and scale.** `node2higvec` is float64 out of sklearn; `torch.cat` promotes it
against the int64 one-hot and the `TemporalData` is cast to `float32`
(`embedding.py:117,126`). No normalisation is applied anywhere:
`torch_geometric.transforms.NormalizeFeatures` and `sklearn.preprocessing` are imported
in every preprocessing notebook and **never called**, and `train.py`/`test.py` contain
no scaling. The values reaching TGN are therefore raw small signed integers whose
magnitude grows with label length (see §1.5.2).

#### 1.5.2 Step 5 hashes *characters*, not tokens — verified

With the pinned `scikit-learn==1.2.0` (`DARPA/settings/requirements.txt:32`),
`FeatureHasher(input_type="string").transform([s])` where `s` is a `str` iterates it
**per character**, because a `str` is itself an iterable of strings. Each character is
hashed with `murmurhash3_bytes_s32(c.encode(), 0)`, placed in bin `|h| mod 16`, and
accumulated with sign `sign(h)` (`alternate_sign=True` is the default). The resulting
vector is a 16-bin signed character histogram:

```
list2str           : 'file/etc/etc/passwd'
KAIROS vector      : [ 0 -3  1 -1  0 -1 -1  3  0  0  0  1  1  0  0  1]
anagram path       : 'file/etc/etc/psaswd'
vector             : [ 0 -3  1 -1  0 -1 -1  3  0  0  0  1  1  0  0  1]   # identical
token-level vector : [ 2  0  1  1  0  0  0  0  0  0  0  0  0  0  0  0]   # the obvious reading
```

- Verified by reimplementation: `code/scratchpad/kairos_feature_hasher.py` (thesis
  umbrella repo) computes MurmurHash3-x86-32 directly, with no sklearn dependency, and
  reproduces the vectors above exactly. Run it to regenerate every number in §1.5.1–1.5.2.
- `environment-settings.md:23` confirms the pin is load-bearing:
  *"We encountered a problem in feature hashing functions with version 1.2.2"* — 1.2.2
  added the guard that rejects a bare string.

**What survives.** Character composition; label length; and path depth as a *magnitude*,
because the prefix expansion repeats each component once per level below it.

**What is lost.**

- *Component boundaries and order.* Any two labels that are character anagrams of their
  expanded forms are indistinguishable. This is not hypothetical at the scale of a
  process table: over the 1278 executable names on the machine this analysis ran on — the
  exact shape of a CADETS `exec` label — 38 (3.0 %) share a vector with another name, e.g.
  `bash`/`hash`/`sha1`, `ln`/`nl`/`nm`, `dd`/`od`/`rdoc`, `as`/`atos`/`sa`,
  `hdiutil`/`hidutil`/`bioutil`. (Local-filesystem proxy: the DARPA `node2id` tables were
  not reachable from this checkout, so this bounds the effect rather than measuring it on
  the real corpus.)
- *Digit order in netflow labels.* `10.0.0.1:443` and `10.0.0.1:434` hash identically.
  Distinct octet orderings usually survive because the prefix expansion repeats them at
  different multiplicities, but port permutations do not.
- *Discriminative weight.* The prefix expansion repeats the **shallowest** component the
  most: for `/usr/lib/systemd/system/foo.service` the leading `/usr` supplies 22 % of the
  90 characters hashed while the basename `foo.service` supplies 13 %. The feature is
  dominated by the least specific part of the path.
- *Scale invariance.* Vector magnitude grows super-linearly with depth — quadratically,
  for components of similar length. As `file` labels, `‖f‖₁` is 6, 12, 20, 30 for `/usr`,
  `/usr/lib`, `/usr/lib/x`, `/usr/lib/x/y`, and the netflow example in §1.5.1 already
  reaches −13 in feature 13. Since nothing normalises (§1.5.1), a deep path and a short
  one differ far more in norm than in direction.

**Consequence for a reimplementation.** Passing a token list (`FH.transform([higlist])`)
instead of the concatenated string produces a different, higher-rank feature space. It is
arguably the intended encoding, but it is *not* what produced the published numbers, and
a model trained on it is not comparable to KAIROS' released checkpoints. Any port of this
pipeline must state which of the two it uses.

#### 1.5.3 `subject2higlist` splits on a different character per dataset

| Experiment | Subject label | Split char | Effect |
| --- | --- | --- | --- |
| CADETS E3 | `exec` | `/` (`path2higlist`) | bare binary name → 1 element |
| THEIA E3 | `path` | `/` (`path2higlist`) | genuine path hierarchy |
| CLEARSCOPE E3 | `cmdLine` | `.` | Android package hierarchy (`com.android.x`) |
| CADETS E5 | `exec` | `.` | bare binary name → usually 1 element |
| THEIA E5 | `path` | `/` | genuine path hierarchy |
| CLEARSCOPE E5 | `cmdLine` | `/` | package-style label split on `/` — mostly 1 element |

CLEARSCOPE E3 and CLEARSCOPE E5 use the *same* label field with *different* split
characters, so their subject features are not constructed the same way.

#### 1.5.4 Not featurized anywhere

- Event: `size`, `sequence`, `threadId`, `predicateObject2`, `programPoint`, `properties`.
- Subject: pid/ppid (except THEIA E3's `tgid`, which enters identity but not features),
  user/principal, start time, parent link, `cmdLine` where the label is `exec`/`path`.
- File: CDM subtype, permissions, size, epoch.
- Netflow: `src_addr`/`src_port` (identity only), byte counts, protocol, IP version.
- Node degree, age, or any structural statistic.

#### 1.5.5 Is the edge-type label leaked into the input?

`msg` carries the edge-type one-hot in its middle segment, and the training target is
decoded from that same segment — which invites the conclusion that the answer is in the
input. It is not, for the edge being predicted:

- `link_pred` sees only `z[assoc[src]], z[assoc[dst]]` — no `msg` of the current edge
  (`train.py:49`).
- `memory.update_state(src, pos_dst, t, msg)` runs **after** `pos_out` is computed
  (`train.py:63`).
- `gnn(...)` is fed `train_data.msg[e_id]` — the messages of *neighbour* edges drawn from
  `LastNeighborLoader`, i.e. edges already seen.

⇒ the current edge's type reaches the model only through memory and neighbour state
written by **earlier** edges. Worth stating explicitly in `Q-MA6`, since the code shape
argues the opposite.

### 1.6 Windowing and memory scope

- Day graphs:
  `SELECT … WHERE timestamp_rec > start AND timestamp_rec < end ORDER BY timestamp_rec`
  — strict inequalities, boundaries at **US/Eastern midnight**
  (`kairos_utils.py:74`, `datetime_to_ns_time_US`).
- E3 loops `range(2, 14)` → 2018-04-02 … 04-13; E5 loops `range(8, 18)` →
  2019-05-08 … 05-17. Anything outside is dropped at vectorization.
- The 15-min analysis window is **test-time only**. `time_window_size` is defined once
  (`config.py:135`, `60000000000 * 15`) and read exactly once (`test.py:115`); `train.py`
  contains **zero** references to it. Training iterates a whole day-graph as one flat
  event stream (`train.py:37`, `seq_batches(batch_size=BATCH)`, `BATCH = 1024` at
  `config.py:124`). The notebooks inline the same constant, always inside the *test*
  function — THEIA E3 **cell [11]** l.119, CLEARSCOPE E3 **cell [14]** l.107,
  CADETS E5 **cell [18]** l.108 (`60000000000*15`); OpTC **cell [17]** l.99 uses
  `60000*15` because its timestamps are in **ms**, not ns.
- **Memory is reset per day-graph** in both `train()` (`train.py:33`) and `test()`
  (`test.py:40`). TGN memory never carries across days; within a day it carries across
  the 15-min windows. In training the reset fires once **per day-graph per epoch**
  (`train.py:117-126`).

⇒ the only structural discretisation during representation learning is the **calendar
day**. The 15-minute window is an evaluation/alerting granularity layered on a continuous
per-day event stream. This matters when positioning KAIROS as a "static time-window
snapshot" system: that label fits its *detection* stage, not its encoder, which consumes
the day as a continuous event stream. Worth re-checking against the ORTHRUS note before
drawing the contrast in the chapter.

#### 1.6.1 Windows are ≥15 min, snapped to batch boundaries

The boundary is tested once per 1024-event batch (`test.py:115`), and on flush
`start_time = t[-1]` (`test.py:140`) sets the next window's origin to the *actual*
overshoot point — so error accumulates rather than snapping back to a grid. A window
closes at the first batch end **after** 15 min have elapsed, and in a quiet period a
single batch can span hours.

Measured over the 454 windows logged in `theia3_graph_learning.ipynb` **cells [12]–[18]**:

| statistic | minutes |
| --- | --- |
| median | 15.48 |
| mean | 15.68 |
| max | 28.9 |
| within 1 s of 15:00 | 32 / 454 |
| > 16 min | 120 / 454 |

CADETS E3 is sparser and drifts much further. The hand-listed ground-truth attack windows
(`evaluation.py:47-52`, duplicated at `:80-85`) span:

```
11:18:26.126 ~ 11:33:35.116    15m 09s
11:33:35.116 ~ 11:48:42.606    15m 07s
11:48:42.606 ~ 12:03:50.186    15m 08s
12:03:50.186 ~ 14:01:32.489    1h 57m 42s   <- one "15-minute" window
```

Nothing downstream normalises for this: one `.txt` file is one unit of evaluation
regardless of the span it covers, so a window's weight in the metrics is independent of
its duration. Quoting "15-minute windows" as the evaluation unit is therefore accurate
only nominally.

---

## 2. Per-experiment detail

Gates and regexes are in §1.1/§1.3; this section is the per-experiment deltas.

### 2.1 CADETS E3 — `DARPA/CADETS_E3/` (the only scripted pipeline)

- Nodes: file, subject, netflow. Edges: **subject → (file ∣ netflow)** only. `msg` = 16+**7**+16 = 39.
- **Subjects come from Event records** (`create_database.py:68`), label = `exec`,
  the bare executable name. Regex failure → the literal label `"null"`. ≈224,146 rows.
- **File paths come from Event records** (`create_database.py:100`), gated by
  `'"predicateObjectPath":null,' not in line and '<unknown>' not in line` (line 121):
  - the `<unknown>` test is a **whole-line** substring test — an `<unknown>` in any other
    field suppresses the path;
  - **last matching event wins**;
  - files never referenced, or only with a null path, get no row — and every event
    touching them is dropped too;
  - `file_obj2hash[uuid]` stores the raw `findall` **list**, indexed `[0]` at line 128 —
    a line that passes the gate but has no `predicateObjectPath` key yields `[]` and raises.
- Netflow: ≈155,322 rows; file: ≈234,245 rows; `node2id`: 268,242 entities
  (`create_database.py:235-251`).
- Reversal: `ACCEPT, RECVFROM, RECVMSG` (`config.py:59`).
- Filter: `include_edge_type` = `WRITE, READ, CLOSE, OPEN, EXECUTE, SENDTO, RECVFROM`
  (`config.py:67`), applied at `embedding.py:105`.
- ⚠️ **`ACCEPT` and `RECVMSG` are reversed on import but absent from `include_edge_type`**
  — both are discarded at vectorization, so the reversal is dead code and accept/recvmsg
  semantics never reach the model.

### 2.2 THEIA E3 — `theia3_datapreprocess.ipynb`

- Edges: **subject → (file ∣ netflow ∣ subject)** — the only experiment keeping
  process→process edges. `msg` = 16+**9**+16 = 41.
- Subjects from real `Subject` records. Identity = `sha256(cmdLine,tgid,path)`;
  **`node2id.msg` = `path` only** (cell [22], `i[-1]`) → cmdline and tgid affect
  *identity* but not *features*; distinct nodes can carry byte-identical vectors.
- Least node collapse of the six (identity includes `tgid`).
- `res = re.findall(...)[0]` is **not** inside the `try` (cell [11]) — a Subject line
  lacking `cmdLine` or `tgid` raises and aborts the loop.
- Files keyed on `filename`, not `path`; FileObjects without one are skipped.
- Endpoint resolution order in cell [27] is `subject2hash → file2hash → netobj2hash`,
  each overwriting the previous — a uuid in more than one map resolves to the **last** match.
- Reversal: `READ, READ_SOCKET_PARAMS, RECVFROM, RECVMSG`.
- Filter (cell [42]): `e[2] in rel2id` — `CONNECT, EXECUTE, OPEN, READ, RECVFROM,
  RECVMSG, SENDMSG, SENDTO, WRITE`. No `CLOSE`.

### 2.3 CLEARSCOPE E3 — `clearscope3_datapreprocess.ipynb`

- Edges: subject → (file ∣ netflow). `msg` = 16+**8**+16 = 40.
- Subject label = `cmdLine`. A `Subject` with null `cmdLine` fails both the `try` and
  the fallback → **the subject is dropped entirely**, and with it all its events.
- File label = `path`.
- `subject2higlist` splits the cmdline on `.` (Android packages) while files use `/`.
- Reversal: `ACCEPT, RECVFROM, RECVMSG`.
- Filter (cell [33]): `CLOSE, OPEN, READ, WRITE, RECVFROM, RECVMSG, SENDMSG, SENDTO`.

### 2.4 CADETS E5 — `cadets5_datapreprocess.ipynb`

- Edges: subject → (file ∣ netflow). `msg` = 16+**9**+16 = 41.
- **Two-pass registration, and the edge-type filter decides which nodes exist:**
  1. cell [10] registers every cdm20 `Subject` / `FileObject` uuid with placeholder `'none'`.
  2. cell [12] fills subject labels from `"exec"` on Event lines — **only for events whose
     type is in `include_edge_type`**.
  3. cell [16] fills file labels from `predicateObjectPath`, same restriction;
     missing path → `'null'`.
  4. cell [14] drops every subject still `'none'`; cell [17] drops files that are
     `'none'` **or** `'null'`.
- ⇒ a process or file participating only in e.g. `EVENT_CLONE` or `EVENT_MMAP`
  **does not exist as a node at all**. The node population, `max_node_num`, and the whole
  index assignment are a function of the edge-type list.
- No `EVENT_FLOWS_TO` guard (cell [25]). Reversal: `READ, RECVFROM, RECVMSG`.
- Raw volume recorded in the notebook: `total_event_count: 1193669198`.
- Filter (cell [42]): `CLOSE, OPEN, READ, WRITE, EXECUTE, RECVFROM, RECVMSG, SENDMSG, SENDTO`.

### 2.5 THEIA E5 — `theia5_datapreprocess.ipynb`

- Edges: subject → (file ∣ netflow). `msg` = 16+**9**+16 = 41.
- Subject label = `"path"` from the `Subject` record; records without a path are dropped.
- File label = `"filename"`.
- No `EVENT_FLOWS_TO` guard, no type filter at insert time (cell [21]).
  Reversal: `RECVFROM, RECVMSG, READ`.
- **Two type lists coexist**: `include_edge_type` (cell [19], has `CLOSE`, no `CONNECT`) is
  **never used**; the graph filter (cell [38]) uses `rel2id` (has `CONNECT`, no `CLOSE`).
  ⇒ `EVENT_CLOSE` is imported into Postgres but excluded from every graph.
- `subject2higlist` splits on `/`.

### 2.6 CLEARSCOPE E5 — `clearscope5_datapreprocess.ipynb`

- Edges: subject → (file ∣ netflow). `msg` = 16+**10**+16 = 42.
- Subject label = `cmdLine`; file label from `{"map":{"path":"…"`.
- **Only experiment filtering edge types before the DB insert.** cells [19]–[20] build
  `filter_type` and drop everything else *before* `insert into event_table`:
  `ACCEPT, CLONE, CLOSE, CREATE_OBJECT, EXECUTE, OPEN, READ, RECVFROM, SENDTO, WRITE`.
  ⇒ the Postgres `event_table` is **already filtered** — unlike every other dataset, it is
  not a faithful record of the log, and the edge-type set cannot be widened without
  re-parsing.
- `include_edge_type` (cell [17]) is dead code; `filter_type` wins.
- Reversal: `READ, RECVFROM, RECVMSG` — `ACCEPT` is **kept but not reversed**, the
  opposite of CADETS E3 / CLEARSCOPE E3.
- `subject2higlist` splits on `/` (CLEARSCOPE E3 splits the same field on `.`).

### 2.7 OpTC — `optc_datapreprocess.ipynb`

Structurally different from every DARPA TC pipeline.

- **Real JSON parsing** (`json.loads`), not regex — the only one.
- Schema: one `event_table(src_id, src_type, edge_type, dst_id, dst_type, hostname,
  timestamp, data_label)` + `nodeid2msg`. **No per-type node tables, no `node2id`.**
- **Node identity = raw uuid** — the only experiment that does not hash a label.
  `msg` = 16+**10**+16 = 42.
- Node types: `actorID` is always typed `PROCESS`; the object is typed by the record's
  `object` field, restricted to `node_type_used = {FILE, FLOW, PROCESS}` (cell [5];
  `SHELL` commented out). `MODULE, REGISTRY, TASK, THREAD, USER_SESSION` never enter.
- Labels are host-prefixed, `"<host>_@<path>"`:
  - process → `properties.image_path`
  - FILE → `properties.file_path`
  - FLOW → `"<direction>#<src_ip>:<src_port>-><dest_ip>:<dest_port>"`
  - a record missing the property makes `process_raw_dic` return `{}` (bare `except`) and
    the whole event is dropped.
- **PROCESS objects never get a label written** — only actors do. A process that is only
  ever an object (e.g. the target of a `TERMINATE`) has no `nodeid2msg` entry and every
  edge touching it is dropped at vectorization.
- **Host allow-list `is_selected_hosts` is redefined between the two import cells**:
  benign (cell [17]) = `0201, 0402, 0660, 0501, 0051, 0209`;
  evaluation (cell [21]) swaps `0209 → 0207`.
- **Timestamps are milliseconds** (`timestamp*1000 + int(ms)`), not nanoseconds; parsing
  assumes exactly a `-04:00` suffix and a 3-digit fraction.
- **Raw text modified before parsing**: `line = line.replace('\\\\','/')` — Windows
  separators rewritten to `/`.
- `reverse_edge_type = ["READ"]` is declared (cell [5]) and **never applied** — OpTC edges
  are never reversed.
- **Node indices are per-graph**: each `(day, host, data_label)` rebuilds
  `node_uuid2index` from 0. Memory is reset per graph, so it is self-consistent, but there
  is **no memory or index continuity across host-days**.
- Ground truth: `labels.csv` matched on `actorID`/`objectID`, restricted to actions in `edge2vec`.

### 2.8 StreamSpot — `StreamSpot/src/preprocess.py`

- Input is already an edge list; the TSV is inserted **verbatim**
  (`preprocess.py:35-45`). **No filtering, no exclusion, no relabelling.**
- Features are **one-hot only** — 8-dim node type, 26-dim edge type, no hashing, no paths
  (`preprocess.py:109-121`). `msg` = 8+26+8 = 42.
- **No timestamps exist.** `t` is the Postgres `_id` serial, i.e. row order
  (`preprocess.py:140`, comment: *"Use logical order of the event to represent the time"*).
- ⚠️ **Non-deterministic encoding**: `node_type` / `edge_type` are Python `set`s and the
  one-hot index is assigned by iteration order (`preprocess.py:47-107`, `112-121`). With
  string-hash randomisation the mapping differs between interpreter runs. It is internally
  consistent (one process writes all 600 graphs), but a re-run produces a *different*
  encoding, so the published pre-trained model is incompatible with freshly preprocessed data.
- Split (`train.py:29-34`, `test.py:148-263`): train on graphs 0, 100, 200, 400, 500;
  validate on 505 and 1–24/101–124/201–224/401–424/501–524; test includes 300–399 (attacks).

## 3. Cross-experiment comparison

### 3.1 What a node *is*

| Experiment | subject identity (hashed) | subject label (featurized) | file identity/label | netflow identity | netflow label |
| --- | --- | --- | --- | --- | --- |
| CADETS E3 | `exec` | `exec` | `predicateObjectPath` | `src,sport,dst,dport` | `dst:dport` |
| THEIA E3 | `cmdLine,tgid,path` | **`path` only** | `filename` | `src,sport,dst,dport` | `dst:dport` |
| CLEARSCOPE E3 | `cmdLine` | `cmdLine` | `path` | `src,sport,dst,dport` | `dst:dport` |
| CADETS E5 | `exec` | `exec` | `predicateObjectPath` | `src,sport,dst,dport` | `dst:dport` |
| THEIA E5 | `path` | `path` | `filename` | `src,sport,dst,dport` | `dst:dport` |
| CLEARSCOPE E5 | `cmdLine` | `cmdLine` | `path` (from `map`) | `src,sport,dst,dport` | `dst:dport` |
| OpTC | raw uuid | `host_@image_path` | `host_@file_path` | raw uuid | `host_@dir#ip:port->ip:port` |
| StreamSpot | raw id | one-hot type | one-hot type | — | — |

### 3.2 Edges

| Experiment | subj→subj? | `EVENT_FLOWS_TO` excluded | reversed types | type filter applied at | # types in one-hot |
| --- | --- | --- | --- | --- | --- |
| CADETS E3 | no | yes | ACCEPT, RECVFROM, RECVMSG | vectorization | 7 |
| THEIA E3 | **yes** | yes | READ, READ_SOCKET_PARAMS, RECVFROM, RECVMSG | vectorization | 9 |
| CLEARSCOPE E3 | no | yes | ACCEPT, RECVFROM, RECVMSG | vectorization | 8 |
| CADETS E5 | no | **no** | READ, RECVFROM, RECVMSG | node registration **and** vectorization | 9 |
| THEIA E5 | no | **no** | RECVFROM, RECVMSG, READ | vectorization | 9 |
| CLEARSCOPE E5 | no | **no** | READ, RECVFROM, RECVMSG (**not** ACCEPT) | **DB insert** | 10 |
| OpTC | yes (PROCESS objects) | n/a | **none** (declared, unused) | vectorization | 10 |
| StreamSpot | yes | n/a | none | none | 26 |

### 3.3 Time and split

| Experiment | day range | boundary tz | window (test) | train graphs | max_node_num |
| --- | --- | --- | --- | --- | --- |
| CADETS E3 | 2018-04-02…13 | US/Eastern | 15 min | 4-02, 4-03, 4-04 | 268,243 |
| THEIA E3 | 2018-04-02…13 | US/Eastern | 15 min | 4-03, 4-04, 4-05 | 828,398 |
| CLEARSCOPE E3 | 2018-04-02…13 | US/Eastern | 15 min | 4-04, 4-05, 4-06 | 172,724 |
| CADETS E5 | 2019-05-08…17 | US/Eastern | 15 min | 5-08, 5-09, 5-11 | 262,626 |
| THEIA E5 | 2019-05-08…17 | US/Eastern | 15 min | 5-08, 5-09 | 967,389 |
| CLEARSCOPE E5 | 2019-05-08…17 | US/Eastern | 15 min | 5-08, 5-09, 5-11 | 139,961 |
| OpTC | 2019-09-21…25 | Etc/GMT+4 | per-graph | six 9-22 host graphs | per-run `max+2` |
| StreamSpot | — (row order) | — | — | graphs 0/100/200/400/500 | 5,045,000 |

Note that the E3/E5 test days overlap the train days
(`CADETS_E3/test.py:146-156` re-runs 4-03…4-05 to seed the node-IDF statistics before
testing on 4-06 and 4-07).

The `window (test)` column is the *nominal* setting; realised windows are ≥15 min and
drift (§1.6.1).

#### 3.3.1 Graph sizes and window counts (THEIA E3)

`theia3_datapreprocess.ipynb` **cell [42]** preserves its run output, so the per-day sizes
are exact (events fetched → edges kept after the type filter of §1.4). The CADETS E3 `.py`
pipeline ships no logs, so these are the only first-party size numbers in the repo.

| Day | events | edges kept | role | test windows |
| --- | --- | --- | --- | --- |
| 04-03 | 9,799,359 | 8,230,837 | **train** | 54 |
| 04-04 | 6,672,049 | 4,930,304 | **train** | 93 |
| 04-05 | 2,127,824 | 1,489,011 | **train** | 47 |
| 04-09 | 712,059 | 685,635 | test | 33 |
| 04-10 | 6,389,285 | 6,274,151 | test (attack) | 43 |
| 04-11 | 7,511,085 | 7,285,220 | test | 91 |
| 04-12 | 7,293,214 | 7,024,937 | test | 93 |
| 04-13 | 3,895,125 | 3,759,064 | unused | — |

04-02 and 04-06…04-08 return 0 events and yield empty graphs.

Training is therefore **3 day-graphs, 14.65M edges**, ≈14,300 batches per epoch. Window
counts are the IDF loops' tqdm totals — one `.txt` file per window —
`theia3_graph_learning.ipynb` **cell [20]** (194 = 54+93+47 over 04-03…04-05),
**cell [21]** (43, 04-10), **cell [22]** (91, 04-11), **cell [23]** (93, 04-12); the
per-day split of the 194 is from the logged window lines in **cells [12]–[14]**. Day
04-03 has only 54 windows despite being the largest graph because its data starts at
10:02, not midnight — partial days are common and window counts are not comparable
across days.

#### 3.3.2 Epoch counts differ by experiment

`epoch_num = 50` (`config.py:131`) governs CADETS E3 only. THEIA E3 also uses 50
(`theia3_graph_learning.ipynb` **cell [10]**, `range(1, 51)`), but CLEARSCOPE E3
(**cell [10]**), CADETS E5 (**cell [13]**), THEIA E5 (**cell [12]**) and CLEARSCOPE E5
(**cell [10]**) all use **30** (`range(1, 31)`), and OpTC uses **10** (**cell [16]**) over
six per-host 9-22 graphs. Do not quote 50 as a system-wide figure.

#### 3.3.3 Evaluated window totals

Paper Table 4 (`resources/KAIROS.txt:1025-1060`) gives E3-CADETS TP 4 / TN 174 / FP 1 /
FN 0 → **179 windows** scored across the two test days (04-06, 04-07). A perfect 15-min
grid over two days would be 192; the shortfall is the drift of §1.6.1. TP = 4 matches the
four hand-listed attack windows exactly, confirming the ground truth is exhaustive rather
than sampled.

---

## 4. Catalogue of exclusions

**Entity types never imported (DARPA TC):** Principal, MemoryObject, SrcSinkObject,
UnnamedPipeObject, IpcObject, RegistryKeyObject, PacketSocketObject, Host, and all
Tag/provenance records.

**Attribute values dropped from every kept record:** event `size`, `sequence`,
`threadId`, `programPoint`, `predicateObject2` (and hence every two-object event's
second endpoint), `properties`, `hostId`, principal/user, and for subjects everything
outside the single chosen label field.

**Value-conditioned exclusions:**

| Rule | Where | Effect |
| --- | --- | --- |
| `'"predicateObjectPath":null,' not in line` | CADETS E3 `create_database.py:121` | file nodes without a path never created |
| `'<unknown>' not in line` | same | any record containing `<unknown>` **anywhere** suppresses the path |
| `if len(i) != 64` | all DARPA node inserts | selects uuid keys out of the bidirectional dict; a uuid of length 64 would be silently dropped |
| label `'none'` / `'null'` | CADETS E5 cells [14], [17] | unlabelled subjects/files dropped, along with all their edges |
| `re.findall(...)[0]` inside `try/except: pass` | netflow/file parsers everywhere | records whose field layout differs from the regex are dropped without count |
| subject with null `cmdLine` / `path` | CLEARSCOPE E3 cell [8], THEIA E5 cell [9] | subject dropped entirely |
| both endpoints must exist | all event parsers | any event touching an un-imported entity is dropped |
| `node_type_used`, `is_selected_hosts` | OpTC cells [5], [17], [21] | 3 of 9 object types, 6 of 500+ hosts |
| endpoint has no `image_path`/`file_path` | OpTC cell [5] + vectorization | edge dropped |

**Silent-failure surface.** Every parser uses a bare `except: pass` or an unbounded
`try`. `fail_count` is incremented in some notebooks but never compared against a
total, and never asserted. There is no record anywhere of how many raw records were
dropped, so import loss is unmeasured.

---

## 5. Findings that can change downstream outcomes

Ordered by expected impact on a TGN-vs-KAIROS comparison.

### 5.1 Nodes are keyed on labels, not uuids — massive entity collapse

The `node2id` key is `sha256(label)`. From the counts asserted in
`create_database.py:235-251`, CADETS E3 has 155,322 netflow + 224,146 subject +
234,245 file **uuid rows** → 613,713 — collapsing to **268,242 `node2id` entities**
(≈56% reduction).

Concretely, in CADETS E3 **every process that ever ran `/usr/sbin/sshd` is one node**,
sharing one TGN memory slot and one neighbour list. Process instance identity, pid
lineage and process lifetime are gone before the model sees anything. The severity
varies by dataset (§3.1): THEIA E3 keys on `cmdLine,tgid,path` and so retains most
instance identity; CADETS E3/E5 (`exec`) collapse the hardest; OpTC does not collapse
at all (uuid keys).

This is the single most important input to `Q-MA5` (what the memory is indexed on). Any
claim about "stateful node memory" in KAIROS is a claim about memory per *label class*,
not per entity — and that differs by dataset, which makes cross-dataset comparison of
the memory's contribution unsound as published.

### 5.2 Node features are character-level hashes (§1.5.2)

Verified against the pinned sklearn. Paths that are character anagrams are
indistinguishable; the hierarchical expansion only survives as magnitude. A
reimplementation that hashes path *tokens* — the obvious reading of the code — changes
the feature space and invalidates comparison with the published numbers and the
pre-trained models. Reproductions must pin `scikit-learn==1.2.0` (or replicate the
character iteration explicitly).

### 5.3 The edge-type list determines the node population (CADETS E5)

Node labels are filled only from events whose type is in `include_edge_type`
(cells [12], [16]), and unlabelled nodes are then deleted. Changing the edge-type list
in CADETS E5 silently changes which *nodes* exist, `max_node_num`, and the entire index
assignment. This is not true for the other datasets, where nodes come from entity
records — so "same config, different dataset" is not the same experiment.

### 5.4 Only half the graph is present in five of six DARPA experiments

Except THEIA E3, events are stored only when the object resolves to a **file or
netflow** node. Process→process events (`EVENT_CLONE`, `EVENT_FORK`, `EVENT_EXECUTE`
against a subject, `EVENT_CHANGE_PRINCIPAL`, …) never enter the graph, so the process
tree is absent and attack lineage exists only through shared files/sockets. THEIA E3
alone keeps subject→subject edges — one more reason its numbers are not comparable to
CADETS/CLEARSCOPE.

### 5.5 Direction semantics are inconsistent across experiments

`EVENT_ACCEPT` is reversed in CADETS E3 and CLEARSCOPE E3, kept-but-not-reversed in
CLEARSCOPE E5, and dropped entirely in CADETS E3's graphs (§2.1). `EVENT_READ` is
reversed in THEIA E3/E5 and both E5 CADETS/CLEARSCOPE, but **not** in CADETS E3 or
CLEARSCOPE E3 — where reads point process→file. The "information-flow direction"
convention therefore differs between datasets in the same paper.

### 5.6 Identity ≠ features in THEIA E3

Identity hashes `cmdLine,tgid,path`, features use `path` only. Distinct memory slots
receive byte-identical initial features. Fine for a memory-based model (that is
arguably the point), but it means a *stateless* baseline on the same data cannot tell
those nodes apart at all — which biases exactly the memory-vs-no-memory comparison this
thesis is about. Worth an explicit control.

### 5.7 `event_table` is not raw for CLEARSCOPE E5

The 10-type `filter_type` is applied before the insert (§2.6). Anyone re-deriving graphs
from the database for CLEARSCOPE E5 gets a pre-filtered corpus and cannot widen the
edge-type set without re-parsing the logs.

### 5.8 OpTC has no cross-graph node continuity

Per-`(day, host, label)` index spaces, memory reset per graph, no direction reversal,
millisecond timestamps. OpTC results are 24 independent short sequences, not a
continuous temporal graph.

### 5.9 StreamSpot has no time and a non-deterministic encoding

`t` is row order (§2.8). Any temporal claim on StreamSpot is a claim about the order the
rows happened to sit in the file.

---

## 6. Downstream filters that also shape the results

Not import-stage, but they operate on the imported data and belong in the same audit.

- **Hand-written node keyword filters.** `cal_set_rel` zeroes the IDF of nodes matching
  a hard-coded keyword list before the anomalous-queue construction
  (`CADETS_E3/anomalous_queue_construction.py:93-113`). The comment concedes these are
  chosen from the *test* data: *"These nodes frequently exist in the testing data but
  don't contribute much to the detection"*. The lists are per-dataset and per-notebook,
  e.g.:
  - CADETS E3: `netflow`, `/home/george/Drafts`, `usr`, `proc`, `var`, `cadet`,
    `/var/log/debug.log`, `/var/log/cron`, `/home/charles/Drafts`, `/etc/ssl/cert.pem`,
    `/tmp/.31.3022e`
  - THEIA E3: adds `null`, `/dev/pts`, `salt-minion.log`, `675`, `thunderbird`, `/bin/`,
    `/sbin/sysctl`, `/data/replay_logdb/`, `/home/admin/eraseme`, `/stat`
  - CLEARSCOPE E3: `glx_alsa_675`, `/data/system/`, `/storage/emulated/`,
    `/data/data/com.android`, `nz9885vc.default`
  - OpTC: ~40 entries including `svchost.exe`, `python.exe`, `rundll32.exe`, `Windows`,
    `Users`, `.dll`, `.tmp`
  - CLEARSCOPE E5 inlines the list as a chained `and ... not in i` condition
    (`clearscope5_graph_learning.py`, `cal_set_rel_bak`)

  Several are broad substrings (`usr`, `var`, `proc`, `675`, `Users`, `Windows`), so the
  suppression is much wider than the specific paths suggest. Note the CADETS E3 script
  version *keeps* keyword-matching nodes but floors their IDF, while several notebooks
  (`is_include_key_word_bak(i) is not True`) **skip** them outright — the two variants
  are not equivalent.
- **Anomaly threshold** `mean + 1.5·std` per window
  (`anomalous_queue_construction.py:29`).
- **Ground truth is a hand-listed set of four 15-min windows** for CADETS E3
  (`evaluation.py:47-52`), plus a keyword-based attack-edge counter using nine
  hard-coded indicators (`evaluation.py:60-70`). Detection is scored at
  *time-window* granularity, not node or edge.
- **Loss accumulation bug** in `test.py:128-131`: `loss` still holds the last batch's
  scalar when the per-window sum starts, and the reported window loss is
  `(last_batch_loss + Σ edge losses) / event_count`. It affects the logged
  per-window loss, not the per-edge losses that drive detection.

---

## 7. Reproduction note

The character-level feature-hashing finding (§1.5.2) was verified directly:

```
uv venv --python 3.9 && uv pip install "scikit-learn==1.2.0" "numpy<2"
```

then hashing `list2str(['file'] + path2higlist('/etc/passwd'))` as a bare `str`, and
comparing against a character-anagram path and against the token-list form. Script is
in the session scratchpad; it should be promoted to `code/` as the `RW7.P3` artifact
(one script, one assertion, citing the `environment-settings.md:23` pin).
