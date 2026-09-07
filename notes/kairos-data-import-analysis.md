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

All six DARPA TC experiments follow the same five stages. Only the regexes,
the label fields and the edge-type lists differ.

1. **Parse.** Raw CDM JSON is read **line by line as text** and matched with
   `re.findall`. There is no JSON parsing and no schema validation; a record is
   selected by a substring test (`if "NetFlowObject" in line`) and its fields by a
   positional regex. A record whose regex does not match is silently dropped
   (bare `except: pass`).
2. **Three node tables only** — `file_node_table`, `subject_node_table`,
   `netflow_node_table` (`DARPA/settings/database.md`). Every other CDM entity type
   (Principal, MemoryObject, SrcSinkObject, UnnamedPipeObject, IpcObject, RegistryKey,
   Host, Tag/provenance records) is never imported.
3. **Node identity = SHA-256 of a label string.** `stringtomd5()` is named for MD5 but
   calls `hashlib.sha256` (`DARPA/CADETS_E3/create_database.py:21`). The digest of a
   *label* (path / exec / cmdline / 4-tuple) is the node key, and `node2id` is keyed on
   it. **UUIDs are therefore not node identity** — every entity sharing a label collapses
   into one node (see §5.1).
4. **Dense index.** `node2id` assigns `index_id` by insertion order, always
   files → subjects → netflows. This integer is the TGN memory slot.
5. **Vectorize per day.** One `TemporalData` per calendar day, edges filtered by type,
   `msg = [ src_feat(16) | edge_onehot(k) | dst_feat(16) ]`.

### 1.1 Node features

Identical in all six (`DARPA/CADETS_E3/embedding.py:23-77`, and the corresponding
notebook cells):

```python
higlist = [<type token>] + path2higlist(label)   # or ip2higlist for netflow
vec     = FeatureHasher(n_features=16, input_type="string").transform([list2str(higlist)])
```

`path2higlist('/etc/passwd')` → `['', '/etc', '/etc/passwd']`, prefixed with the
type token and **concatenated into one string** by `list2str`.

**Verified**: with the pinned `scikit-learn==1.2.0`
(`DARPA/settings/requirements.txt:32`), `FeatureHasher(input_type="string").transform([s])`
where `s` is a `str` iterates the string **per character**. The node feature is therefore
a 16-dim signed hash of a *character multiset*, not of the path components:

```
list2str           : 'file/etc/etc/passwd'
KAIROS vector      : [ 0 -3  1 -1  0 -1 -1  3  0  0  0  1  1  0  0  1]
anagram path       : 'file/etc/etc/psaswd'
vector             : [ 0 -3  1 -1  0 -1 -1  3  0  0  0  1  1  0  0  1]   # identical
token-level vector : [ 2  0  1  1  0  0  0  0  0  0  0  0  0  0  0  0]   # what it looks like it intends
```

`environment-settings.md:23` confirms this is load-bearing, not accidental:
*"We encountered a problem in feature hashing functions with version 1.2.2"* —
1.2.2 added the guard that rejects a bare string, so the pin preserves the
character-level behaviour. Any port that passes a token list instead produces a
**different feature space** and is not comparable to the published numbers.

Consequences: the hierarchical prefix expansion survives only as a *magnitude*
signal (deeper paths repeat their prefix characters more often); path structure,
component boundaries and component order are all discarded; anagram paths are
indistinguishable.

### 1.2 Edge features and the training target

`msg` carries the edge-type one-hot in its middle segment, and the training label is
recovered from that same segment (`train.py:53-56`, `tensor_find(m[16:-16], 1) - 1`).
This is *not* direct leakage for the edge being predicted: `link_pred` sees only
`z[src], z[dst]`, and `memory.update_state(..., msg)` runs **after** `pos_out` is
computed. The current edge's type reaches the model only through the memory/neighbour
state written by *earlier* edges. Worth stating explicitly in `Q-MA6`, since the
`msg`-contains-the-label shape invites the opposite conclusion.

### 1.3 Windowing and memory scope

- Day graphs: `SELECT ... WHERE timestamp_rec > start AND timestamp_rec < end`, with
  boundaries at **US/Eastern midnight** (`kairos_utils.py:74`, `datetime_to_ns_time_US`). Strict inequalities.
- E3 loops `range(2, 14)` → 2018-04-02 … 2018-04-13; E5 loops `range(8, 18)` →
  2019-05-08 … 2019-05-17. Anything outside is dropped at vectorization.
- The 15-min analysis window is a *test-time* construct only
  (`config.py:135`, `time_window_size = 60000000000 * 15`), applied at `test.py:115`.
- **Memory is reset per day-graph**, in both `train()` (`train.py:33`) and `test()`
  (`test.py:40`). TGN memory never carries across days; within a day it carries across
  the 15-min windows.

---

## 2. Per-experiment detail

### 2.1 CADETS E3 — `DARPA/CADETS_E3/`

| | |
| --- | --- |
| **Nodes** | file, subject (process), netflow |
| **Edges** | subject → (file ∣ netflow) only |
| **msg dim** | 16 + **7** + 16 = 39 |

**netflow** (`create_database.py:29`) — from any line containing `NetFlowObject`.
Identity string `srcaddr,srcport,dstaddr,dstport`; **node label written to `node2id` is
only `dstaddr:dstport`** (`create_database.py:170`). So two flows differing only in
source port are distinct nodes with *identical* features. ≈155,322 rows.

**subject** (`create_database.py:68`) — **not parsed from `Subject` records.** The uuid
and executable are scraped out of *Event* lines with
`'"subject":{...UUID":"(.*?)"}(.*?)"exec":"(.*?)"'`. Label = `exec`, i.e. the bare
executable name. On regex failure the label becomes the literal string `"null"`.
≈224,146 rows. No cmdline, no pid/tgid, no parent, no user.

**file** (`create_database.py:100`) — two passes. Pass 1 collects all `FileObject`
uuids. Pass 2 takes the path from *Event* lines, gated by:
`'"predicateObjectPath":null,' not in line and '<unknown>' not in line`
(`create_database.py:121`). ≈234,245 rows.
- The `<unknown>` test is a **substring test over the whole JSON line**, so an
  `<unknown>` anywhere in the record (any field) suppresses the path.
- Last matching event wins — a file whose path is recorded differently across events
  keeps the last one seen in file order.
- Files never referenced by an event, or only with a null path, get **no row**, and
  every event touching them is consequently dropped too.
- `file_obj2hash[uuid]` stores the raw `re.findall` **list**; `create_database.py:128`
  indexes `[0]`. A line that passes the gate but has no `predicateObjectPath` key at all
  yields `[]` and raises — unguarded.

**Events** (`create_database.py:198`) — line must contain
`{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"` and **must not contain
`EVENT_FLOWS_TO` anywhere**. Both endpoints must already exist in the node tables. Kept
attributes: `src_hash, src_index, type, dst_hash, dst_index, timestampNanos`.
**Dropped**: `size`, `sequence`, `threadId`, `predicateObject2`, `programPoint`,
`properties`, `hostId`.

**Direction reversal** — `edge_reversed = [ACCEPT, RECVFROM, RECVMSG]` (`config.py:59`).

**Graph filter** — `include_edge_type` = WRITE, READ, CLOSE, OPEN, EXECUTE, SENDTO,
RECVFROM (`config.py:67`), applied at `embedding.py:105`.
⚠️ **Inconsistency**: `EVENT_ACCEPT` and `EVENT_RECVMSG` are reversed on import but are
**not** in `include_edge_type`, so both are discarded at vectorization — the reversal for
those two types is dead code, and accept/recvmsg semantics never reach the model.

### 2.2 THEIA E3 — `theia3_datapreprocess.ipynb`

| | |
| --- | --- |
| **Edges** | subject → (file ∣ netflow ∣ **subject**) |
| **msg dim** | 16 + **9** + 16 = 41 |

- **subject** cell [11] — from real `Subject` records.
  Identity = `sha256(cmdLine + "," + tgid + "," + path)`; table stores `cmdLine, tgid, path`.
  **But `node2id.msg` = `path` only** (cell [22], `i[-1]`), so cmdline and tgid
  contribute to *identity* but not to *features*: two nodes can be distinct and yet
  carry byte-identical feature vectors. This is the only experiment where identity keys
  on pid-ish information (`tgid`), i.e. the least node-collapse of the six.
  `res = re.findall(...)[0]` is **not** inside the `try` — a Subject line without both
  `cmdLine` and `tgid` raises and aborts the loop.
- **file** cell [19] — label = `"filename"` (not `path`). FileObjects without a
  `filename` field are silently skipped.
- **Events** cell [27] — the only experiment permitting **subject→subject** edges
  (`if objectid in subject2hash`). Endpoint resolution order is
  `subject2hash → file2hash → netobj2hash`, each overwriting the previous: a uuid
  present in more than one map resolves to the **last** match, not the first.
  Excludes `EVENT_FLOWS_TO`.
- **Reversal** = READ, READ_SOCKET_PARAMS, RECVFROM, RECVMSG.
- **Graph filter** cell [42] = `e[2] in rel2id`, 9 types: CONNECT, EXECUTE, OPEN, READ,
  RECVFROM, RECVMSG, SENDMSG, SENDTO, WRITE. (No CLOSE.)

### 2.3 CLEARSCOPE E3 — `clearscope3_datapreprocess.ipynb`

| | |
| --- | --- |
| **Edges** | subject → (file ∣ netflow) |
| **msg dim** | 16 + **8** + 16 = 40 |

- **subject** cell [8] — label = `cmdLine`. A `Subject` record with a null `cmdLine`
  fails both the `try` and the fallback → **the subject is dropped entirely**, and with
  it every event that references it.
- **file** cell [10] — label = `path`.
- **Feature quirk** cell [20] — `subject2higlist` splits the cmdline on `'.'`
  (Android package-name hierarchy) while files use `'/'`.
- **Reversal** = ACCEPT, RECVFROM, RECVMSG. Excludes `EVENT_FLOWS_TO`.
- **Graph filter** cell [33] = 8 types: CLOSE, OPEN, READ, WRITE, RECVFROM, RECVMSG,
  SENDMSG, SENDTO.

### 2.4 CADETS E5 — `cadets5_datapreprocess.ipynb`

| | |
| --- | --- |
| **Edges** | subject → (file ∣ netflow) |
| **msg dim** | 16 + **9** + 16 = 41 |

Two-pass registration, and the **edge-type filter feeds back into node selection**:

- cell [10] registers every cdm20 `Subject` and `FileObject` uuid with the placeholder
  `'none'`.
- cell [12] fills subject labels from `"exec"` on Event lines — **only for events whose
  type is in `include_edge_type`** (cell [6]).
- cell [16] fills file labels from `predicateObjectPath`, again **only for included
  event types**; missing path → `'null'`.
- cell [14] drops every subject still `'none'`; cell [17] drops files that are `'none'`
  **or** `'null'`.

So a process or file that only ever participates in, say, `EVENT_CLONE` or
`EVENT_MMAP` is not merely edge-filtered — it **does not exist as a node at all**.
The node population is a function of the edge-type list.

- **Events** cell [25] — **no `EVENT_FLOWS_TO` guard** (unlike the E3 notebooks).
  Reversal = READ, RECVFROM, RECVMSG.
- Raw volume recorded in the notebook: `total_event_count: 1193669198`.
- **Graph filter** cell [42] = 9 types: CLOSE, OPEN, READ, WRITE, EXECUTE, RECVFROM,
  RECVMSG, SENDMSG, SENDTO. Days `range(8,18)`.
- `subject2higlist` splits `exec` on `'.'`; files on `'/'`.

### 2.5 THEIA E5 — `theia5_datapreprocess.ipynb`

| | |
| --- | --- |
| **Edges** | subject → (file ∣ netflow) |
| **msg dim** | 16 + **9** + 16 = 41 |

- **subject** cell [9] — label = `"path"` from the `Subject` record; records without a
  path are dropped (both `try` branches fail).
- **file** cell [12] — label = `"filename"`.
- **Events** cell [21] — no `EVENT_FLOWS_TO` guard; no type filter at insert time.
  Reversal = RECVFROM, RECVMSG, READ.
- **Two different type lists coexist**: `include_edge_type` (cell [19], 9 types incl.
  CLOSE, no CONNECT) is defined but **never used**; the graph filter (cell [38]) uses
  `rel2id` — CONNECT, EXECUTE, OPEN, READ, RECVFROM, RECVMSG, SENDMSG, SENDTO, WRITE.
  Net effect: `EVENT_CLOSE` is imported into Postgres but excluded from every graph.
- `subject2higlist` splits on `'/'` here (unlike CADETS E5 / CLEARSCOPE E3).

### 2.6 CLEARSCOPE E5 — `clearscope5_datapreprocess.ipynb`

| | |
| --- | --- |
| **Edges** | subject → (file ∣ netflow) |
| **msg dim** | 16 + **10** + 16 = 42 |

- **subject** cell [8] — label = `cmdLine`; **file** cell [10] — label from
  `{"map":{"path":"..."`.
- **Only experiment that filters edge types before the DB insert.** cells [19]–[20]
  build `filter_type` and drop everything else *before* `insert into event_table`:
  ACCEPT, CLONE, CLOSE, CREATE_OBJECT, EXECUTE, OPEN, READ, RECVFROM, SENDTO, WRITE.
  The Postgres `event_table` is therefore already filtered — it is not a faithful
  record of the log, unlike every other dataset.
- `include_edge_type` (cell [17]) is defined but unused; `filter_type` wins.
- **Reversal** = READ, RECVFROM, RECVMSG — note `EVENT_ACCEPT` is *kept* but **not
  reversed**, the opposite of CADETS E3 / CLEARSCOPE E3.
- `subject2higlist` splits on `'/'`.

### 2.7 OpTC — `optc_datapreprocess.ipynb`

Structurally different from all the DARPA TC pipelines.

| | |
| --- | --- |
| **Schema** | one `event_table(src_id, src_type, edge_type, dst_id, dst_type, hostname, timestamp, data_label)` + `nodeid2msg`. No per-type node tables, no `node2id`. |
| **Node identity** | **raw uuid** — the only experiment that does *not* hash a label |
| **msg dim** | 16 + **10** + 16 = 42 |

- **Nodes**: `actorID` is always typed `PROCESS`; the object is typed by the record's
  `object` field, restricted to `node_type_used = {FILE, FLOW, PROCESS}` (cell [5];
  `SHELL` is commented out). `MODULE, REGISTRY, TASK, THREAD, USER_SESSION` never enter.
- **Labels** are host-prefixed, `"<host>_@<path>"`:
  process → `properties.image_path`; FILE → `properties.file_path`;
  FLOW → `"<direction>#<src_ip>:<src_port>-><dest_ip>:<dest_port>"`.
  A record missing the relevant property makes `process_raw_dic` return `{}` (bare
  `except`) and the whole event is dropped.
- **PROCESS objects never get a label written** — only actors do. A process that is only
  ever an object (e.g. the target of a `TERMINATE`) has no `nodeid2msg` entry, and every
  edge touching it is dropped at vectorization
  (`if e[3] not in node_uuid2path ... continue`).
- **Host allow-list** (`is_selected_hosts`) — and it is **redefined between the two
  import cells**: benign import (cell [17]) uses
  `0201, 0402, 0660, 0501, 0051, 0209`; evaluation import (cell [21]) swaps
  `0209 → 0207`.
- **Timestamps are milliseconds**, not nanoseconds (`datetime_to_timestamp_US`,
  `timestamp*1000 + int(ms)`). Fractional-second handling assumes exactly a
  `-04:00` suffix and a 3-digit fraction.
- **Data modification before parsing**: `line = line.replace('\\\\','/')` — Windows
  path separators are rewritten to `/` in the raw JSON text before `json.loads`.
- `reverse_edge_type = ["READ"]` is declared (cell [5]) and **never applied** — OpTC
  edges are never reversed.
- **Node indices are per-graph.** Each `(day, host, data_label)` graph rebuilds
  `node_uuid2index` from 0. The same uuid gets different integers in different graphs.
  Memory is reset per graph (`train()` / `test_day_new()`), so this is self-consistent,
  but it means **no memory or index continuity across host-days at all**.
- Ground truth: `labels.csv` matched on `actorID`/`objectID`, restricted to actions in
  `edge2vec`.

### 2.8 StreamSpot — `StreamSpot/src/preprocess.py`

- Input is already an edge list; the TSV is inserted **verbatim** into `raw_data`
  (`preprocess.py:35-45`). **No filtering, no exclusion, no relabelling.**
- Nodes = the ids from the file. Features are **8-dim one-hot node types** and
  **26-dim one-hot edge types** — no hashing, no paths (`preprocess.py:109-121`).
  `msg` dim = 8 + 26 + 8 = 42.
- **No timestamps exist.** `t` is the Postgres `_id` serial, i.e. row order
  (`preprocess.py:140`, comment: *"Use logical order of the event to represent the time"*).
  Every StreamSpot "temporal" result is therefore over a synthetic integer ordering.
- ⚠️ **Non-deterministic encoding**: `node_type` and `edge_type` are Python `set`s and
  the one-hot index is assigned by iteration order (`preprocess.py:47-107` and `112-121`). With
  string-hash randomisation the mapping differs between interpreter runs. It is
  internally consistent because one `preprocess.py` process writes all 600 graphs — but
  a re-run produces a *different* encoding, so the published pre-trained model is
  incompatible with freshly preprocessed data.
- Split (`train.py:29-34`, `test.py:148-263`): train on graphs 0, 100, 200, 400, 500;
  validate on 505 and ranges 1–24/101–124/201–224/401–424/501–524; test includes
  300–399 (the attack batch).

---

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

### 5.2 Node features are character-level hashes (§1.1)

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

The character-level feature-hashing finding (§1.1) was verified directly:

```
uv venv --python 3.9 && uv pip install "scikit-learn==1.2.0" "numpy<2"
```

then hashing `list2str(['file'] + path2higlist('/etc/passwd'))` as a bare `str`, and
comparing against a character-anagram path and against the token-list form. Script is
in the session scratchpad; it should be promoted to `code/` as the `RW7.P3` artifact
(one script, one assertion, citing the `environment-settings.md:23` pin).
