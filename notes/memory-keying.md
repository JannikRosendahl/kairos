# KAIROS — what the TGN node memory is keyed to

Analysed at commit `a0cb2ab58bdf637298f09e8dea0aea5eb1a6af9f`, 2026-09-14.
Every claim below carries a `file:line` source. Companion notes:
[`../notes/kairos-data-import-analysis.md`](../notes/kairos-data-import-analysis.md)
(§3.1, §5.1) and [`../notes/model-methodology-and-architecture-analysis.md`](../notes/model-methodology-and-architecture-analysis.md)
(§2, §3.3) — this report is the self-contained, per-dataset derivation of the keying.

**Reading the citations.** `DARPA/CADETS_E3/*.py` are ordinary Python files.
All other datasets ship as Jupyter notebooks; those line numbers are **raw-JSON
line numbers in the `.ipynb` file**, which is pretty-printed one source line per
file line, so `sed -n '314p' DARPA/THEIA_E3/theia3_graph_learning.ipynb` shows
exactly the cited line.

---

## 0. Answer in one paragraph

A TGN memory slot in KAIROS is **not** a system entity. It is an integer
`index_id` that densely enumerates the **distinct SHA-256 hashes of a
per-dataset string of node attributes** — never the CDM `uuid`. Which attributes
go into that string differs per dataset and per node type. For **CADETS E3** a
subject slot is `sha256(exec)`, i.e. one slot per *executable name*, shared by
every process that ever ran it. For **THEIA E3** a subject slot is
`sha256(cmdLine + "," + tgid + "," + path)`, i.e. roughly one slot per *process
instance*. Files and netflows are keyed on path and on the 4-tuple
`src,sport,dst,dport` respectively in both. The memory tensor is allocated as
`TGNMemory(max_node_num, ...)` and indexed by that same `index_id`.

---

## 1. The chain, hop by hop (CADETS E3)

CADETS E3 is the only dataset with a scripted (non-notebook) pipeline, so it is
the clearest place to read the chain. Five hops:

### Hop 1 — the hash function

```python
def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()
```
`DARPA/CADETS_E3/create_database.py:23-27`

Despite the name it is **SHA-256**, hex digest, 64 chars. The 64-char length is
later used as a sentinel to tell a hash apart from a uuid
(`create_database.py:59`, `:92`, `:127`).

### Hop 2 — what string is hashed, per node type

| type | preimage | source |
| --- | --- | --- |
| netflow | `srcaddr + "," + srcport + "," + dstaddr + "," + dstport` | `create_database.py:48-49` |
| subject | the `exec` field of the **Event** record | `create_database.py:78-81`, hashed at `:93` |
| file | `predicateObjectPath` (the **Event**'s object path, not the FileObject) | `create_database.py:122-123`, hashed at `:128` |

Two things to note about CADETS E3 specifically:

- The subject label comes from the regex
  `'"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}(.*?)"exec":"(.*?)"'`
  applied to lines containing `"Event"` (`create_database.py:77-79`) — **not**
  from `Subject` entity records. So no `cmdLine`, no `pid`/`tgid`, no start
  time enters the key. Processes whose events never carry `exec` get the literal
  key `"null"` (`create_database.py:85`).
- File paths are read off Event records and only when a path is present:
  lines with `"predicateObjectPath":null,` or `<unknown>` are skipped
  (`create_database.py:121-123`).

### Hop 3 — hash → dense `index_id`

`node_list` is a dict **keyed by hash**, filled file-first, then subject, then
netflow (`create_database.py:146`, `:158`, `:170`). It is then enumerated in
insertion order:

```python
node_list_database = []
node_index = 0
for i in node_list:
    node_list_database.append([i] + node_list[i] + [node_index])
    node_index += 1
```
`create_database.py:176-180`, inserted into `node2id` at `:182-186`.

The table is `node2id(hash_id PK, node_type, msg, index_id)` —
`DARPA/settings/database.md:78-88`. Because `node_list` is keyed by hash, **the
number of `index_id` values equals the number of distinct hashes**, not the
number of uuids.

### Hop 4 — `index_id` lands in the edge stream

The lookup table is built both directions:

```python
nodeid2msg[i[0]] = i[-1]        # hash    -> index_id
nodeid2msg[i[-1]] = {i[1]: i[2]}  # index_id -> {node_type: label}
```
`create_database.py:191-194` (identical helper at `kairos_utils.py:117-126`).

Events are stored with **both** the hash and the index:

```python
datalist.append([subjectId, nodeid2msg[subjectId], relation_type,
                 objectId,  nodeid2msg[objectId],  time_rec])
```
`create_database.py:221-223` (reversed-direction branch at `:216-219`), into
`event_table(src_node, src_index_id, operation, dst_node, dst_index_id, timestamp_rec, _id)`
— `DARPA/settings/database.md:30-38`.

Vectorization reads columns 1 and 4, i.e. the **index ids**:

```python
edge_temp = [int(e[1]), int(e[4]), e[2], e[5]]
...
dataset.src = torch.tensor(src); dataset.dst = torch.tensor(dst)
```
`DARPA/CADETS_E3/embedding.py:104`, `:113-127`.

### Hop 5 — `index_id` is the memory slot

```python
max_node_num = 268243  # the number of nodes in node2id table +1
min_dst_idx, max_dst_idx = 0, max_node_num
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
```
`DARPA/CADETS_E3/model.py:7-10`

```python
memory = TGNMemory(max_node_num, node_feat_size, node_state_dim, time_dim,
                   message_module=IdentityMessage(...), aggregator_module=LastAggregator())
...
neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)
```
`DARPA/CADETS_E3/train.py:79-86`, `:102`

`TGNMemory`'s first argument is `num_nodes`; it allocates the state tensor of
that many rows and is addressed by `memory(n_id)` where `n_id` is built straight
from `batch.src` / `batch.dst` (`train.py:42-47`, `test.py:63-68`). So:

> **memory slot = `index_id` = rank of `sha256(label-string)` in `node2id`.**

The same index also addresses the neighbour list (`train.py:64`) and the static
input feature (`embedding.py:117`, `node2higvec[i[0]]`), so identity, features
and neighbourhood are all keyed the same way.

### CADETS E3 — the collapse, in numbers

Counts asserted in the pipeline's own comments: 155,322 netflow uuids
(`create_database.py:235`), 224,146 subject uuids (`:239`), 234,245 file uuids
(`:243`) — 613,713 uuids total — collapsing to **268,242 `node2id` entities**
(`:247`), which is exactly `max_node_num - 1` at `model.py:7`. That is a **56%
reduction**; every process that ever exec'd `/usr/sbin/sshd` shares one memory
slot.

---

## 2. The chain (THEIA E3)

Same five hops, different preimages. Notebook line numbers are raw-JSON lines.

### Hop 1 — hash function
`DARPA/THEIA_E3/theia3_datapreprocess.ipynb:19` — same `stringtomd5` = SHA-256.

### Hop 2 — preimages

| type | preimage | source |
| --- | --- | --- |
| netflow | `srcaddr + "," + srcport + "," + dstaddr + "," + dstport` | `theia3_datapreprocess.ipynb:301-302` |
| subject | `cmdLine + "," + tgid + "," + path` | `theia3_datapreprocess.ipynb:385`, hashed at `:392` |
| file | `filename` from the FileObject record | `theia3_datapreprocess.ipynb:3731-3732` |

Subject fields come from **`Subject` entity records**, not Event records:

```python
if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' in line :
    res=re.findall('Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"properties":{"map":{"tgid":"(.*?)"',line)[0]
```
`theia3_datapreprocess.ipynb:374-375`, fields bound at `:381-383`; `path` is a
separate optional match defaulting to `"null"` (`:377-380`).

This is the key difference from CADETS E3: `tgid` (the thread-group id, i.e. the
pid) is **inside** the hash, so THEIA E3 slots are approximately *process
instances*, not executable names. Verified stored shape
`[uuid, hash, cmdLine, tgid, path]` (`:393` built, `:3607` inserted) against the DDL
`subject_node_table(node_uuid, hash_id, "cmdLine", tgid, path)` —
`DARPA/settings/database.md:143-151`.

### Hop 3 — hash → `index_id`
`node_list[i[1]] = ["file"|"subject"|"netflow", i[-1]]` keyed on `hash_id`
(`theia3_datapreprocess.ipynb:3789`, `:3802`, `:3816`), enumerated at
`:3829-3833`, inserted into `node2id` at `:3842-3845`. Same `node2id` DDL as CADETS
(`DARPA/settings/database.md:156-166`).

Concrete evidence of the resulting table, from the notebook's own saved output:

```
{'17ce196318e17c2c5c2fbbb631e7de266ab192597eba89487682492e8c77cb6f': 0,
 0: {'file': '/run/shm/org.chromium.wLJAmM'},
 '164157bba7655bd5906ba831ca6d081d94422778ca59f31b6e8a661d2d0aa5fb': 1,
 1: {'file': '/home/admin/.cache/mozilla/firefox/.../FDD3309...'}, ...}
```
`theia3_datapreprocess.ipynb:3874-3879` — hash → int, int → `{type: label}`.

### Hop 4 — edge stream
`theia3_datapreprocess.ipynb:4917-4926` converts every uuid to its hash, gates on
both sides being 64-char hashes (`:4927`) and stores
`(hash, index, rel, hash, index, time)` (`:4931`, `:4933`); vectorization takes
`edge_temp=[int(e[1]),int(e[4]),e[2],e[5]]` at `:5369` and assigns
`dataset.src/dst` at `:5384-5394`.

THEIA E3 is the **only** DARPA experiment whose `event_table` also admits
subject→subject edges (`if objectid in subject2hash.keys()` at `:4921-4922`);
the others require the object to be a file or netflow.

### Hop 5 — memory slot
```python
max_node_num = 828398
neighbor_loader = LastNeighborLoader(max_node_num, size=20, device=device)
...
memory = TGNMemory(max_node_num, train_data.msg.size(-1), memory_dim, time_dim, ...)
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
```
`DARPA/THEIA_E3/theia3_graph_learning.ipynb:314`, `:316`, `:369-376`, `:396`.

### THEIA E3 — numbers

- `subject_node_table` rows (uuid-level): **279,369** — `theia3_datapreprocess.ipynb:3658`.
- Distinct `node2id` entities actually featurized: **828,312** — from the saved
  progress bar over `node_msg_dic_list`, `theia3_datapreprocess.ipynb:5052`.
- `max_node_num` = **828,398** (`theia3_graph_learning.ipynb:314`), i.e. **86
  slots more than there are nodes**. Harmless over-allocation (unused rows stay
  at the zero-initialised state), but it is not the `N+1` convention CADETS E3
  documents, so it should not be cited as a node count.

---

## 3. Cross-dataset summary of the key

What string is SHA-256'd to produce the memory slot, per dataset:

| dataset | subject key | file key | netflow key | `max_node_num` |
| --- | --- | --- | --- | --- |
| **CADETS E3** | `exec` (from Event)<br>`create_database.py:79`,`:93` | `predicateObjectPath` (from Event)<br>`create_database.py:122`,`:128` | `src,sport,dst,dport`<br>`create_database.py:48-49` | 268,243<br>`model.py:7` |
| **THEIA E3** | `cmdLine,tgid,path` (from Subject)<br>`theia3_datapreprocess.ipynb:385`,`:392` | `filename` (from FileObject)<br>`theia3_datapreprocess.ipynb:3731-3732` | `src,sport,dst,dport`<br>`theia3_datapreprocess.ipynb:301-302` | 828,398<br>`theia3_graph_learning.ipynb:314` |
| CLEARSCOPE E3 | `cmdLine` (from Subject)<br>`clearscope3_datapreprocess.ipynb:311`,`:315`,`:337` | `path` (from FileObject)<br>`clearscope3_datapreprocess.ipynb:374`,`:377`,`:391` | `src,sport,dst,dport`<br>`clearscope3_datapreprocess.ipynb:242-243` | 172,724<br>`clearscope3_graph_learning.ipynb:287` |
| CADETS E5 | `exec` (from Event)<br>`cadets5_datapreprocess.ipynb:712-715`,`:750` | `predicateObjectPath` (from Event)<br>`cadets5_datapreprocess.ipynb:816-820`,`:837` | `src,sport,dst,dport`<br>`cadets5_datapreprocess.ipynb:587`,`:589` | 262,626<br>`cadets5_graph_learning.ipynb:308` |
| THEIA E5 | `path` (from Subject)<br>`theia5_datapreprocess.ipynb:399`,`:403`,`:437` | `filename` (from FileObject)<br>`theia5_datapreprocess.ipynb:3500`,`:3503`,`:3518` | `src,sport,dst,dport`<br>`theia5_datapreprocess.ipynb:327`,`:329` | 967,389<br>`theia5_graph_learning.ipynb:295` |
| CLEARSCOPE E5 | `cmdLine` (from Subject)<br>`clearscope5_datapreprocess.ipynb:367`,`:371`,`:392` | `path` (from `map`)<br>`clearscope5_datapreprocess.ipynb:431`,`:435`,`:449` | `src,sport,dst,dport`<br>`clearscope5_datapreprocess.ipynb:296`,`:298` | 139,961<br>`clearscope5_graph_learning.ipynb:293` |
| OpTC | **raw uuid**, re-enumerated per `(day, host, label)`<br>`optc_datapreprocess.ipynb:1868-1886` | raw uuid (same) | raw uuid (same) | `maxnode_num+2`<br>`optc_graph_learning.ipynb:479` |

OpTC is the outlier: `node_uuid2index` is built per graph from the events of
that graph (`optc_datapreprocess.ipynb:1868-1879`) and saved per
day/host/label (`:1888`), so there is **no** hash, no cross-graph slot
continuity, and no entity collapse.

---

## 4. What the slot is *not* keyed to

Verified absent from the preimage in the two target datasets:

- **CDM `uuid`.** Present in the node tables as `node_uuid`
  (`DARPA/settings/database.md:46`, `:120`) and used to resolve events
  (`create_database.py:211-215`), but never hashed into the key.
- **Time / process lifetime.** Nothing timestamp-like enters `stringtomd5`.
  A slot is reused by the next process with the same key.
- **Host.** Single-host datasets; only OpTC prefixes `host_` — and only in the
  *label*, not the key (`optc_datapreprocess.ipynb`, see §3.1 of the import note).
- **Node type, for CADETS E3.** `node_list` is keyed by bare hash
  (`create_database.py:146`,`:158`,`:170`), so a subject whose `exec` string is
  byte-identical to some file's `predicateObjectPath` would occupy **one** slot,
  with `node_type` decided by insertion order (file first, subject overwriting).
  Structurally possible; frequency not measured here. THEIA E3 is not exposed to
  this for subjects (the comma-joined triple cannot collide with a bare
  `filename`), but has the same bare-hash keying (`theia3_datapreprocess.ipynb:3789`, `:3802`, `:3816`).

---

## 5. Memory lifetime over that key

The slot is global and stable across days, but the *state* in it is not carried
across graphs:

- Training resets before **every** day-graph: `memory.reset_state()` and
  `neighbor_loader.reset_state()` at the top of `train()` —
  `DARPA/CADETS_E3/train.py:33-34`, called per graph per epoch at
  `train.py:117-126`. THEIA E3: `theia3_graph_learning.ipynb:416-417`, loop at
  `:1374-1375`.
- Testing resets per day-graph as well: `DARPA/CADETS_E3/test.py:40-41`, with
  `test()` invoked once per day at `test.py:173-211`. THEIA E3:
  `theia3_graph_learning.ipynb:1413-1414`.
- Within a graph, state is written after each batch via
  `memory.update_state(src, pos_dst, t, msg)` (`train.py:63`, `test.py:86`) with
  `LastAggregator` (`train.py:85`), so a slot holds the effect of the **last**
  message addressed to it.

So the effective scope is: *one slot per label-hash, state accumulated within a
single day-graph, zeroed between day-graphs.*

---

## 6. Why this matters for the thesis

1. **"Per-node memory" is per-label-class memory.** Any statement about what TGN
   memory contributes in KAIROS is a statement about state attached to an
   executable name (CADETS E3) or a process instance (THEIA E3). These are not
   the same experiment.
2. **The comparison across datasets is confounded.** CADETS E3 collapses 613,713
   uuids into 268,242 slots (§1); THEIA E3 keeps `tgid` and collapses far less.
   A memory-vs-no-memory delta measured on CADETS E3 and on THEIA E3 is
   measuring two different granularities.
3. **THEIA E3 separates identity from features.** The key is
   `cmdLine,tgid,path` but the featurized label stored in `node2id` is
   `i[-1]` = `path` alone (`theia3_datapreprocess.ipynb:3802` for subjects,
   consumed at `:5038-5039`). Distinct slots therefore receive byte-identical
   input features, and only the memory can tell them apart — which flatters a
   stateful model and handicaps a stateless control on exactly the axis under
   test. Worth an explicit control in the evaluation.
4. **A TGN port must pick a key deliberately.** Keying on uuid (true entity
   memory) is a *different system* from what KAIROS published, and will not
   reproduce its numbers.

---

## 7. Reproducing these checks

```bash
cd related-work/kairos

# CADETS E3 (plain Python)
sed -n '23,27p;48,49p;78,79p;93p;122,123p;128p;176,194p' DARPA/CADETS_E3/create_database.py
sed -n '7,10p' DARPA/CADETS_E3/model.py
sed -n '79,86p;102p' DARPA/CADETS_E3/train.py

# THEIA E3 (notebook — raw JSON lines)
sed -n '19p;301,302p;374,393p;3731,3732p;3789p;3802p;3816p;3829,3833p;3874,3879p' DARPA/THEIA_E3/theia3_datapreprocess.ipynb
sed -n '314,316p;369,376p;396p' DARPA/THEIA_E3/theia3_graph_learning.ipynb

# every key preimage in the repo, at once
# (a few hits are commented-out alternatives, e.g. the dstaddr-only netflow key
#  in the E5 notebooks — check the leading `#` before citing)
grep -rn 'nodeproperty *=' DARPA/
grep -rn 'stringtomd5(' DARPA/
grep -rn 'max_node_num *=' DARPA/
```

Against a populated database, the keying is directly observable:

```sql
-- CADETS E3: one slot, many uuids
SELECT s.hash_id, s.exec, n.index_id, count(*) AS uuids
FROM subject_node_table s JOIN node2id n ON n.hash_id = s.hash_id
GROUP BY 1,2,3 ORDER BY uuids DESC LIMIT 20;

-- collapse ratio
SELECT (SELECT count(*) FROM node2id) AS slots,
       (SELECT count(*) FROM subject_node_table)
     + (SELECT count(*) FROM file_node_table)
     + (SELECT count(*) FROM netflow_node_table) AS uuid_rows;
```
