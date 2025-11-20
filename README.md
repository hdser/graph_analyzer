# Graph Analyzer

![Graph Analyzer](img/header-graph_analyzer.png)

End-to-end toolkit for **trust-network analysis and visualization**:

- Compute 120+ NetworkX metrics on Circles v1/v2 trust graphs  
- Stream graphs + metrics directly into **Cytoscape Desktop**  
- View any Cytoscape network in a **browser** via a small FastAPI + Cytoscape.js app  
- Maintain **blacklist / whitelist** decisions in a SQLite DB

The repo is organized so that the “heavy” graph logic lives in Python, the
interactive layout lives in Cytoscape Desktop, and the browser viewer is a thin
layer on top.

---

## 1. Installation

### 1.1 Python environment

From the repo root:

```bash
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Core dependencies (metrics, streaming, blacklist tools)
pip install -r requirements.txt
````

### 1.2 Cytoscape Desktop

Install **Cytoscape 3.9+** from the official site and make sure:

* Cytoscape is running while you use `stream_to_cytoscape.py` or the web viewer.
* CyREST is enabled (it is by default on `http://127.0.0.1:1234/v1`).

You can open the example session:

```text
cytoscape/stream_graphs.cys
```

to get a feel for the layouts/styles used.

### 1.3 Web viewer dependencies (optional, for browser UI)

The web viewer lives in `web_viewer/` and has its own lightweight requirements:

```bash
# From repo root, with venv activated
pip install -r web_viewer/requirements.txt
```

---

## 2. Configuration (`.env`)

Both `graph_metrics.py` and `stream_to_cytoscape.py` use the same DB/metrics
configuration via a `.env` file in the repo root.

Create it from the example (if present) or from scratch:

```bash
cp .env.example .env  # if .env.example exists
# or create .env manually
```

Example:

```env
# PostgreSQL connection
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=your_host
DB_NAME=your_database

# Graph metrics
OUTPUT_FILE=graph_metrics.csv
N_JOBS=4            # Number of parallel workers (0 or empty = auto cores-1)
METRICS_MODE=all    # See "Metrics modes" below
```

`stream_to_cytoscape.py` also accepts `--metrics-mode` on the CLI; that value
overrides `METRICS_MODE` from `.env` for that run.

---

## 3. SQL files → networks

All network construction is driven by **SQL files** in the `sql/` folder.
Each SQL is expected to return a table with:

* `source` – address / avatar sending or trusting
* `target` – address / avatar receiving or being trusted
* (optional) extra columns used as edge attributes, e.g.:

  * `amount` – for flows / edge width
  * `timestamp` – for temporal coloring
  * any other attributes you want in Cytoscape

Current files:

```text
sql/
  crc_v1_trusts.sql   # Circles v1 trust graph
  crc_v2_trusts.sql   # Circles v2 trust graph
  crc_v2_invites.sql  # Invitation graph
  crc_v2_flows.sql    # Token flow graph
```

These are used mainly by `stream_to_cytoscape.py`, but you can reuse them
anywhere else you like.

---

## 4. Graph metrics: `graph_metrics.py`

This script computes 120+ metrics on a **directed trust graph** using NetworkX,
with configurable metric categories and parallel processing.

### 4.1 Basic usage

```bash
# Using METRICS_MODE from .env
python graph_metrics.py

# See supported modes and categories
python graph_metrics.py --help
```

Typical config in `.env`:

```env
METRICS_MODE=essential
N_JOBS=4
OUTPUT_FILE=data/graph_metrics_v2.csv
```

The script:

1. Connects to PostgreSQL using `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_NAME`.
2. Runs the built-in query (currently v2 trust relations).
3. Builds an `nx.DiGraph` with `(source, target)` edges.
4. Computes metrics according to `METRICS_MODE`.
5. Writes a CSV with one row per node:

   ```csv
   avatar,in_degree,pagerank,community_id,total_degree,...
   0xabc...,5,0.00123,54,32,...
   ```

Booleans are converted to 0/1, `NaN/inf` are replaced by 0.

### 4.2 Metrics modes

You can choose **which categories** to compute using `METRICS_MODE`
(in `.env` or via `--metrics-mode` in streaming).

Preset modes:

* `basic` – topology + clustering (very fast)
* `essential` – topology + centrality + clustering + community
* `moderate` – essential + paths + structural
* `all` – all 15 categories (120+ metrics, can be expensive)

You can also pass categories directly:

```env
METRICS_MODE=topology,centrality,paths,reach
```

Categories include:

* `topology`, `centrality`, `clustering`, `community`, `paths`,
* `distances`, `structural`, `reciprocity`, `reach`, `components`,
* `vitality`, `dispersion`, `efficiency`, `flow`, `dominance`.

(See the previous README content if you need the full metric list; the code in
`graph_metrics.py` matches that description.)

---

## 5. Streaming into Cytoscape: `stream_to_cytoscape.py`

This script is responsible for:

* Running the chosen SQL file(s) to build one or more edge sets
* Computing node metrics **once** on a chosen “metrics graph”
* Creating **one Cytoscape network per SQL file**
* Streaming updates over time if desired

It talks to Cytoscape via `py4cytoscape` and CyREST.

### 5.1 CLI arguments

From the repo root:

```bash
python stream_to_cytoscape.py --help
```

Conceptually, the important options are:

* `--run-type {once,stream}`

  * `once` – do a single load into Cytoscape and exit
  * `stream` – repeat at a fixed interval, updating metrics / networks

* `--sql-files <comma-separated list>`
  Paths to SQL files that each define a graph layer, e.g.:

  ```bash
  --sql-files sql/crc_v2_trusts.sql,sql/crc_v2_invites.sql
  ```

* `--metrics-mode <mode>`
  Same semantics as in `graph_metrics.py` (`basic`, `essential`, etc.).
  If omitted, falls back to `METRICS_MODE` in `.env`.

* `--metrics-graph-id <id>`
  Which graph (by ID derived from SQL filename) to use for metrics.
  For example, `sql/crc_v2_trusts.sql` will usually map to `crc_v2_trusts`.
  Metrics are computed on that graph and the resulting node attributes are
  reused for all layers.

* `--interval <seconds>` (stream mode only)
  How often to re-run SQL + metrics and update Cytoscape.

### 5.2 Typical workflows

#### One-time network creation

Create just the v2 trust network with metrics and style in Cytoscape:

```bash
python stream_to_cytoscape.py \
  --run-type once \
  --sql-files sql/crc_v2_trusts.sql \
  --metrics-mode topology,community
```

This will:

1. Run `crc_v2_trusts.sql` to get `source,target,...` edges.
2. Build a directed NetworkX graph.
3. Compute metrics for all avatars.
4. Push into Cytoscape as one network, with all node columns attached.

#### Multiple layers, shared metrics

Create both trust and invite graphs, computing metrics only on the trust graph:

```bash
python stream_to_cytoscape.py \
  --run-type once \
  --sql-files sql/crc_v2_trusts.sql,sql/crc_v2_invites.sql \
  --metrics-mode topology,community \
  --metrics-graph-id crc_v2_trusts
```

This will:

* Run both SQL files.
* Build one NetworkX graph for `crc_v2_trusts` and compute metrics.
* Create **two networks** in Cytoscape:

  * `crc_v2_trusts (shared metrics)`
  * `crc_v2_invites (shared metrics)`
* Attach the **same node metrics** (computed from the trust graph) to both
  networks, so you can compare structure vs trust centrality, etc.

#### Streaming mode

Recompute SQL + metrics and update Cytoscape on a schedule:

```bash
python stream_to_cytoscape.py \
  --run-type stream \
  --sql-files sql/crc_v2_trusts.sql,sql/crc_v2_invites.sql \
  --metrics-mode topology,community \
  --metrics-graph-id crc_v2_trusts \
  --interval 60
```

* First iteration creates the networks as above.
* Every 60 seconds:

  * Re-runs the SQL.
  * Rebuilds graphs and recomputes metrics.
  * Either updates existing node attributes in Cytoscape, or (depending on how
    you’ve configured the script) recreates the networks if edge sets change
    dramatically.

> **Note:** Make sure Cytoscape is open before running the streaming script.
> You’ll see the networks appear in the left-hand panel as they are created.

---

## 6. Web viewer: `web_viewer/`

The web viewer is a separate, optional component that lets you inspect any
Cytoscape network in a **browser**, using Cytoscape.js.

It **does not** replace Cytoscape Desktop; it just reads the current state from
Cytoscape via CyREST and renders it in a web UI.

### 6.1 Running the web app

From the repo root:

```bash
source venv/bin/activate
cd web_viewer

# Dependencies should already be installed via:
# pip install -r requirements.txt

python -m uvicorn app:app --reload
```

You should see something like:

```text
Uvicorn running on http://127.0.0.1:8000
```

Open your browser at:

> [http://127.0.0.1:8000](http://127.0.0.1:8000)

### 6.2 How it works

* `web_viewer/app.py` (FastAPI) exposes:

  * `GET /api/networks` – list of networks currently in Cytoscape
  * `GET /api/networks/{suid}/view` – exports nodes, edges, and positions
* `web_viewer/static/app.js`:

  * Fetches the list of networks and populates a `<select>` dropdown.
  * On selection, fetches `/api/networks/{suid}/view`.
  * Builds a Cytoscape.js instance with:

    * `layout: { name: "preset" }` – uses the exact positions from Desktop.
    * A **custom Cytoscape.js style** that approximates your
      `cytoscape/styles_stream.xml`:

      * Dark background (`#191919`).
      * Nodes sized by `total_degree`.
      * Nodes colored by `community_id`.
      * Edge width by `amount`.
      * Edge color by `timestamp`.

Because Cytoscape Desktop’s vizmap format is different from Cytoscape.js
styles, the web viewer doesn’t import `styles_stream.xml` directly; instead, it
uses the same underlying **data columns and ranges** to reproduce a similar look
on the web.

### 6.3 Performance tweaks for large graphs

The viewer uses several options to keep pan/zoom responsive:

* `pixelRatio: 1` – avoids expensive Retina rendering.
* `textureOnViewport: true` – caches the canvas as a texture.
* `motionBlur: true` – smooths panning.
* `hideEdgesOnViewport: true` – hides edges while moving.
* `hideLabelsOnViewport: true` – hides labels while moving.

If you want **no labels at all**, edit `web_viewer/static/app.js` and set node
style:

```javascript
{
  selector: "node",
  style: {
    // ...
    label: "",          // no labels
    "text-opacity": 0
  }
}
```

---

## 7. Blacklist / Whitelist workflow

The repo includes two helper scripts to manage blacklists/whitelists in
`data/blacklist.db` (SQLite) using CSV files and graph metrics.

### 7.1 Compare CSV blacklists vs DB: `compare_blacklist_csvs.py`

This script:

* Merges multiple CSV blacklist sheets.
* Cross-checks them with existing `Blacklist` and `Whitelist` tables in
  `data/blacklist.db`.
* Uses metrics from `graph_metrics_v1.csv` / `graph_metrics_v2.csv` to apply
  rules (v1/v2 heuristics).
* Outputs a consolidated `blacklist_full_updated.csv` with reasons/source.

Example:

```bash
python compare_blacklist_csvs.py \
  --csv data/blacklist_Sheet1.csv data/blacklist_Sheet2.csv \
  --db data/blacklist.db \
  --graph-metrics-v1 data/graph_metrics_v1.csv \
  --graph-metrics-v2 data/graph_metrics_v2.csv \
  --output-csv data/blacklist_full_updated.csv \
  --verbose
```

Key args:

* `--csv` – one or more CSV sheets with candidate blacklist entries.
* `--db` – SQLite DB, default `data/blacklist.db`.
* `--graph-metrics-v1/2` – metrics CSVs from `graph_metrics.py`.
* `--output-csv` – merged result.
* `--verbose` – more logging.

### 7.2 Apply CSV updates to DB: `update_blacklist.py`

This script writes changes into `data/blacklist.db`:

* `Blacklist` table
* `Whitelist` table

Add/update blacklist from a CSV:

```bash
python update_blacklist.py \
  --db data/blacklist.db \
  --list-type blacklist \
  --csv data/blacklist_full_updated.csv
```

Add to whitelist:

```bash
python update_blacklist.py \
  --db data/blacklist.db \
  --list-type whitelist \
  --csv data/my_whitelist.csv
```

Remove entries:

```bash
# Remove from blacklist
python update_blacklist.py \
  --db data/blacklist.db \
  --list-type blacklist \
  --csv data/remove_from_blacklist.csv \
  --remove

# Remove from whitelist
python update_blacklist.py \
  --db data/blacklist.db \
  --list-type whitelist \
  --csv data/remove_from_whitelist.csv \
  --remove
```

---

## 8. Repository layout

For orientation:

```text
.
├── README.md                  # This file
├── graph_metrics.py           # Core metrics computation
├── stream_to_cytoscape.py     # Streaming graphs & metrics into Cytoscape
├── compare_blacklist_csvs.py  # Merge / analyze blacklist CSVs vs DB
├── update_blacklist.py        # Apply CSV updates to blacklist/whitelist DB
├── requirements.txt           # Main Python deps
├── .env                       # DB + metrics config (not committed)
│
├── sql/                       # All graph-defining SQL queries
│   ├── crc_v1_trusts.sql
│   ├── crc_v2_trusts.sql
│   ├── crc_v2_invites.sql
│   └── crc_v2_flows.sql
│
├── cytoscape/
│   ├── stream_graphs.cys      # Example Cytoscape session
│   ├── styles.xml             # Additional vizmap styles (Desktop only)
│   └── styles_stream.xml      # Reference style used when designing graphs
│
├── data/
│   ├── blacklist.db           # SQLite DB for black/whitelists
│   └── blacklist_original.db  # Original DB snapshot
│
├── figs/                      # Exported PNGs of graphs
│   └── ...                    # Trust/Invitation graph images
│
└── web_viewer/                # Browser-based viewer
    ├── app.py                 # FastAPI app, talks to Cytoscape via CyREST
    ├── requirements.txt       # Web viewer deps (FastAPI, Uvicorn, py4cytoscape)
    └── static/
        ├── index.html         # UI shell
        ├── app.js             # Cytoscape.js glue, custom styles
        └── style.css          # Layout / colors for the viewer
```



