#!/usr/bin/env python3
import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE_FIELDS = {
    "id",
    "image_id",
    "caption",
    "source_caption",
    "caption_replaced",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local web UI to review and edit captions with scoring metadata."
    )
    parser.add_argument("--input-json", required=True, help="Base input JSON.")
    parser.add_argument(
        "--merge-json",
        action="append",
        default=[],
        help="Optional JSON files with additional fields to merge by id.",
    )
    parser.add_argument(
        "--failures-jsonl",
        action="append",
        default=[],
        help="Optional failures JSONL files with missing_tokens field.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON path for edited annotations.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--page-size", type=int, default=50, help="Default page size.")
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_records(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def merge_annotations(base_data, merge_paths):
    annotations = base_data["annotations"]
    by_id = {ann["id"]: ann for ann in annotations}
    for path in merge_paths:
        data = load_json(path)
        other = data.get("annotations", [])
        for ann in other:
            target = by_id.get(ann.get("id"))
            if not target:
                continue
            for key, value in ann.items():
                if key in ("id", "image_id"):
                    continue
                if key in BASE_FIELDS and key in target:
                    continue
                target[key] = value
    return annotations


def attach_missing_tokens(annotations, failure_paths):
    if not failure_paths:
        return
    by_id = {ann["id"]: ann for ann in annotations}
    for path in failure_paths:
        records = load_jsonl_records(path)
        for record in records:
            missing = record.get("missing_tokens")
            if not missing:
                continue
            target = by_id.get(record.get("id"))
            if not target:
                continue
            target["missing_tokens"] = missing


class CaptionStore:
    def __init__(self, annotations, output_path, page_size):
        self.annotations = annotations
        self.output_path = Path(output_path)
        self.page_size = page_size
        self.by_id = {ann["id"]: ann for ann in annotations}
        self.edited = set()
        self.sort_cache = {}
        self.lock = threading.Lock()
        self._extra_fields = self._collect_extra_fields()
        self.fluency_threshold = None
        self.semantic_threshold = None

    def _collect_extra_fields(self):
        fields = set()
        for ann in self.annotations:
            fields.update(ann.keys())
        fields.difference_update(BASE_FIELDS)
        fields.discard("missing_tokens")
        fields.discard("_edited")
        return sorted(fields)

    def extra_fields(self):
        return self._extra_fields

    def get_sorted_ids(self, sort_key, order, missing_only, edited_only):
        cache_key = (sort_key, order, missing_only, edited_only)
        if cache_key in self.sort_cache:
            return self.sort_cache[cache_key]

        def key_func(ann):
            value = ann.get(sort_key)
            if isinstance(value, (int, float)):
                return (0, value)
            if value is None:
                return (1, float("inf"))
            return (0, str(value))

        filtered = []
        for ann in self.annotations:
            if missing_only and not ann.get("missing_tokens"):
                continue
            if edited_only and ann["id"] not in self.edited:
                continue
            filtered.append(ann)

        reverse = order == "desc"
        if sort_key == "id":
            filtered.sort(key=lambda ann: ann.get("id", 0), reverse=reverse)
        else:
            filtered.sort(key=key_func, reverse=reverse)
        ids = [ann["id"] for ann in filtered]
        self.sort_cache[cache_key] = ids
        return ids

    def update_caption(self, ann_id, text):
        with self.lock:
            ann = self.by_id.get(ann_id)
            if not ann:
                return False
            ann["caption_replaced"] = text
            self.edited.add(ann_id)
            self.sort_cache.clear()
            return True

    def page(self, offset, limit, sort_key, order, missing_only, edited_only):
        ids = self.get_sorted_ids(sort_key, order, missing_only, edited_only)
        total = len(ids)
        subset = ids[offset : offset + limit]
        items = []
        for ann_id in subset:
            ann = self.by_id[ann_id]
            extra = {k: ann.get(k) for k in self.extra_fields() if k in ann}
            items.append(
                {
                    "id": ann.get("id"),
                    "image_id": ann.get("image_id"),
                    "caption": ann.get("caption"),
                    "source_caption": ann.get("source_caption"),
                    "caption_replaced": ann.get("caption_replaced"),
                    "missing_tokens": ann.get("missing_tokens") or [],
                    "extra_fields": extra,
                    "edited": ann_id in self.edited,
                }
            )
        return {"total": total, "items": items}

    def save(self):
        output = {
            "info": self._root.get("info"),
            "images": self._root.get("images"),
            "licenses": self._root.get("licenses"),
            "annotations": self.annotations,
            "simplification": self._root.get("simplification"),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False)

    def set_root(self, root):
        self._root = root
        fluency = root.get("lm_fluency_scoring") or {}
        fluency_thresholds = fluency.get("source_thresholds") or {}
        if fluency_thresholds:
            self.fluency_threshold = max(fluency_thresholds.values())
        semantic = root.get("semantic_scoring") or {}
        semantic_thresholds = semantic.get("similarity_thresholds") or {}
        if semantic_thresholds:
            self.semantic_threshold = max(semantic_thresholds.values())


class ReviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_index().encode("utf-8"))
            return
        if parsed.path == "/api/metadata":
            payload = {
                "total": len(self.server.store.annotations),
                "page_size": self.server.store.page_size,
                "fields": self.server.store.extra_fields(),
                "fluency_threshold": self.server.store.fluency_threshold,
                "semantic_threshold": self.server.store.semantic_threshold,
            }
            return self._send_json(payload)
        if parsed.path == "/api/items":
            params = parse_qs(parsed.query)
            offset = int(params.get("offset", ["0"])[0])
            limit = int(params.get("limit", [str(self.server.store.page_size)])[0])
            sort_key = params.get("sort", ["id"])[0]
            order = params.get("order", ["asc"])[0]
            missing_only = params.get("missing_only", ["0"])[0] == "1"
            edited_only = params.get("edited_only", ["0"])[0] == "1"
            payload = self.server.store.page(
                offset, limit, sort_key, order, missing_only, edited_only
            )
            return self._send_json(payload)
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return self._send_json({"error": "invalid json"}, status=400)

        if parsed.path == "/api/update":
            ann_id = data.get("id")
            text = data.get("caption_replaced")
            if ann_id is None or text is None:
                return self._send_json({"error": "missing id or caption_replaced"}, status=400)
            ok = self.server.store.update_caption(ann_id, text)
            if not ok:
                return self._send_json({"error": "id not found"}, status=404)
            return self._send_json({"ok": True})

        if parsed.path == "/api/save":
            self.server.store.save()
            return self._send_json({"ok": True, "output": str(self.server.store.output_path)})

        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        return

    def _send_json(self, payload, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def render_index():
    return """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Caption Review</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f4ef;
      --panel: #ffffff;
      --ink: #1c1c1c;
      --accent: #00716d;
      --muted: #6a6a6a;
      --border: #e0ddd7;
    }
    body {
      margin: 0;
      font-family: "Hiragino Sans", "Noto Sans JP", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      padding: 16px 24px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    header h1 {
      font-size: 18px;
      margin: 0;
    }
    label {
      font-size: 12px;
      color: var(--muted);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    select, input, button {
      font-size: 14px;
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #fff;
    }
    button {
      background: var(--accent);
      color: #fff;
      border: none;
      cursor: pointer;
    }
    button.secondary {
      background: #dfeeed;
      color: #004844;
    }
    main {
      padding: 16px 24px 40px;
    }
    .row {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 12px;
      box-shadow: 0 3px 10px rgba(0,0,0,0.04);
    }
    .row header {
      position: static;
      background: none;
      border: none;
      padding: 0 0 8px;
    }
    .meta {
      font-size: 12px;
      color: var(--muted);
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }
    textarea {
      width: 100%;
      min-height: 64px;
      border-radius: 10px;
      border: 1px solid var(--border);
      padding: 8px;
      font-size: 14px;
      font-family: inherit;
      margin-top: 6px;
    }
    .scores {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      font-size: 12px;
      margin: 6px 0 8px;
    }
    .score {
      padding: 4px 8px;
      border-radius: 999px;
      background: #efede7;
      color: #1c1c1c;
      border: 1px solid #e0ddd7;
    }
    .score.good { background: #d7f0e3; border-color: #bfe2cf; }
    .score.bad { background: #f5c6c6; border-color: #e8a9a9; }
    .score.na { background: #f0f0f0; color: var(--muted); }
    .token-chip {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      background: #f0e6d6;
      font-size: 12px;
      margin-right: 4px;
    }
    .edited {
      color: var(--accent);
      font-weight: 600;
    }
    .topline {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 6px;
    }
    .source, .caption, .replaced {
      margin-top: 6px;
    }
    .source strong, .caption strong, .replaced strong {
      display: inline-block;
      min-width: 120px;
    }
    .caption {
      font-size: 12px;
      color: var(--muted);
    }
    .missing {
      margin-top: 6px;
    }
  </style>
</head>
<body>
  <header>
    <h1>Caption Review</h1>
    <label>Sort
      <select id="sort-field"></select>
    </label>
    <label>Order
      <select id="sort-order">
        <option value="asc">asc</option>
        <option value="desc">desc</option>
      </select>
    </label>
    <label>Page size
      <input id="page-size" type="number" min="1" value="50" />
    </label>
    <label>
      <span>Missing tokens</span>
      <select id="missing-only">
        <option value="0">all</option>
        <option value="1">missing only</option>
      </select>
    </label>
    <label>
      <span>Edited</span>
      <select id="edited-only">
        <option value="0">all</option>
        <option value="1">edited only</option>
      </select>
    </label>
    <button id="apply-filters" class="secondary">Apply</button>
    <button id="save-all">Save JSON</button>
    <div id="status"></div>
  </header>
  <main>
    <div id="pager"></div>
    <div id="rows"></div>
  </main>
  <script>
    const state = {
      offset: 0,
      limit: 50,
      sort: "id",
      order: "asc",
      missingOnly: "0",
      editedOnly: "0",
      total: 0,
      fields: [],
      fluencyThreshold: null,
      semanticThreshold: null
    };

    async function fetchMeta() {
      const res = await fetch("/api/metadata");
      const data = await res.json();
      state.total = data.total;
      state.limit = data.page_size;
      state.fields = data.fields;
      state.fluencyThreshold = data.fluency_threshold;
      state.semanticThreshold = data.semantic_threshold;
      initControls();
      await fetchPage();
    }

    function initControls() {
      const sortSelect = document.getElementById("sort-field");
      sortSelect.innerHTML = "";
      const baseFields = ["id", "lm_delta_ppl", "lm_target_ppl", "lm_source_ppl", "semantic_similarity"];
      const fields = Array.from(new Set(baseFields.concat(state.fields))).filter(Boolean);
      fields.forEach(field => {
        const opt = document.createElement("option");
        opt.value = field;
        opt.textContent = field;
        sortSelect.appendChild(opt);
      });
      document.getElementById("page-size").value = state.limit;
    }

    async function fetchPage() {
      const params = new URLSearchParams({
        offset: state.offset,
        limit: state.limit,
        sort: state.sort,
        order: state.order,
        missing_only: state.missingOnly,
        edited_only: state.editedOnly
      });
      const res = await fetch("/api/items?" + params.toString());
      const data = await res.json();
      state.total = data.total;
      renderPage(data.items);
      renderPager();
    }

    function renderPager() {
      const pager = document.getElementById("pager");
      const start = state.offset + 1;
      const end = Math.min(state.offset + state.limit, state.total);
      pager.innerHTML = `
        <div class="meta">Showing ${start} - ${end} / ${state.total}</div>
        <button id="prev">Prev</button>
        <button id="next">Next</button>
      `;
      document.getElementById("prev").onclick = () => {
        state.offset = Math.max(0, state.offset - state.limit);
        fetchPage();
      };
      document.getElementById("next").onclick = () => {
        if (state.offset + state.limit < state.total) {
          state.offset += state.limit;
          fetchPage();
        }
      };
    }

    function renderPage(items) {
      const rows = document.getElementById("rows");
      rows.innerHTML = "";
      items.forEach(item => {
        const row = document.createElement("div");
        row.className = "row";
        row.innerHTML = `
          <div class="topline">
            <div class="meta">
              <span>ID: ${item.id}</span>
              <span>Image: ${item.image_id ?? "-"}</span>
              <span class="${item.edited ? "edited" : ""}">${item.edited ? "edited" : ""}</span>
            </div>
            <div class="scores"></div>
          </div>
          <div class="missing"></div>
          <div class="source"><strong>source</strong> ${item.source_caption ?? ""}</div>
          <div class="source"><strong>caption_replaced</strong> ${item.caption_replaced ?? ""}</div>
          <div class="caption"><strong>caption</strong> ${item.caption ?? ""}</div>
          <div class="replaced">
            <strong>caption_replaced</strong>
            <textarea data-id="${item.id}">${item.caption_replaced ?? ""}</textarea>
            <button class="update" data-id="${item.id}">Update</button>
          </div>
        `;
        const scores = row.querySelector(".scores");
        Object.entries(item.extra_fields || {}).forEach(([key, value]) => {
          const div = document.createElement("div");
          const displayValue = (typeof value === "number")
            ? value.toFixed(3)
            : value;
          div.textContent = `${key}: ${displayValue}`;
          div.className = "score";
          if (value === null || value === undefined || value === "") {
            div.classList.add("na");
          } else if (key === "lm_target_ppl" && state.fluencyThreshold !== null) {
            div.classList.add(value >= state.fluencyThreshold ? "bad" : "good");
          } else if (key === "semantic_similarity" && state.semanticThreshold !== null) {
            div.classList.add(value <= state.semanticThreshold ? "bad" : "good");
          }
          scores.appendChild(div);
        });
        const missing = row.querySelector(".missing");
        if (item.missing_tokens && item.missing_tokens.length) {
          missing.innerHTML = "<strong class='score bad'>missing</strong> " + item.missing_tokens.map(t => `<span class="token-chip">${t}</span>`).join("");
        }
        rows.appendChild(row);
      });
      rows.querySelectorAll("button.update").forEach(btn => {
        btn.onclick = async () => {
          const id = Number(btn.dataset.id);
          const textarea = rows.querySelector(`textarea[data-id="${id}"]`);
          await updateCaption(id, textarea.value);
          btn.textContent = "Updated";
          setTimeout(() => (btn.textContent = "Update"), 800);
        };
      });
    }

    async function updateCaption(id, text) {
      await fetch("/api/update", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({id: id, caption_replaced: text})
      });
    }

    document.getElementById("apply-filters").onclick = async () => {
      state.sort = document.getElementById("sort-field").value;
      state.order = document.getElementById("sort-order").value;
      state.limit = Number(document.getElementById("page-size").value) || state.limit;
      state.missingOnly = document.getElementById("missing-only").value;
      state.editedOnly = document.getElementById("edited-only").value;
      state.offset = 0;
      await fetchPage();
    };

    document.getElementById("save-all").onclick = async () => {
      const res = await fetch("/api/save", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
      const data = await res.json();
      document.getElementById("status").textContent = data.output ? `saved: ${data.output}` : "saved";
    };

    fetchMeta();
  </script>
</body>
</html>
"""


def main():
    args = parse_args()
    base = load_json(args.input_json)
    if "annotations" not in base:
        raise SystemExit("Input JSON must contain annotations.")
    annotations = merge_annotations(base, args.merge_json)
    attach_missing_tokens(annotations, args.failures_jsonl)
    store = CaptionStore(annotations, args.output_json, args.page_size)
    store.set_root(base)

    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    server.store = store
    print(f"Open: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
