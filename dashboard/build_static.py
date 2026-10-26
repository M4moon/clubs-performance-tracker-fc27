#!/usr/bin/env python3
"""Build a static copy of the dashboard with the data baked in, for GitHub Pages.

Usage: python dashboard/build_static.py data/raw results/model docs/index.html

The live dashboard (dashboard/al-haramis-lab.html) reads a database that a scheduled collector
keeps filling. This script injects the saved data as window.__SEED instead, so the same page
works as a plain file with no backend. The header says it is a snapshot and when it was taken.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dir(path):
    out = []
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        out.append(d["data"] if isinstance(d, dict) and isinstance(d.get("data"), dict) else d)
    return out


def replace_once(s, old, new):
    assert s.count(old) == 1, f"expected exactly one match for: {old[:60]}"
    return s.replace(old, new)


def main(raw, model_dir, outp):
    matches = sorted(load_dir(os.path.join(raw, "matches")), key=lambda m: m["ts"])
    snaps = sorted(load_dir(os.path.join(raw, "snapshots")), key=lambda s: s["date"], reverse=True)
    bench = {k: json.load(open(os.path.join(raw, "benchmarks", k + ".json"), encoding="utf-8")) for k in ("div", "near", "top", "events")}
    opps = {str(o["id"]): o for o in load_dir(os.path.join(raw, "opponents"))}
    oppm = {o["matchId"]: o for o in load_dir(os.path.join(raw, "oppmatch"))}
    runs = sorted(load_dir(os.path.join(raw, "modelruns")), key=lambda r: r["trainedAt"])
    model = json.load(open(os.path.join(model_dir, "model_latest.json"), encoding="utf-8"))
    review = json.load(open(os.path.join(model_dir, "review.json"), encoding="utf-8"))
    taken = max(s["takenAt"] for s in snaps)
    seed = {"matches": matches, "snaps": snaps, "roles": {}, "codes": {}, "bench": bench, "opps": opps, "oppm": oppm,
            "model": model, "modelRuns": runs, "review": review, "reviewLoaded": True, "canWrite": False,
            "meta": {"lastRun": taken, "status": "ok"}, "static": True, "staticAt": matches[-1]["ts"]}

    html = open(os.path.join(HERE, "al-haramis-lab.html"), encoding="utf-8").read()
    # the header normally shows when the collector last ran; a snapshot says what it is instead
    html = replace_once(html, '$("#statusLine").innerHTML = S.noStore ?',
                        '$("#statusLine").innerHTML = S.static ? `Static snapshot · newest game ${esc(dateFmt(S.staticAt))} · '
                        '<a href="https://github.com/M4moon/clubs-performance-tracker-fc27">how it was built</a>` : S.noStore ?')
    html = replace_once(html, "This page updates itself when new games are saved, with no reload needed, and double-checks every 2 minutes while it's open.",
                        "This copy is a static snapshot of that data; the live version updates itself as games are saved.")
    payload = json.dumps(seed, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = replace_once(html, "<script>\n(() => {", f"<script>window.__SEED = {payload};</script>\n<script>\n(() => {{")
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    open(outp, "w", encoding="utf-8").write(html)
    print(f"wrote {outp}: {len(matches)} matches, {os.path.getsize(outp) // 1024} KB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
