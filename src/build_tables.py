#!/usr/bin/env python3
"""Turn the raw EA match documents into flat CSV tables.

Usage: python src/build_tables.py data/raw data/processed

Reads data/raw/{matches,oppmatch,opponents,snapshots}/*.json and writes:
  matches.csv               one row per game, our side and theirs; opponent_skill is the opponent's
                            estimated EA skill rating before kick-off (see pregame_skill)
  player_appearances.csv    one row per Al Haramis player per game, with the event-counter metrics
  opponent_appearances.csv  one row per opponent player per game (games from 13 Oct 2026 onwards)
  player_events_long.csv    every raw EA event counter: match, player, code, count
  snapshots.csv             EA's daily club totals
"""
import csv
import datetime
import glob
import json
import os
import sys

SESSION_GAP = 2700  # seconds between kick-offs that start a new session (same rule as the model)
PTS = {"W": 3, "D": 1, "L": 0}


def load_dir(path):
    out = []
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        out.append(d["data"] if isinstance(d, dict) and isinstance(d.get("data"), dict) else d)
    return out


def load_matches(raw):
    ms = sorted(load_dir(os.path.join(raw, "matches")), key=lambda m: m["ts"])
    sess, prev = 0, None
    for m in ms:
        if prev is None or m["ts"] - prev > SESSION_GAP:
            sess += 1
            gi = 0
        gi += 1
        m["_sess"], m["_game"], prev = sess, gi, m["ts"]
    return ms


def event_defs(raw):
    """[(group, key, label, codes, better)] from the community event mapping."""
    e = json.load(open(os.path.join(raw, "benchmarks", "events.json"), encoding="utf-8"))
    return [tuple(r) for r in e["derived"]]


def iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")


def pct(a, b):
    return round(100 * a / b, 1) if b else ""


def write(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def pregame_skill(raw):
    """Each opponent's EA skill rating before kick-off, estimated exactly as the model does (model/train_model.py):
    the rating saved after the game minus the swing that game's result gave the opponent."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
    import train_model
    ctx = train_model.prepare(raw)
    return {m["id"]: round(ctx["opp_skill"](m)) for m in ctx["matches"]}, ctx["swing"]


def main(raw, out):
    os.makedirs(out, exist_ok=True)
    ms = load_matches(raw)
    defs = event_defs(raw)
    opps = {int(o["id"]): o for o in load_dir(os.path.join(raw, "opponents"))}
    post = {d["matchId"]: d["skillRating"] for d in load_dir(os.path.join(raw, "oppskill")) if d.get("skillRating") is not None}
    pre, swing = pregame_skill(raw)

    mrows = []
    for m in ms:
        us, them = m.get("us", {}), m.get("them", {})
        o = opps.get(int(m.get("oppId") or 0), {})
        human_goals = sum(p.get("g", 0) for p in m["players"])
        mrows.append({
            "match_id": m["id"], "kickoff_utc": iso(m["ts"]), "ts": m["ts"], "session": m["_sess"], "game_in_session": m["_game"],
            "type": m["type"], "opponent": m["opp"], "opponent_id": m.get("oppId"),
            "opponent_skill": pre[m["id"]], "opponent_skill_saved": post.get(m["id"], o.get("skillRating", "")),
            "result": m["res"], "points": PTS[m["res"]], "gf": m["gf"], "ga": m["ga"], "gd": m["gf"] - m["ga"],
            "humans": m["humans"], "opp_humans": m["oppHumans"],
            "shots": us.get("shots"), "opp_shots": them.get("shots"),
            "passes_att": us.get("pa"), "passes_made": us.get("pm"), "opp_passes_att": them.get("pa"), "opp_passes_made": them.get("pm"),
            "pass_pct": pct(us.get("pm", 0), us.get("pa", 0)), "pass_share": pct(us.get("pa", 0), us.get("pa", 0) + them.get("pa", 0)),
            "tackles_att": us.get("ta"), "tackles_won": us.get("tm"), "opp_tackles_att": them.get("ta"), "opp_tackles_won": them.get("tm"),
            "human_goals": human_goals, "ai_goals": m["gf"] - human_goals, "opp_avg_rating": them.get("rating", ""),
            "has_events": int(any(p.get("ev") for p in m["players"])),
        })
    mcols = list(mrows[0].keys())
    write(os.path.join(out, "matches.csv"), mrows, mcols)

    arows, erows = [], []
    for m in ms:
        for p in m["players"]:
            ev = p.get("ev") or {}
            r = {"match_id": m["id"], "kickoff_utc": iso(m["ts"]), "ts": m["ts"], "session": m["_sess"], "game_in_session": m["_game"],
                 "type": m["type"], "opponent": m["opp"], "result": m["res"], "points": PTS[m["res"]], "gf": m["gf"], "ga": m["ga"],
                 "humans": m["humans"], "opp_humans": m["oppHumans"],
                 "player": p["p"], "position": p["pos"], "archetype_id": p.get("arch"), "rating": p["r"], "goals": p["g"], "assists": p["a"],
                 "shots": p["sh"], "passes_att": p["pa"], "passes_made": p["pm"], "tackles_att": p["ta"], "tackles_won": p["tm"],
                 "motm": p.get("mom", 0), "seconds": p.get("sec", ""), "idle_seconds": p.get("idle", ""), "red_cards": p.get("red", 0),
                 "saves": p.get("sv", 0), "goals_conceded": p.get("gc", 0), "has_events": int(bool(ev))}
            for _, key, _, codes, _ in defs:
                r["ev_" + key] = sum(int(ev.get(str(c), 0)) for c in codes) if ev else ""
            arows.append(r)
            for code, n in sorted(ev.items(), key=lambda kv: int(kv[0])):
                erows.append({"match_id": m["id"], "player": p["p"], "code": int(code), "count": n})
    acols = list(arows[0].keys())
    write(os.path.join(out, "player_appearances.csv"), arows, acols)
    write(os.path.join(out, "player_events_long.csv"), erows, ["match_id", "player", "code", "count"])

    orows = []
    for om in load_dir(os.path.join(raw, "oppmatch")):
        for p in om.get("players", []):
            orows.append({"match_id": om["matchId"], "opponent": om["opp"], "opponent_id": om["oppId"], "player": p["p"], "position": p["pos"],
                          "rating": p["r"], "goals": p["g"], "assists": p["a"], "shots": p["sh"], "passes_att": p["pa"], "passes_made": p["pm"],
                          "tackles_att": p["ta"], "tackles_won": p["tm"], "motm": p.get("mom", 0), "seconds": p.get("sec", "")})
    orows.sort(key=lambda r: (r["match_id"], r["player"]))
    write(os.path.join(out, "opponent_appearances.csv"), orows, list(orows[0].keys()))

    srows = []
    for s in sorted(load_dir(os.path.join(raw, "snapshots")), key=lambda s: s["date"]):
        srows.append({k: s.get(k) for k in ("date", "takenAt", "skillRating", "currentDivision", "gamesPlayed", "wins", "ties", "losses", "goals", "goalsAgainst")})
    write(os.path.join(out, "snapshots.csv"), srows, list(srows[0].keys()))
    print(f"pre-game opponent skill: saved rating minus {swing} points per result")
    print(f"{len(mrows)} matches, {len(arows)} player appearances, {len(orows)} opponent appearances, {len(erows)} event counts, {len(srows)} snapshots")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
