#!/usr/bin/env python3
"""Standalone collector for EA FC Pro Clubs match data.

Usage: python collector/collect.py --club 387937 --out data/raw [--platform common-gen5]

Saves every league and playoff game EA still lists for the club, in the same document shape as
data/raw/matches/*.json, plus the opponents' player lines, each opponent club's record (and its
skill rating right after the game, in oppskill/) and a daily snapshot of the club's totals. Run it on a schedule: EA only keeps a club's most recent
games, so anything not saved within a day or two is gone.

How this relates to the data in this repo: the games in data/raw were collected by a scheduled
Claude task that read the same EA endpoints through a web-fetch tool (its sandbox could not reach
EA directly) and applied the same rules as this script. This script is a direct-HTTP port of that
pipeline. Its parsing and checks are unit-tested on EA-shaped fixtures (tests/), but it has not
been run against the live EA API from this repo's build environment. EA's API is unofficial and
undocumented, and it may reject requests without browser-like headers.
"""
import argparse
import datetime
import json
import os
import time
import urllib.parse
import urllib.request

BASE = "https://proclubs.ea.com/api/fc/"
HEADERS = {  # EA's edge tends to refuse requests that don't look like a browser
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json", "Referer": "https://www.ea.com/", "Origin": "https://www.ea.com",
}
MATCH_TYPES = {"leagueMatch": "league", "playoffMatch": "playoff"}
MAX_REAL_TIME = 1500   # seconds; EA sometimes reports impossible per-player real time


# ---------- HTTP ----------
def url(endpoint, cb, **params):
    """EA's CDN serves hours-old cached copies unless the URL is new. The cache key ignores a
    parameter named "_" but not "cb", so every request carries cb=<unix time>."""
    params["cb"] = cb
    return BASE + endpoint + "?" + urllib.parse.urlencode(params)


def get_json(u, retries=2):
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(u, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if i == retries:
                raise
            time.sleep(2 * (i + 1))


# ---------- parsing ----------
def num(v, kind=int):
    """EA sends most numbers as strings."""
    if v is None or v == "":
        return 0
    try:
        return kind(v)
    except (TypeError, ValueError):
        return kind(float(v))


def parse_events(player):
    """match_event_aggregate_0..3 hold 'code:count' pairs separated by commas. Merge them into one
    {code: count} dict; if a code repeats, the first value wins."""
    ev = {}
    for i in range(4):
        raw = player.get(f"match_event_aggregate_{i}")
        if not raw:
            continue
        for pair in str(raw).split(","):
            if ":" not in pair:
                continue
            code, count = pair.split(":", 1)
            code, count = code.strip(), count.strip()
            if code.isdigit() and code not in ev:
                ev[code] = num(count)
    return ev


def _player(p, with_events):
    out = {"p": p.get("playername", ""), "pos": p.get("pos", ""), "arch": num(p.get("archetypeid")), "r": round(num(p.get("rating"), float), 2),
           "g": num(p.get("goals")), "a": num(p.get("assists")), "sh": num(p.get("shots")), "pa": num(p.get("passattempts")),
           "pm": num(p.get("passesmade")), "ta": num(p.get("tackleattempts")), "tm": num(p.get("tacklesmade")), "mom": num(p.get("mom")),
           "sec": num(p.get("realtimegame")), "red": num(p.get("redcards")), "sv": num(p.get("saves")), "gc": num(p.get("goalsconceded"))}
    if with_events:
        out["idle"] = num(p.get("realtimeidle"))
        ev = parse_events(p)
        if ev:
            out["ev"] = ev
    return out


def cap_real_time(players):
    """If a player's real time exceeds MAX_REAL_TIME, use the largest teammate value at or under it."""
    ok = [p["sec"] for p in players if p["sec"] <= MAX_REAL_TIME]
    for p in players:
        if p["sec"] > MAX_REAL_TIME:
            p["note"] = f"EA reported {p['sec']}s real time; capped"
            p["sec"] = max(ok) if ok else MAX_REAL_TIME
    return players


def parse_match(raw, club_id, mtype, fetched_at):
    """One EA match object -> (match document, opponent-lines document)."""
    cid = str(club_id)
    opp_id = next(k for k in raw["clubs"] if k != cid)
    clubs, agg = raw["clubs"], raw.get("aggregate", {})
    ours = cap_real_time([_player(p, True) for p in raw["players"].get(cid, {}).values()])
    theirs = cap_real_time([_player(p, False) for p in raw["players"].get(opp_id, {}).values()])
    gf, ga = num(clubs[cid].get("goals")), num(clubs[cid].get("goalsAgainst"))
    a_us, a_th = agg.get(cid, {}), agg.get(opp_id, {})
    opp_name = (clubs[opp_id].get("details") or {}).get("name") or clubs[opp_id].get("name", "")
    ratings = [p["r"] for p in theirs]
    doc = {"id": str(raw["matchId"]), "ts": num(raw["timestamp"]), "type": mtype, "opp": opp_name, "oppId": num(opp_id),
           "gf": gf, "ga": ga, "res": "W" if gf > ga else "L" if gf < ga else "D",
           "humans": len(ours), "oppHumans": len(theirs),
           "us": {"shots": num(a_us.get("shots")), "pm": num(a_us.get("passesmade")), "pa": num(a_us.get("passattempts")),
                  "tm": num(a_us.get("tacklesmade")), "ta": num(a_us.get("tackleattempts")), "ast": num(a_us.get("assists")), "hg": num(a_us.get("goals"))},
           "them": {"shots": num(a_th.get("shots")), "pm": num(a_th.get("passesmade")), "pa": num(a_th.get("passattempts")),
                    "tm": num(a_th.get("tacklesmade")), "ta": num(a_th.get("tackleattempts")),
                    "rating": round(sum(ratings) / len(ratings), 2) if ratings else None},
           "players": ours, "source": "ea", "fetchedAt": fetched_at}
    problems = validate(doc)
    if problems:
        doc["flag"], doc["flagNote"] = "unverified", "; ".join(problems)
    opp_doc = {"matchId": doc["id"], "oppId": doc["oppId"], "opp": opp_name, "humans": sum(1 for p in theirs if p["sec"] > 0),
               "players": theirs, "source": "ea", "fetchedAt": fetched_at}
    return doc, opp_doc


def validate(doc):
    """Player lines must add up to EA's club aggregate. Returns a list of problems (empty = fine)."""
    out = []
    for k_player, k_agg in (("sh", "shots"), ("pa", "pa"), ("pm", "pm"), ("ta", "ta"), ("tm", "tm")):
        s = sum(p[k_player] for p in doc["players"])
        if s != doc["us"][k_agg]:
            out.append(f"{k_player} players {s} vs aggregate {doc['us'][k_agg]}")
    g = sum(p["g"] for p in doc["players"])
    if g != doc["us"]["hg"]:
        out.append(f"goals players {g} vs aggregate {doc['us']['hg']}")
    if g > doc["gf"]:
        out.append(f"player goals {g} exceed club goals {doc['gf']}")
    for p in doc["players"]:   # code 215 is close to, but not exactly, passes made
        if "ev" in p and abs(p["ev"].get("215", p["pm"]) - p["pm"]) > 3:
            out.append(f"{p['p']} event 215 {p['ev'].get('215')} vs passes made {p['pm']}")
    return out


def stale_opponents(last_opponents, known_opp_ids):
    """EA's overallStats lists the opponents of the club's 5 latest games (lastOpponent0..4). If one
    is in neither the saved games nor the match lists just fetched, those lists were a stale copy."""
    return [o for o in last_opponents if o and int(o) not in known_opp_ids]


# ---------- storage ----------
def load_ids(out, coll):
    d = os.path.join(out, coll)
    return {f[:-5] for f in os.listdir(d)} if os.path.isdir(d) else set()


def save(out, coll, doc_id, doc):
    os.makedirs(os.path.join(out, coll), exist_ok=True)
    with open(os.path.join(out, coll, f"{doc_id}.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False, sort_keys=True)


def first(x):
    return x[0] if isinstance(x, list) and x else x


# ---------- run ----------
def run(club, out, platform):
    cb = int(time.time())
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    known = load_ids(out, "matches")
    known_opp = set()
    for mid in known:
        known_opp.add(int(json.load(open(os.path.join(out, "matches", mid + ".json")))["oppId"]))

    def fetch_lists(cbv, n):
        found = {}
        for ea_type, mtype in MATCH_TYPES.items():
            for raw in get_json(url("clubs/matches", cbv, matchType=ea_type, platform=platform, clubIds=club, maxResultCount=n)) or []:
                found[str(raw["matchId"])] = (raw, mtype)
        return found

    found = fetch_lists(cb, 20)
    stats = first(get_json(url("clubs/overallStats", cb, platform=platform, clubIds=club)))
    listed_opp = {int(next(k for k in r["clubs"] if k != str(club))) for r, _ in found.values()}
    missing = stale_opponents([stats.get(f"lastOpponent{i}") for i in range(5)], known_opp | listed_opp)
    if missing:   # stale lists: ask again with a different cache-buster
        found.update(fetch_lists(cb + 1, 10))
        listed_opp = {int(next(k for k in r["clubs"] if k != str(club))) for r, _ in found.values()}
        missing = stale_opponents(missing, known_opp | listed_opp)

    new = [(mid, raw, t) for mid, (raw, t) in found.items() if mid not in known]
    for mid, raw, mtype in sorted(new, key=lambda x: int(x[1]["timestamp"])):
        doc, opp_doc = parse_match(raw, club, mtype, now)
        save(out, "matches", mid, doc)
        save(out, "oppmatch", mid, opp_doc)
        o = first(get_json(url("clubs/overallStats", cb, platform=platform, clubIds=doc["oppId"]))) or {}
        # the opponent's rating right after this game; the model subtracts this game's own effect on it
        save(out, "oppskill", mid, {"matchId": mid, "oppId": doc["oppId"], "skillRating": num(o.get("skillRating")) or None, "fetchedAt": now})
        path = os.path.join(out, "opponents", f"{doc['oppId']}.json")
        prev = json.load(open(path)) if os.path.exists(path) else {}
        save(out, "opponents", doc["oppId"], {
            "id": doc["oppId"], "name": doc["opp"], "skillRating": num(o.get("skillRating")), "gamesPlayed": num(o.get("gamesPlayed")),
            "wins": num(o.get("wins")), "ties": num(o.get("ties")), "losses": num(o.get("losses")), "goals": num(o.get("goals")),
            "goalsAgainst": num(o.get("goalsAgainst")), "promotions": num(o.get("promotions")), "relegations": num(o.get("relegations")),
            "winStreak": num(o.get("wstreak")), "unbeatenStreak": num(o.get("unbeatenstreak")), "playoffGames": num(o.get("gamesPlayedPlayoff")),
            "statsAt": now, "matches": sorted(set(prev.get("matches", [])) | {mid})})

    today = now[:10]
    if today not in load_ids(out, "snapshots"):   # written after the games, so the record never runs ahead of them
        members = get_json(url("members/stats", cb, platform=platform, clubId=club)).get("members", [])
        save(out, "snapshots", today, {
            "date": today, "takenAt": now, "skillRating": num(stats.get("skillRating")), "gamesPlayed": num(stats.get("gamesPlayed")),
            "wins": num(stats.get("wins")), "ties": num(stats.get("ties")), "losses": num(stats.get("losses")), "goals": num(stats.get("goals")),
            "goalsAgainst": num(stats.get("goalsAgainst")), "promotions": num(stats.get("promotions")), "relegations": num(stats.get("relegations")),
            "currentDivision": num(stats.get("currentDivision")) or None, "unbeatenStreak": num(stats.get("unbeatenstreak")), "winStreak": num(stats.get("wstreak")),
            "members": [{"p": m.get("name"), "gp": num(m.get("gamesPlayed")), "wr": num(m.get("winRate")), "g": num(m.get("goals")), "a": num(m.get("assists")),
                         "shotPct": num(m.get("shotSuccessRate")), "pm": num(m.get("passesMade")), "passPct": num(m.get("passSuccessRate")),
                         "r": num(m.get("ratingAve"), float), "tm": num(m.get("tacklesMade")), "tklPct": num(m.get("tackleSuccessRate")),
                         "ovr": num(m.get("proOverall")), "mom": num(m.get("manOfTheMatch")), "red": num(m.get("redCards")), "fav": m.get("favoritePosition")}
                        for m in members]})
    status = "ok" if not missing else "EA list missing recent game vs club " + ", ".join(map(str, missing))
    save(out, "meta", "state", {"lastRun": now, "status": status, "lastNew": len(new)})
    print(f"{len(new)} new games saved; status: {status}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--club", type=int, default=387937)
    ap.add_argument("--platform", default="common-gen5")
    ap.add_argument("--out", default="data/raw")
    a = ap.parse_args()
    run(a.club, a.out, a.platform)
