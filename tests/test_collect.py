"""Tests for the collector's parsing and checks.

There is no network access here, so each test rebuilds an EA-shaped match (string numbers,
players keyed by club then player id, event counters as 'code:count' strings) from a game that
was actually saved in data/raw, runs it through the collector, and checks that the saved
document comes back exactly.
"""
import copy
import glob
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "collector"))
import collect  # noqa: E402

CLUB = 387937
RAW = os.path.join(ROOT, "data", "raw")


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def saved(coll, doc_id):
    return read(os.path.join(RAW, coll, f"{doc_id}.json"))


def ea_player(p, events=True):
    out = {"playername": p["p"], "pos": p["pos"], "archetypeid": str(p["arch"]), "rating": f"{p['r']:.2f}", "goals": str(p["g"]),
           "assists": str(p["a"]), "shots": str(p["sh"]), "passattempts": str(p["pa"]), "passesmade": str(p["pm"]),
           "tackleattempts": str(p["ta"]), "tacklesmade": str(p["tm"]), "mom": str(p["mom"]), "realtimegame": str(p["sec"]),
           "redcards": str(p.get("red", 0)), "saves": str(p.get("sv", 0)), "goalsconceded": str(p.get("gc", 0))}
    if events:
        out["realtimeidle"] = str(p.get("idle", 0))
        pairs = [f"{c}:{n}" for c, n in p.get("ev", {}).items()]
        for i in range(4):   # EA spreads the pairs over four fields
            out[f"match_event_aggregate_{i}"] = ",".join(pairs[i::4])
    return out


def ea_match(m, om):
    cid, oid = str(CLUB), str(m["oppId"])
    return {"matchId": m["id"], "timestamp": str(m["ts"]),
            "clubs": {cid: {"goals": str(m["gf"]), "goalsAgainst": str(m["ga"]), "details": {"name": "Al Haramis FC"}},
                      oid: {"goals": str(m["ga"]), "goalsAgainst": str(m["gf"]), "details": {"name": m["opp"]}}},
            "players": {cid: {str(1000 + i): ea_player(p) for i, p in enumerate(m["players"])},
                        oid: {str(2000 + i): ea_player(p, False) for i, p in enumerate(om["players"])}},
            "aggregate": {cid: {"shots": str(m["us"]["shots"]), "passesmade": str(m["us"]["pm"]), "passattempts": str(m["us"]["pa"]),
                                "tacklesmade": str(m["us"]["tm"]), "tackleattempts": str(m["us"]["ta"]), "assists": str(m["us"]["ast"]),
                                "goals": str(m["us"]["hg"])},
                          oid: {"shots": str(m["them"]["shots"]), "passesmade": str(m["them"]["pm"]), "passattempts": str(m["them"]["pa"]),
                                "tacklesmade": str(m["them"]["tm"]), "tackleattempts": str(m["them"]["ta"])}}}


class ParseSavedGames(unittest.TestCase):
    def test_round_trip_every_game_with_opponent_lines(self):
        ids = [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(RAW, "oppmatch", "*.json"))]
        self.assertGreater(len(ids), 60)
        for mid in ids:
            m, om = saved("matches", mid), saved("oppmatch", mid)
            if m.get("flag"):   # saved from an incomplete EA response; can't be rebuilt exactly
                continue
            doc, opp_doc = collect.parse_match(ea_match(m, om), CLUB, m["type"], m["fetchedAt"])
            for k in ("id", "ts", "type", "opp", "oppId", "gf", "ga", "res", "humans", "us"):
                self.assertEqual(doc[k], m[k], f"{mid} {k}")
            # oppHumans counts every listed opponent; one saved game counted only those with game time
            left_early = sum(1 for p in om["players"] if not p["sec"])
            self.assertIn(m["oppHumans"], (doc["oppHumans"], doc["oppHumans"] - left_early), f"{mid} oppHumans")
            self.assertEqual(doc["them"]["shots"], m["them"]["shots"])
            self.assertAlmostEqual(doc["them"]["rating"], m["them"]["rating"], places=1)
            for got, want in zip(doc["players"], m["players"]):
                for k in ("p", "pos", "arch", "r", "g", "a", "sh", "pa", "pm", "ta", "tm", "mom", "sec"):
                    self.assertEqual(got[k], want[k], f"{mid} {want['p']} {k}")
                self.assertEqual(got.get("ev", {}), {c: n for c, n in want.get("ev", {}).items()}, f"{mid} {want['p']} events")
            self.assertEqual([p["p"] for p in opp_doc["players"]], [p["p"] for p in om["players"]])


class Checks(unittest.TestCase):
    def setUp(self):
        mid = os.path.basename(sorted(glob.glob(os.path.join(RAW, "oppmatch", "*.json")))[-1])[:-5]
        self.m, self.om = saved("matches", mid), saved("oppmatch", mid)

    def test_clean_game_passes_validation(self):
        doc, _ = collect.parse_match(ea_match(self.m, self.om), CLUB, "league", "now")
        self.assertNotIn("flag", doc)

    def test_player_lines_that_do_not_add_up_are_flagged(self):
        raw = ea_match(self.m, self.om)
        first = next(iter(raw["players"][str(CLUB)].values()))
        first["shots"] = str(int(first["shots"]) + 2)
        doc, _ = collect.parse_match(raw, CLUB, "league", "now")
        self.assertEqual(doc["flag"], "unverified")
        self.assertIn("sh players", doc["flagNote"])

    def test_real_time_over_1500_seconds_is_capped(self):
        raw = ea_match(self.m, self.om)
        first = next(iter(raw["players"][str(CLUB)].values()))
        first["realtimegame"] = "86400"
        doc, _ = collect.parse_match(raw, CLUB, "league", "now")
        capped = doc["players"][0]
        self.assertLessEqual(capped["sec"], 1500)
        self.assertIn("capped", capped["note"])

    def test_event_pairs_merge_and_first_value_wins(self):
        ev = collect.parse_events({"match_event_aggregate_0": "0:5,215:17", "match_event_aggregate_1": "215:99,6:2",
                                   "match_event_aggregate_2": "", "match_event_aggregate_3": "junk,111:12"})
        self.assertEqual(ev, {"0": 5, "215": 17, "6": 2, "111": 12})

    def test_stale_list_detection(self):
        self.assertEqual(collect.stale_opponents(["1", "2", "3"], {1, 2, 3}), [])
        self.assertEqual(collect.stale_opponents(["1", "9", None], {1, 2}), ["9"])

    def test_every_request_uses_the_cb_cache_buster(self):
        u = collect.url("clubs/matches", 1792609200, matchType="leagueMatch", platform="common-gen5", clubIds=CLUB, maxResultCount=20)
        self.assertIn("cb=1792609200", u)
        self.assertNotIn("_=", u)
        self.assertIn("maxResultCount=20", u)


class SavedDataIntegrity(unittest.TestCase):
    """The checks the collector applies, run over everything already saved."""

    def test_saved_games_add_up(self):
        bad = []
        for f in glob.glob(os.path.join(RAW, "matches", "*.json")):
            m = read(f)
            if m.get("flag"):
                continue
            probs = [p for p in collect.validate(copy.deepcopy(m)) if "event 215" not in p]
            if probs:
                bad.append((m["id"], probs))
        self.assertEqual(bad, [])

    def test_results_match_scores(self):
        for f in glob.glob(os.path.join(RAW, "matches", "*.json")):
            m = read(f)
            self.assertEqual(m["res"], "W" if m["gf"] > m["ga"] else "L" if m["gf"] < m["ga"] else "D", m["id"])


if __name__ == "__main__":
    unittest.main()
