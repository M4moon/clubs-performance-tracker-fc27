#!/usr/bin/env python3
"""Al Haramis FC Lab - season review: what the model said vs what happened.

Usage: python3 -I review_model.py <db_dir> <out_json>
<db_dir> holds ArtifactData dumps: matches/, opponents/, modelruns/ (each file raw or {"id":..,"data":{..}}).
Writes one JSON document for ArtifactData collection "review", doc "latest".

For every session, the XGBoost setup used by the collector's model (same features, same settings)
is trained only on games from earlier sessions and asked to predict each game of the session for the
lineup that actually played, plus each player's numbers. That is what the system would have said
before kick-off. Separately, the forecasts the system actually published (collection "modelruns")
are matched to the games that followed them, with how closely the lineup followed the recommendation.
"""
import json, glob, os, sys, math, datetime
import numpy as np
import xgboost as xgb

SESSION_GAP = 2700
MIN_APPS = 3
SEEDS = [0, 1, 2, 3, 4]
PARAMS = dict(n_estimators=150, learning_rate=0.05, max_depth=2, min_child_weight=3,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, objective="reg:squarederror")
PLAYER_TARGETS = ["r", "g", "a", "sh", "pm", "tm"]
POS = {"goalkeeper": 0, "defender": 1, "midfielder": 2, "forward": 3}
PLAYOFF_WINDOW = 4 * 86400   # a season's playoff run lasts at most 4 days from its first playoff game


def load(dirp):
    out = []
    for f in glob.glob(os.path.join(dirp, "*.json")):
        d = json.load(open(f))
        out.append(d["data"] if isinstance(d, dict) and isinstance(d.get("data"), dict) else d)
    return out


def fit_predict(X, y, Xp):
    preds = []
    for s in SEEDS:
        m = xgb.XGBRegressor(random_state=s, **PARAMS)
        m.fit(X, y)
        preds.append(m.predict(Xp))
    return np.mean(preds, axis=0)


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def wdl(mu, sigma):
    sigma = max(sigma, 0.8)
    pw = 1 - norm_cdf((0.5 - mu) / sigma)
    pl = norm_cdf((-0.5 - mu) / sigma)
    return {"W": round(pw, 3), "D": round(max(0.0, 1 - pw - pl), 3), "L": round(pl, 3)}


def iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(db, outp):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    matches = sorted(load(os.path.join(db, "matches")), key=lambda m: m["ts"])
    opps = {int(o["id"]): o for o in load(os.path.join(db, "opponents")) if o.get("id") is not None}
    runs = sorted([r for r in load(os.path.join(db, "modelruns")) if r.get("trainedAt")], key=lambda r: r["trainedAt"])

    # seasons: a season ends when its playoff run is over
    season, first_po = 1, None
    for m in matches:
        if first_po is not None and m["ts"] > first_po + PLAYOFF_WINDOW:
            season += 1; first_po = None
        if m.get("type") == "playoff" and first_po is None:
            first_po = m["ts"]
        m["_season"] = season
    seasons = []
    for s in sorted(set(m["_season"] for m in matches)):
        g = [m for m in matches if m["_season"] == s]
        po = [m for m in g if m.get("type") == "playoff"]
        seasons.append({"n": s, "start": iso(g[0]["ts"]), "end": iso(g[-1]["ts"]), "games": len(g),
                        "playoffStart": iso(po[0]["ts"]) if po else None, "playoffWindowEnds": iso(po[0]["ts"] + PLAYOFF_WINDOW) if po else None,
                        "complete": bool(po) and (s < season or datetime.datetime.now().timestamp() > po[0]["ts"] + PLAYOFF_WINDOW)})

    sess, prev = 0, None
    for m in matches:
        if prev is None or m["ts"] - prev > SESSION_GAP:
            sess += 1; gi = 0
        gi += 1; m["_sess"], m["_game"], prev = sess, gi, m["ts"]
        m["_present"] = [p for p in m.get("players", []) if (p.get("sec") or 0) > 0]
    for i, m in enumerate(matches):
        prev5 = matches[max(0, i - 5):i]
        m["_form"] = (float(np.mean([1.0 if g["gf"] > g["ga"] else 0.0 for g in prev5])) if prev5 else float("nan"),
                      float(np.mean([g["gf"] - g["ga"] for g in prev5])) if prev5 else float("nan"))

    skills = [o["skillRating"] for o in opps.values() if o.get("skillRating")]
    med_skill = float(np.median(skills)) if skills else 1450.0
    def opp_skill(m):
        o = opps.get(int(m.get("oppId", 0) or 0))
        return float(o["skillRating"]) if o and o.get("skillRating") else med_skill

    apps = {}
    for m in matches:
        for p in m["_present"]:
            apps[p["p"]] = apps.get(p["p"], 0) + 1
    regulars = sorted([n for n, c in apps.items() if c >= MIN_APPS], key=lambda n: -apps[n])

    def mrow(present, opp_h, skill, playoff, game, form):
        names = set(present); pos = list(present.values())
        return [len(names), opp_h, skill, playoff, min(game, 6), len([n for n in names if n not in regulars]),
                pos.count("defender") + pos.count("goalkeeper"), pos.count("midfielder"), pos.count("forward")] + \
               [1 if n in names else 0 for n in regulars] + [form[0], form[1]]
    X = np.array([mrow({p["p"]: p["pos"] for p in m["_present"]}, m.get("oppHumans", 3), opp_skill(m),
                       1 if m.get("type") == "playoff" else 0, m["_game"], m["_form"]) for m in matches], float)
    YF = np.array([m["gf"] for m in matches], float); YA = np.array([m["ga"] for m in matches], float)
    y = YF - YA

    # ---------- team: walk-forward by session ----------
    pred = {}
    resid = []
    for s in sorted(set(m["_sess"] for m in matches)):
        tr = [i for i, m in enumerate(matches) if m["_sess"] < s]
        te = [i for i, m in enumerate(matches) if m["_sess"] == s]
        if len(tr) < 10: continue
        sigma = float(np.sqrt(np.mean(np.square(resid)))) if len(resid) >= 5 else float(np.std(y[tr]))
        p = fit_predict(X[tr], y[tr], X[te]); pf = fit_predict(X[tr], YF[tr], X[te]); pa = fit_predict(X[tr], YA[tr], X[te])
        base = float(np.mean(y[tr])); bgf = float(np.mean(YF[tr])); bga = float(np.mean(YA[tr]))
        bp = wdl(base, float(np.std(y[tr])))
        for k, i in enumerate(te):
            pr = wdl(float(p[k]), sigma)
            pred[i] = {"gd": round(float(p[k]), 2), "gf": round(float(max(pf[k], 0)), 2), "ga": round(float(max(pa[k], 0)), 2),
                       "W": pr["W"], "D": pr["D"], "L": pr["L"], "trainGames": len(tr),
                       "base": {"gd": round(base, 2), "gf": round(bgf, 2), "ga": round(bga, 2), "W": bp["W"], "D": bp["D"], "L": bp["L"]}}
        resid += [float(y[i] - p[k]) for k, i in enumerate(te)]

    # ---------- published forecasts (what the system actually showed) ----------
    def run_before(ts):
        t = iso(ts); r = [x for x in runs if x["trainedAt"] < t]
        return r[-1] if r else None

    games = []
    for i, m in enumerate(matches):
        lineup = sorted(p["p"] for p in m["_present"])
        g = {"id": m["id"], "ts": m["ts"], "season": m["_season"], "type": m.get("type", "league"), "sess": m["_sess"],
             "opp": m.get("opp"), "gf": m["gf"], "ga": m["ga"], "res": m.get("res"), "humans": len(lineup), "lineup": lineup,
             "oppSkill": round(opp_skill(m))}
        if i in pred: g["pred"] = pred[i]
        r = run_before(m["ts"])
        if r and r.get("bestLineup"):
            rec = list(r["bestLineup"]); hit = [n for n in rec if n in lineup]
            extra = [n for n in lineup if n not in rec]
            g["sys"] = {"run": r["trainedAt"], "rec": rec, "probs": r.get("bestProbs"),
                        "overlap": round(len(hit) / len(rec), 2) if rec else None, "missing": [n for n in rec if n not in lineup],
                        "extra": extra, "followed": bool(rec) and not extra and len(hit) == len(rec)}
        games.append(g)

    # ---------- players: walk-forward by session ----------
    rows, meta, hist_r = [], [], {}
    for mi, m in enumerate(matches):
        for p in m["_present"]:
            h = hist_r.get(p["p"], [])[-5:]
            rows.append([opp_skill(m), m.get("oppHumans", 3), len(m["_present"]), min(m["_game"], 6),
                         1 if m.get("type") == "playoff" else 0, POS.get(p["pos"], 2)] + [1 if p["p"] == n else 0 for n in regulars]
                        + [float(np.mean(h)) if h else float("nan")])
            meta.append((m["_sess"], mi, p))
        for p in m["_present"]: hist_r.setdefault(p["p"], []).append(p["r"])
    PX = np.array(rows, float); psess = np.array([s for s, _, _ in meta])
    papps = [{"m": matches[mi]["id"], "p": p["p"], "pos": p["pos"], "act": {k: float(p.get(k) or 0) for k in PLAYER_TARGETS}}
             for _, mi, p in meta]
    for key in PLAYER_TARGETS:
        py = np.array([float(p.get(key) or 0) for _, _, p in meta])
        errs = []
        for s in sorted(set(psess)):
            tr, te = psess < s, psess == s
            if tr.sum() < 30: continue
            tol = float(np.sqrt(np.mean(np.square(errs)))) if len(errs) >= 20 else float(np.std(py[tr]))
            pr = fit_predict(PX[tr], py[tr], PX[te])
            tri = np.where(tr)[0]
            for k, i in enumerate(np.where(te)[0]):
                nm = meta[i][2]["p"]; hist = [py[j] for j in tri if meta[j][2]["p"] == nm]
                b = float(np.mean(hist)) if hist else float(np.mean(py[tr]))
                a = papps[i]
                a.setdefault("pred", {})[key] = round(float(max(pr[k], 0)), 2)
                a.setdefault("base", {})[key] = round(b, 2)
                a.setdefault("tol", {})[key] = round(tol, 2)
                a["newPlayer"] = not hist
            errs += list(py[te] - pr)
    papps = [a for a in papps if "pred" in a]

    out = {"computedAt": now, "version": 1, "seasonRule": "A season ends when its playoff run is over (playoff games within 4 days of the first one).",
           "method": "Each session is predicted by the collector's XGBoost setup trained only on games from earlier sessions, for the lineup that actually played.",
           "seasons": seasons, "games": games, "apps": papps, "nRuns": len(runs),
           "firstRun": runs[0]["trainedAt"] if runs else None}
    json.dump(out, open(outp, "w"), separators=(",", ":"))
    print("games", len(games), "predicted", sum(1 for g in games if "pred" in g), "with published forecast", sum(1 for g in games if "sys" in g),
          "player apps predicted", len(papps), "bytes", os.path.getsize(outp))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
