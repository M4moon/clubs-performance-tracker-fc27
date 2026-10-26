#!/usr/bin/env python3
# ---------------------------------------------------------------------------------------------
# Al Haramis FC Lab - shared core for the ONE prediction model the dashboard uses:
#   Hierarchical Poisson / Negative Binomial goals model (Dixon-Coles family, no low-score tweak).
# Chosen on 23 Oct 2026 after a clean comparison (see SELECTION below). Do not add other models.
#
# Opponent strength = the opponent's EA skill rating as it stood BEFORE kick-off (estimated):
#   the rating the collector fetched after the game (collection "oppskill", one doc per match,
#   falling back to the "opponents" doc) minus the swing that this game's result gave the opponent.
#   The per-result swing is estimated from our own daily snapshots.
# ---------------------------------------------------------------------------------------------
import json, glob, os, sys, math, itertools, datetime, warnings
import numpy as np
from scipy import stats as _st
from sklearn.linear_model import PoissonRegressor as _PoissonReg
warnings.filterwarnings("ignore")

SESSION_GAP = 2700          # seconds between games that starts a new session
MIN_APPS = 3                # games needed before a player gets their own effect
MIN_TRAIN = 10              # games needed before a session is forecast
ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]   # shrinkage strengths tried (picked by log loss on earlier sessions)
DEFAULT_ALPHA = 1.0
DEFAULT_SWING = 12.5        # EA skill points a club gains for a win / loses for a loss (fallback)
MAXG = 15
MODEL_KEY, MODEL_NAME = "poisson", "Hierarchical Poisson / Negative Binomial"
SELECTION = {
    "date": "2026-10-23", "games": 72, "forecastGames": 62,
    "method": "Each session predicted only from earlier sessions; each model's settings tuned only on earlier sessions too; "
              "opponent strength = estimated pre-game EA skill rating. Lower is better on every column.",
    "results": [
        {"model": "Hierarchical Poisson / Negative Binomial", "gdMAE": 2.13, "logLoss": 0.873, "brier": 0.515, "scorelineMAE": 1.57, "chosen": True},
        {"model": "Bradley-Terry / Kalman filter", "gdMAE": 2.25, "logLoss": 0.883, "brier": 0.531, "scorelineMAE": 1.61},
        {"model": "XGBoost (gradient-boosted trees)", "gdMAE": 2.28, "logLoss": 0.924, "brier": 0.561, "scorelineMAE": 1.65},
        {"model": "Simple average", "gdMAE": 2.34, "logLoss": 0.924, "brier": 0.561, "scorelineMAE": 1.63}],
    "why": "Best on goal difference, win/draw/loss probabilities (log loss and Brier) and scorelines. XGBoost was no better "
           "than the simple average once the post-game rating leak was removed.",
}
PLAYER_TARGETS = [("r", "Rating"), ("g", "Goals"), ("a", "Assists"), ("sh", "Shots"),
                  ("pm", "Passes made"), ("pa", "Passes tried"), ("tm", "Tackles won")]
POS = {"goalkeeper": 0, "defender": 1, "midfielder": 2, "forward": 3}


def load(dirp):
    out = []
    for f in glob.glob(os.path.join(dirp, "*.json")):
        d = json.load(open(f))
        out.append(d["data"] if isinstance(d, dict) and isinstance(d.get("data"), dict) else d)
    return out


def iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s):
    try: return datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception: return None


# ------------------------------- opponent strength before kick-off -------------------------------
def estimate_swing(snaps):
    """Least-squares EA skill change per win and per loss from consecutive daily snapshots in the same division."""
    s = sorted([x for x in snaps if x.get("date") and x.get("skillRating") is not None], key=lambda x: x["date"])
    X, y = [], []
    for a, b in zip(s, s[1:]):
        if a.get("currentDivision") != b.get("currentDivision"): continue
        w, l = b.get("wins", 0) - a.get("wins", 0), b.get("losses", 0) - a.get("losses", 0)
        if w + l <= 0 or w < 0 or l < 0: continue
        X.append([w, l]); y.append(b["skillRating"] - a["skillRating"])
    if len(X) < 4: return DEFAULT_SWING, "default"
    c, *_ = np.linalg.lstsq(np.array(X, float), np.array(y, float), rcond=None)
    sw = (c[0] - c[1]) / 2
    if not np.isfinite(sw): return DEFAULT_SWING, "default"
    return float(min(30.0, max(5.0, sw))), "snapshots"


def make_opp_skill(matches, opps, oppskill, swing):
    """Returns opp_skill(m) -> estimated pre-game rating, and a quality summary."""
    raw = {}
    q = {"perMatch": 0, "fallback": 0, "missing": 0, "lateOver24h": 0}
    for m in matches:
        d = oppskill.get(str(m["id"]))
        val, at = None, None
        if d and d.get("skillRating") is not None:
            val, at = float(d["skillRating"]), parse_iso(d.get("fetchedAt")); q["perMatch"] += 1
        else:
            o = opps.get(int(m.get("oppId", 0) or 0))
            if o and o.get("skillRating") is not None:
                val, at = float(o["skillRating"]), parse_iso(o.get("statsAt")); q["fallback"] += 1
        if val is None:
            q["missing"] += 1; continue
        if at and at - m["ts"] > 86400: q["lateOver24h"] += 1
        # undo this game's own effect on the opponent's rating
        opp_res = 1 if m["ga"] > m["gf"] else -1 if m["gf"] > m["ga"] else 0
        raw[m["id"]] = val - swing * opp_res
    med = float(np.median(list(raw.values()))) if raw else 1450.0
    return (lambda m: raw.get(m["id"], med)), q, med


# ------------------------------------------- data prep -------------------------------------------
def prepare(db):
    matches = sorted(load(os.path.join(db, "matches")), key=lambda m: m["ts"])
    opps = {int(o["id"]): o for o in load(os.path.join(db, "opponents")) if o.get("id") is not None}
    oppskill = {str(d.get("matchId")): d for d in load(os.path.join(db, "oppskill")) if d.get("matchId")}
    snaps = sorted(load(os.path.join(db, "snapshots")), key=lambda s: s.get("date", ""))
    sess, prev, gi = 0, None, 0
    for m in matches:
        if prev is None or m["ts"] - prev > SESSION_GAP:
            sess += 1; gi = 0
        gi += 1; m["_sess"], m["_game"], prev = sess, gi, m["ts"]
        m["_present"] = [p for p in m.get("players", []) if (p.get("sec") or 0) > 0]
    swing, swing_src = estimate_swing(snaps)
    opp_skill, oq, med = make_opp_skill(matches, opps, oppskill, swing)
    apps = {}
    for m in matches:
        for p in m["_present"]: apps[p["p"]] = apps.get(p["p"], 0) + 1
    regulars = sorted([n for n, c in apps.items() if c >= MIN_APPS], key=lambda n: -apps[n])
    return {"matches": matches, "opps": opps, "snaps": snaps, "opp_skill": opp_skill, "oppQuality": oq,
            "swing": round(swing, 1), "swingSource": swing_src, "medSkill": med, "apps": apps, "regulars": regulars}


def spec(m, opp_skill):
    return {"present": {p["p"]: p["pos"] for p in m["_present"]}, "oppHumans": m.get("oppHumans", 3),
            "oppSkill": opp_skill(m), "playoff": 1 if m.get("type") == "playoff" else 0, "game": m["_game"]}


# --------------------------------------------- model ---------------------------------------------
def goal_pmf(mu, r):
    k = np.arange(MAXG + 1)
    p = _st.poisson.pmf(k, mu) if (r is None or r > 1e6) else _st.nbinom.pmf(k, r, r / (r + mu))
    return p / p.sum()


def wdl_from_goals(mf, rf, ma, ra):
    joint = np.outer(goal_pmf(mf, rf), goal_pmf(ma, ra))
    return {"W": round(float(np.tril(joint, -1).sum()), 3), "D": round(float(np.trace(joint)), 3), "L": round(float(np.triu(joint, 1).sum()), 3)}


class PoissonTeamModel:
    """Goals for and goals against each ~ NegBin(mu, r); log(mu) = intercept + player effects + context.
    Player effects share a Gaussian prior (L2 shrinkage = partial pooling towards zero)."""
    name = MODEL_NAME

    def __init__(self, regulars, alpha=DEFAULT_ALPHA):
        self.regulars, self.alpha = list(regulars), alpha

    def x(self, s):
        pos = list(s["present"].values()); names = set(s["present"])
        return [len(names) / 4, s["oppHumans"] / 4, (s["oppSkill"] - 1450) / 100, s["playoff"],
                pos.count("defender") / 2, pos.count("midfielder") / 2, pos.count("forward") / 2,
                len([n for n in names if n not in self.regulars])] + [1.0 if n in names else 0.0 for n in self.regulars]

    def feature_names(self):
        return ["humans", "oppHumans", "oppSkill", "playoff", "defenders", "midfielders", "forwards", "otherHumans"] + self.regulars

    def fit(self, specs, gf, ga):
        X = np.array([self.x(s) for s in specs], float)
        self.mf = _PoissonReg(alpha=self.alpha, max_iter=1000).fit(X, gf)
        self.ma = _PoissonReg(alpha=self.alpha, max_iter=1000).fit(X, ga)
        def disp(m, y):
            mu = m.predict(X); ex = float(np.sum((y - mu) ** 2 - mu))
            return float(np.sum(mu ** 2) / ex) if ex > 0 else None
        self.rf, self.ra = disp(self.mf, gf), disp(self.ma, ga)
        return self

    def predict(self, specs):
        if not specs: return []
        X = np.array([self.x(s) for s in specs], float)
        out = []
        for a, b in zip(self.mf.predict(X), self.ma.predict(X)):
            out.append({"gd": float(a - b), "xgf": float(a), "xga": float(b), "probs": wdl_from_goals(a, self.rf, b, self.ra)})
        return out

    def effects(self):
        return [{"f": n, "attack": round(float(a), 3), "defence": round(float(-d), 3)}
                for n, a, d in zip(self.feature_names(), self.mf.coef_, self.ma.coef_)]


def result_of(gd): return "W" if gd > 0 else "D" if gd == 0 else "L"
def logloss(p, r): return -math.log(max(p[r], 1e-3))
def brier(p, r): return sum((p[k] - (1.0 if k == r else 0.0)) ** 2 for k in "WDL")


class Evaluator:
    """Walk-forward by session. Settings for each forecast are chosen using only earlier sessions (nested)."""
    def __init__(self, ctx):
        self.ctx = ctx; M = ctx["matches"]
        self.specs = [spec(m, ctx["opp_skill"]) for m in M]
        self.GF = np.array([m["gf"] for m in M], float); self.GA = np.array([m["ga"] for m in M], float)
        self.GD = self.GF - self.GA; self.S = np.array([m["_sess"] for m in M]); self.sessions = sorted(set(self.S.tolist()))
        self.RES = [result_of(g) for g in self.GD]

    def split(self, s):
        tr = [i for i in range(len(self.S)) if self.S[i] < s]; te = [i for i in range(len(self.S)) if self.S[i] == s]
        return tr, te

    def fit(self, tr, alpha):
        return PoissonTeamModel(self.ctx["regulars"], alpha).fit([self.specs[i] for i in tr], self.GF[tr], self.GA[tr])

    def session_logloss(self, alpha, s):
        """Log losses for session s predicted from all earlier sessions (cached: the same fit serves every outer loop)."""
        if not hasattr(self, "_ll"): self._ll = {}
        k = (alpha, s)
        if k not in self._ll:
            tr, te = self.split(s)
            if len(tr) < MIN_TRAIN or not te: self._ll[k] = None
            else:
                pr = self.fit(tr, alpha).predict([self.specs[i] for i in te])
                self._ll[k] = [logloss(p["probs"], self.RES[i]) for i, p in zip(te, pr)]
        return self._ll[k]

    def wf_logloss(self, alpha, sessions):
        L, used = [], 0
        for s in sessions:
            v = self.session_logloss(alpha, s)
            if v is None: continue
            L += v; used += 1
        return (float(np.mean(L)) if L else None), used

    def pick_alpha(self, sessions):
        best, ba = None, DEFAULT_ALPHA
        for a in ALPHAS:
            v, n = self.wf_logloss(a, sessions)
            if v is not None and n >= 2 and (best is None or v < best): best, ba = v, a
        return ba

    def nested(self):
        """Per forecast game: (index, prediction, baseline) using only earlier sessions."""
        rows = []
        for s in self.sessions:
            tr, te = self.split(s)
            if len(tr) < MIN_TRAIN or not te: continue
            a = self.pick_alpha([t for t in self.sessions if t < s])
            pr = self.fit(tr, a).predict([self.specs[i] for i in te])
            freq = {k: (sum(1 for i in tr if self.RES[i] == k) + 1) / (len(tr) + 3) for k in "WDL"}
            base = {"gd": float(self.GD[tr].mean()), "gf": float(self.GF[tr].mean()), "ga": float(self.GA[tr].mean()), "probs": freq}
            for i, p in zip(te, pr): rows.append((i, p, base, s, len(tr), a))
        return rows


# --------------------------------------- player forecasts ----------------------------------------
def eb_player(vals_by_player):
    """Empirical-Bayes per-player mean: each player's average shrunk towards the squad average,
    more strongly for players with few games. Returns (estimates, squad mean, within-player variance)."""
    vals = {n: v for n, v in vals_by_player.items() if v}
    allv = [x for v in vals.values() for x in v]
    mu = float(np.mean(allv)) if allv else 0.0
    multi = [v for v in vals.values() if len(v) >= 2]
    within = float(np.mean([np.var(v) for v in multi])) if multi else 1.0
    if len(multi) >= 2:
        tau2 = max(float(np.var([np.mean(v) for v in multi])) - within / max(1.0, float(np.mean([len(v) for v in vals.values()]))), 1e-3)
    else:
        tau2 = 1.0
    est = {}
    for n, v in vals.items():
        w = len(v) / max(within, 1e-3); est[n] = (w * float(np.mean(v)) + mu / tau2) / (w + 1 / tau2)
    return est, mu, within


# =================================================================================================
# Season review: what the model said vs what happened. Writes the "review/latest" document.
# Usage: python3 -I review_model.py <db_dir> <out_json>
# <db_dir> holds matches/, opponents/, snapshots/, modelruns/ and (optional) oppskill/.
# Each session is predicted by the same Poisson model, trained only on earlier sessions with its
# settings also chosen only from earlier sessions, for the lineup that actually played.
# =================================================================================================
PLAYOFF_WINDOW = 4 * 86400   # a season's playoff run lasts at most 4 days from its first playoff game
REVIEW_STATS = ["r", "g", "a", "sh", "pm", "tm"]


def main(db, outp):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ctx = prepare(db)
    matches, opp_skill = ctx["matches"], ctx["opp_skill"]
    runs = sorted([r for r in load(os.path.join(db, "modelruns")) if r.get("trainedAt")], key=lambda r: r["trainedAt"])

    season, first_po = 1, None
    for m in matches:
        if first_po is not None and m["ts"] > first_po + PLAYOFF_WINDOW:
            season += 1; first_po = None
        if m.get("type") == "playoff" and first_po is None: first_po = m["ts"]
        m["_season"] = season
    seasons = []
    for s in sorted(set(m["_season"] for m in matches)):
        g = [m for m in matches if m["_season"] == s]; po = [m for m in g if m.get("type") == "playoff"]
        seasons.append({"n": s, "start": iso(g[0]["ts"]), "end": iso(g[-1]["ts"]), "games": len(g),
                        "playoffStart": iso(po[0]["ts"]) if po else None, "playoffWindowEnds": iso(po[0]["ts"] + PLAYOFF_WINDOW) if po else None,
                        "complete": bool(po) and (s < season or datetime.datetime.now().timestamp() > po[0]["ts"] + PLAYOFF_WINDOW)})

    # ---------- team: nested walk-forward ----------
    ev = Evaluator(ctx)
    pred = {}
    for i, p, b, s, ntr, a in ev.nested():
        bp = b["probs"]
        pred[i] = {"gd": round(p["gd"], 2), "gf": round(max(p["xgf"], 0), 2), "ga": round(max(p["xga"], 0), 2),
                   "W": p["probs"]["W"], "D": p["probs"]["D"], "L": p["probs"]["L"], "trainGames": ntr,
                   "base": {"gd": round(b["gd"], 2), "gf": round(b["gf"], 2), "ga": round(b["ga"], 2),
                            "W": round(bp["W"], 3), "D": round(bp["D"], 3), "L": round(bp["L"], 3)}}

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
            rec = list(r["bestLineup"]); hit = [n for n in rec if n in lineup]; extra = [n for n in lineup if n not in rec]
            g["sys"] = {"run": r["trainedAt"], "model": r.get("model", "xgb"), "rec": rec, "probs": r.get("bestProbs"),
                        "overlap": round(len(hit) / len(rec), 2) if rec else None, "missing": [n for n in rec if n not in lineup],
                        "extra": extra, "followed": bool(rec) and not extra and len(hit) == len(rec)}
        games.append(g)

    # ---------- players: empirical-Bayes forecasts, walk-forward by session ----------
    meta = [(m["_sess"], mi, p) for mi, m in enumerate(matches) for p in m["_present"]]
    psess = np.array([s for s, _, _ in meta])
    papps = [{"m": matches[mi]["id"], "p": p["p"], "pos": p["pos"], "act": {k: float(p.get(k) or 0) for k in REVIEW_STATS}} for _, mi, p in meta]
    for key in REVIEW_STATS:
        errs = []
        for s in ev.sessions:
            tr, te = psess < s, psess == s
            if tr.sum() < 30: continue
            own = {}
            for (s_, _, p), ok in zip(meta, tr):
                if ok: own.setdefault(p["p"], []).append(float(p.get(key) or 0))
            est, mu, within = eb_player(own)
            tol = float(np.sqrt(np.mean(np.square(errs)))) if len(errs) >= 20 else math.sqrt(within)
            for i in np.where(te)[0]:
                nm = meta[i][2]["p"]; a = papps[i]
                pv = float(est.get(nm, mu)); bv = float(np.mean(own[nm])) if nm in own else mu
                a.setdefault("pred", {})[key] = round(max(pv, 0), 2)
                a.setdefault("base", {})[key] = round(bv, 2)
                a.setdefault("tol", {})[key] = round(tol, 2)
                a["newPlayer"] = nm not in own
                errs.append(a["act"][key] - pv)
    papps = [a for a in papps if "pred" in a]

    out = {"computedAt": now, "version": 2, "model": MODEL_KEY, "modelName": MODEL_NAME,
           "seasonRule": "A season ends when its playoff run is over (playoff games within 4 days of the first one).",
           "method": "Each session is predicted by the dashboard's one model (hierarchical Poisson / Negative Binomial), trained only on games from earlier sessions with its settings also chosen from earlier sessions only, for the lineup that actually played. Player forecasts are each player's average so far, shrunk towards the squad average. Opponent strength is the opponent's estimated EA skill rating before kick-off.",
           "seasons": seasons, "games": games, "apps": papps, "nRuns": len(runs), "firstRun": runs[0]["trainedAt"] if runs else None}
    json.dump(out, open(outp, "w"), separators=(",", ":"))
    print("games", len(games), "predicted", sum(1 for g in games if "pred" in g), "with published forecast", sum(1 for g in games if "sys" in g),
          "player apps predicted", len(papps), "bytes", os.path.getsize(outp))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
