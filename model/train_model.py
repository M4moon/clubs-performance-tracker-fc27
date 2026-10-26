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
# Training: writes the "model/latest" document (and <out>_run.json for "modelruns").
# Usage: python3 -I train_model.py <db_dir> <out_json>
# <db_dir> holds ArtifactData dumps: matches/, opponents/, snapshots/ and (optional) oppskill/.
# =================================================================================================
def main(db, outp):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ctx = prepare(db)
    matches, opp_skill, regulars, apps = ctx["matches"], ctx["opp_skill"], ctx["regulars"], ctx["apps"]
    if len(matches) < 12:
        json.dump({"trainedAt": now, "status": "not enough games", "nMatches": len(matches), "key": MODEL_KEY, "name": MODEL_NAME}, open(outp, "w"))
        json.dump({"trainedAt": now, "nMatches": len(matches), "model": MODEL_KEY}, open(os.path.splitext(outp)[0] + "_run.json", "w"))
        return
    ev = Evaluator(ctx)
    last5 = matches[-5:]
    cur_form = (float(np.mean([g["gf"] > g["ga"] for g in last5])), float(np.mean([g["gf"] - g["ga"] for g in last5])))

    # ---------------- honest accuracy: nested walk-forward ----------------
    rows = ev.nested()
    GD, GF, GA, RES = ev.GD, ev.GF, ev.GA, ev.RES
    wf = []
    for s in sorted({r[3] for r in rows}):
        R = [r for r in rows if r[3] == s]; te = [r[0] for r in R]
        pgd = np.array([r[1]["gd"] for r in R]); b = R[0][2]["gd"]
        wf.append({"session": int(s), "date": iso(matches[te[0]]["ts"])[:10], "games": len(te), "trainGames": R[0][4], "alpha": R[0][5],
                   "modelMAE": round(float(np.mean(np.abs(GD[te] - pgd))), 2), "baselineMAE": round(float(np.mean(np.abs(GD[te] - b))), 2),
                   "logLoss": round(float(np.mean([logloss(r[1]["probs"], RES[r[0]]) for r in R])), 3),
                   "predGD": round(float(pgd.sum()), 1), "actualGD": int(GD[te].sum())})
    mae = lambda f: round(float(np.mean([f(r) for r in rows])), 2) if rows else None
    acc = {"modelMAE": mae(lambda r: abs(GD[r[0]] - r[1]["gd"])), "baselineMAE": mae(lambda r: abs(GD[r[0]] - r[2]["gd"])),
           "logLoss": round(float(np.mean([logloss(r[1]["probs"], RES[r[0]]) for r in rows])), 3) if rows else None,
           "baselineLogLoss": round(float(np.mean([logloss(r[2]["probs"], RES[r[0]]) for r in rows])), 3) if rows else None,
           "brier": round(float(np.mean([brier(r[1]["probs"], RES[r[0]]) for r in rows])), 3) if rows else None,
           "baselineBrier": round(float(np.mean([brier(r[2]["probs"], RES[r[0]]) for r in rows])), 3) if rows else None,
           "games": len(rows)}
    scoreline = {"modelMAE": mae(lambda r: (abs(GF[r[0]] - r[1]["xgf"]) + abs(GA[r[0]] - r[1]["xga"])) / 2),
                 "baselineMAE": mae(lambda r: (abs(GF[r[0]] - r[2]["gf"]) + abs(GA[r[0]] - r[2]["ga"])) / 2)}

    # ---------------- final model on every game; settings picked on all sessions ----------------
    alpha = ev.pick_alpha(ev.sessions)
    allidx = list(range(len(matches)))
    MODEL = ev.fit(allidx, alpha)
    predict = MODEL.predict

    # scenario: median pre-game skill of recent opponents, typical opponent humans, playoff game
    recent = matches[-15:]
    sc_skill = float(np.median([opp_skill(m) for m in recent]))
    sc_opph = int(round(np.median([m.get("oppHumans", 3) for m in recent])))
    sc = {"oppHumans": sc_opph, "oppSkill": sc_skill, "playoff": 1, "game": 2}

    # ---------------- club roster (latest snapshots plus anyone seen in a game) ----------------
    fav = {}
    for s in ctx["snaps"]:
        for mem in s.get("members", []) or []:
            if mem.get("p"): fav[mem["p"]] = mem.get("fav") or fav.get(mem["p"])
    roster = sorted(set(fav) | set(apps), key=lambda n: (-apps.get(n, 0), n.lower()))[:10]
    def norm_pos(p):
        p = (p or "").lower()
        return p if p in POS else "midfielder"
    main_pos, allowed = {}, {}
    for n in set(apps) | set(roster):
        cnt = {}
        for m in matches:
            for p in m["_present"]:
                if p["p"] == n: cnt[p["pos"]] = cnt.get(p["pos"], 0) + 1
        ranked_pos = [k for k, _ in sorted(cnt.items(), key=lambda kv: -kv[1])]
        if ranked_pos: main_pos[n] = ranked_pos[0]
        if n in roster:
            opts = ([ranked_pos[0]] + [k for k in ranked_pos[1:] if cnt[k] >= 2])[:3] if ranked_pos else []
            if not opts: opts = [norm_pos(fav.get(n))]
            allowed[n] = [o for o in opts if o != "goalkeeper"] or ["midfielder"]
            main_pos.setdefault(n, allowed[n][0])
    together = lambda c: sum(1 for m in matches if set(c) <= {p["p"] for p in m["_present"]})

    # ---------------- player forecasts (empirical Bayes), walk-forward checked ----------------
    meta = [(m["_sess"], p) for m in matches for p in m["_present"]]
    psess = np.array([s for s, _ in meta])
    def vals(key, mask):
        d = {}
        for (s_, p), ok in zip(meta, mask):
            if ok: d.setdefault(p["p"], []).append(float(p.get(key) or 0))
        return d
    players_out = {n: {"p": n, "apps": apps.get(n, 0), "pos": main_pos.get(n, "midfielder"), "pred": {}, "avg": {}, "range": {}} for n in roster}
    player_eval = {}
    for key, label in PLAYER_TARGETS:
        em, eb = [], []
        for s_ in ev.sessions:
            tr, te = psess < s_, psess == s_
            if tr.sum() < 30: continue
            est, mu, _ = eb_player(vals(key, tr))
            own = vals(key, tr)
            for i in np.where(te)[0]:
                p = meta[i][1]; y = float(p.get(key) or 0)
                em.append(abs(y - est.get(p["p"], mu))); eb.append(abs(y - (np.mean(own[p["p"]]) if p["p"] in own else mu)))
        player_eval[key] = {"label": label, "modelMAE": round(float(np.mean(em)), 2) if em else None, "baselineMAE": round(float(np.mean(eb)), 2) if eb else None}
        est, mu, within = eb_player(vals(key, np.ones(len(meta), bool)))
        sd = math.sqrt(within)
        for n in roster:
            hist = [float(p.get(key) or 0) for _, p in meta if p["p"] == n]
            v = float(est.get(n, mu))
            players_out[n]["pred"][key] = round(max(v, 0), 2)
            players_out[n]["avg"][key] = round(float(np.mean(hist)), 2) if hist else None
            players_out[n]["range"][key] = [round(max(v - sd, 0), 2), round(v + sd, 2)]
    for n in roster:
        recent_r = [p["r"] for m in matches[-12:] for p in m["_present"] if p["p"] == n]
        players_out[n]["recentRating"] = round(float(np.mean(recent_r)), 2) if recent_r else None
        players_out[n]["recentApps"] = len(recent_r)
    # each player's expected rating at each allowed position: own average there, shrunk towards their overall level
    r_est, r_mu, _ = eb_player(vals("r", np.ones(len(meta), bool)))
    pos_rating, with_stats = {}, {}
    for n in roster:
        base = float(r_est.get(n, r_mu)); byp = {}
        for _, p in meta:
            if p["p"] == n: byp.setdefault(p["pos"], []).append(float(p["r"]))
        pos_rating[n] = {pos: round((sum(byp.get(pos, [])) + 3 * base) / (len(byp.get(pos, [])) + 3), 2) for pos in allowed[n]}
        gms = [m for m in matches if any(p["p"] == n for p in m["_present"])]
        with_stats[n] = {"games": len(gms), "ppg": round(sum(3 if m["gf"] > m["ga"] else 1 if m["gf"] == m["ga"] else 0 for m in gms) / len(gms), 2) if gms else None,
                         "gd": round(sum(m["gf"] - m["ga"] for m in gms) / len(gms), 2) if gms else None,
                         "ratingByPos": {k: [round(float(np.mean(v)), 2), len(v)] for k, v in byp.items()}}

    # ---------------- lineups that have actually played (exact sets of humans) ----------------
    groups = {}
    for m in matches:
        key = tuple(sorted(p["p"] for p in m["_present"]))
        if key: groups.setdefault(key, []).append(m)
    actual = []
    for key, gms in groups.items():
        pos = {}
        for n in key:
            cnt = {}
            for m in gms:
                for p in m["_present"]:
                    if p["p"] == n: cnt[p["pos"]] = cnt.get(p["pos"], 0) + 1
            pos[n] = max(cnt, key=cnt.get)
        pr = predict([{**sc, "present": pos}])[0]
        W = sum(1 for m in gms if m["gf"] > m["ga"]); D = sum(1 for m in gms if m["gf"] == m["ga"]); n = len(gms)
        actual.append({"players": [{"p": p, "pos": pos[p]} for p in key], "size": len(key), "predGD": round(pr["gd"], 2),
                       "probs": pr["probs"], "xPts": round(3 * pr["probs"]["W"] + pr["probs"]["D"], 2),
                       "xGF": round(pr["xgf"], 1), "xGA": round(pr["xga"], 1), "gamesTogether": n,
                       "record": [W, D, n - W - D], "gf": sum(m["gf"] for m in gms), "ga": sum(m["ga"] for m in gms),
                       "actualPpg": round((3 * W + D) / n, 2), "lastTs": max(m["ts"] for m in gms)})
    actual.sort(key=lambda r: (-(r["gamesTogether"] >= 2), -r["xPts"], -r["gamesTogether"]))
    by_size = {}
    for rr in sorted(actual, key=lambda r: -r["xPts"]):
        by_size.setdefault(str(rr["size"]), [])
        if len(by_size[str(rr["size"])]) < 5: by_size[str(rr["size"])].append(rr)
    proven = [x for x in actual if x["gamesTogether"] >= 2] or actual
    best = max(proven, key=lambda x: (x["xPts"], x["gamesTogether"]))

    # ---------------- lineup builder: every subset of the roster, best position assignment ----------------
    specs_c, meta_c = [], []
    R = len(roster)
    for mask in range(1, 1 << R):
        members = [roster[i] for i in range(R) if mask >> i & 1]
        for assign in itertools.product(*[allowed[n] for n in members]):
            specs_c.append({**sc, "present": dict(zip(members, assign))}); meta_c.append((mask, members, assign))
    best_for = {}
    for (mask, members, assign), pr in zip(meta_c, predict(specs_c)):
        score = 3 * pr["probs"]["W"] + pr["probs"]["D"] + 0.01 * sum(pos_rating[n][a] for n, a in zip(members, assign))
        if mask not in best_for or score > best_for[mask][0]: best_for[mask] = (score, members, assign, pr)
    code = {"defender": "D", "midfielder": "M", "forward": "F"}
    combos = {str(mask): [round(pr["gd"], 2), pr["probs"]["W"], pr["probs"]["D"], pr["probs"]["L"], together(members),
                          "".join(code[a] for a in assign), round(pr["xgf"], 1), round(pr["xga"], 1)]
              for mask, (score, members, assign, pr) in best_for.items()}
    usual = predict([{**sc, "present": {p["p"]: p["pos"] for p in m["_present"]}} for m in matches[-10:]])
    u_gd = float(np.mean([u["gd"] for u in usual]))
    u_probs = {k: round(float(np.mean([u["probs"][k] for u in usual])), 3) for k in "WDL"}

    eff = MODEL.effects()
    tot = sum(abs(e["attack"]) + abs(e["defence"]) for e in eff) or 1
    oq = ctx["oppQuality"]
    out = {
        "trainedAt": now, "status": "ok", "version": 2, "key": MODEL_KEY, "name": MODEL_NAME,
        "nMatches": len(matches), "nApps": len(meta), "nSessions": int(max(m["_sess"] for m in matches)),
        "lastMatchTs": matches[-1]["ts"],
        "scenario": {"oppSkill": round(sc_skill), "oppHumans": sc_opph, "type": "playoff"},
        "match": {"target": "goals for and against", **acc, "walkForward": wf,
                  "importance": sorted([{"f": e["f"], "share": round((abs(e["attack"]) + abs(e["defence"])) / tot, 3)} for e in eff], key=lambda x: -x["share"])[:10]},
        "scoreline": scoreline, "record": {"bestLineup": {"predGD": best["predGD"], "probs": best["probs"]}, "usualSquad": {"predGD": round(u_gd, 2), "probs": u_probs}},
        "form": {"winRate5": round(cur_form[0], 2), "gd5": round(cur_form[1], 2)},
        "bestLineup": best, "lineups": by_size,
        "lineupTable": {"roster": [{"p": n, "apps": apps.get(n, 0), "fav": fav.get(n), "main": main_pos.get(n), "allowed": allowed[n],
                                    "posRating": pos_rating.get(n, {}), "regular": n in regulars, "with": with_stats[n]} for n in roster],
                        "actual": actual, "combos": combos, "fields": ["predGD", "W", "D", "L", "gamesTogether", "positions", "xGF", "xGA"]},
        "beatsBaseline": {"match": bool(acc["logLoss"] is not None and acc["logLoss"] < acc["baselineLogLoss"] and acc["modelMAE"] < acc["baselineMAE"]),
                          **{k: bool(v["modelMAE"] is not None and v["modelMAE"] < v["baselineMAE"]) for k, v in player_eval.items()}},
        "players": sorted(players_out.values(), key=lambda r: (-r["apps"] if r["apps"] < MIN_APPS else 0, -r["pred"].get("r", 0))),
        "playerEval": player_eval,
        "params": {"alpha": alpha, "dispersionFor": round(MODEL.rf, 2) if MODEL.rf else None, "dispersionAgainst": round(MODEL.ra, 2) if MODEL.ra else None},
        "effects": eff,
        "about": "Goals for and goals against are each modelled as counts (Negative Binomial when the scores are more spread out than Poisson allows). Every regular has an attack and a defence effect, pulled towards zero unless the games support it (partial pooling). Win, draw and loss chances come from the two goal distributions. Player forecasts shrink each player's average towards the squad average, more strongly for players with few games.",
        "oppSkillMethod": {"swingPerResult": ctx["swing"], "swingSource": ctx["swingSource"], **oq},
        "selection": SELECTION,
        "notes": ["Opponent strength is each opponent's EA skill rating before kick-off, estimated as the rating saved after the game minus the %s points that game's result moved it." % ctx["swing"]
                  + (" %d older games use a rating saved more than a day later, so theirs is less exact." % oq["lateOver24h"] if oq["lateOver24h"] else ""),
                  "Accuracy figures come from predicting each session using only earlier sessions, with the model's settings also chosen only from earlier sessions.",
                  "Players need %d games to get their own effect; others count as 'occasional players'." % MIN_APPS,
                  "Formation and pitch slots are suggested from where each player performs best; EA's match feed doesn't record formations.",
                  "Only lineups that have actually played together at least twice are recommended, in the positions they actually used."],
    }
    json.dump(out, open(outp, "w"))
    run = {"trainedAt": now, "model": MODEL_KEY, "nMatches": out["nMatches"], "nApps": out["nApps"],
           "matchMAE": acc["modelMAE"], "baselineMAE": acc["baselineMAE"], "logLoss": acc["logLoss"], "baselineLogLoss": acc["baselineLogLoss"],
           "ratingMAE": player_eval["r"]["modelMAE"], "ratingBaselineMAE": player_eval["r"]["baselineMAE"],
           "bestLineup": [p["p"] for p in best["players"]], "bestProbs": best["probs"]}
    json.dump(run, open(os.path.splitext(outp)[0] + "_run.json", "w"))
    print(json.dumps({"nMatches": out["nMatches"], "alpha": alpha, **{k: acc[k] for k in ("modelMAE", "baselineMAE", "logLoss", "baselineLogLoss")}, "swing": ctx["swing"], "opp": oq}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
