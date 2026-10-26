#!/usr/bin/env python3
"""Al Haramis FC Lab - XGBoost model.

Usage: python3 -I train_model.py <db_dir> <out_json>
<db_dir> holds ArtifactData dumps: <db_dir>/matches/*.json and <db_dir>/opponents/*.json
(each file is either the raw document or {"id":..,"data":{..}}).
Writes one JSON document meant for ArtifactData collection "model", doc "latest".
"""
import json, glob, os, sys, math, itertools, datetime
import numpy as np
import xgboost as xgb

SESSION_GAP = 2700          # seconds between games that starts a new session
MIN_APPS = 3                # games needed before a player gets their own feature
SEEDS = [0, 1, 2, 3, 4]     # averaged to keep tiny-data trees stable
PARAMS = dict(n_estimators=150, learning_rate=0.05, max_depth=2, min_child_weight=3,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, objective="reg:squarederror")
PLAYER_TARGETS = [("r", "Rating"), ("g", "Goals"), ("a", "Assists"), ("sh", "Shots"),
                  ("pm", "Passes made"), ("pa", "Passes tried"), ("tm", "Tackles won")]
POS = {"goalkeeper": 0, "defender": 1, "midfielder": 2, "forward": 3}


def load(dirp):
    out = []
    for f in glob.glob(os.path.join(dirp, "*.json")):
        d = json.load(open(f))
        out.append(d.get("data", d) if isinstance(d, dict) and "data" in d and isinstance(d["data"], dict) else d)
    return out


def fit_predict(X, y, Xp):
    preds = []
    for s in SEEDS:
        m = xgb.XGBRegressor(random_state=s, **PARAMS)
        m.fit(X, y)
        preds.append(m.predict(Xp))
    return np.mean(preds, axis=0)


def importance(X, y, names):
    imp = np.zeros(len(names))
    for s in SEEDS:
        m = xgb.XGBRegressor(random_state=s, **PARAMS)
        m.fit(X, y)
        g = m.get_booster().get_score(importance_type="total_gain")
        for k, v in g.items():
            imp[int(k[1:])] += v
    tot = imp.sum() or 1
    return sorted([{"f": names[i], "share": round(float(imp[i] / tot), 3)} for i in range(len(names)) if imp[i] > 0],
                  key=lambda r: -r["share"])


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def wdl(mu, sigma):
    sigma = max(sigma, 0.8)
    pw = 1 - norm_cdf((0.5 - mu) / sigma)
    pl = norm_cdf((-0.5 - mu) / sigma)
    return {"W": round(pw, 3), "D": round(max(0.0, 1 - pw - pl), 3), "L": round(pl, 3)}


# =====================================================================================
# Alternative models (shown via the model dropdown on the Predictions tab)
#   poisson: hierarchical Poisson / Negative Binomial goals model with partial pooling
#   kalman : Bradley-Terry style strength ratings tracked over time with a Kalman filter
# Both produce the same outputs as the XGBoost model so the dashboard can swap between them.
# =====================================================================================
from scipy import stats as _st
from sklearn.linear_model import PoissonRegressor as _PoissonReg

MAXG = 15


def _spec_from_match(m, opp_skill):
    return {"present": {p["p"]: p["pos"] for p in m["_present"]}, "oppHumans": m.get("oppHumans", 3),
            "oppSkill": opp_skill(m), "playoff": 1 if m.get("type") == "playoff" else 0, "game": m["_game"]}


def _goal_pmf(mu, r):
    k = np.arange(MAXG + 1)
    if r is None or r > 1e6:
        p = _st.poisson.pmf(k, mu)
    else:
        p = _st.nbinom.pmf(k, r, r / (r + mu))
    return p / p.sum()


def _wdl_from_goals(mf, rf, ma, ra):
    pf, pa = _goal_pmf(mf, rf), _goal_pmf(ma, ra)
    joint = np.outer(pf, pa)
    W = float(np.tril(joint, -1).sum()); D = float(np.trace(joint)); L = float(np.triu(joint, 1).sum())
    return {"W": round(W, 3), "D": round(D, 3), "L": round(L, 3)}


def _wdl_gauss(mu, var):
    sd = math.sqrt(max(var, 0.5))
    pw = 1 - norm_cdf((0.5 - mu) / sd); pl = norm_cdf((-0.5 - mu) / sd)
    return {"W": round(pw, 3), "D": round(max(0.0, 1 - pw - pl), 3), "L": round(pl, 3)}


class PoissonTeamModel:
    """GF and GA each ~ NegBin(mu, r); log(mu) = intercept + player effects + context effects.
    Player effects get a shared Gaussian prior (L2 shrinkage), i.e. partial pooling towards zero;
    the shrinkage strength is chosen by walk-forward validation."""
    name = "Hierarchical Poisson / Negative Binomial"

    def __init__(self, regulars, alpha=1.0):
        self.regulars, self.alpha = regulars, alpha

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
        X = np.array([self.x(s) for s in specs], float)
        mf, ma = self.mf.predict(X), self.ma.predict(X)
        out = []
        for a, b in zip(mf, ma):
            out.append({"gd": float(a - b), "xgf": float(a), "xga": float(b), "probs": _wdl_from_goals(a, self.rf, b, self.ra)})
        return out

    def effects(self):
        nm = self.feature_names()
        return [{"f": n, "attack": round(float(a), 3), "defence": round(float(-d), 3)} for n, a, d in zip(nm, self.mf.coef_, self.ma.coef_)]


class KalmanBTModel:
    """Bradley-Terry style ratings: our side's strength = sum of the strengths of the players on the pitch
    (+ a squad-size term); the opponent's strength comes from their EA skill rating. Goal difference
    ~ Normal(ours - theirs, r). Player strengths drift between sessions as a random walk (Kalman filter),
    so recent games count more. q (drift), p0 (prior spread) and r (game noise) are picked by walk-forward."""
    name = "Bradley-Terry / Kalman filter"

    def __init__(self, names, q=0.05, p0=1.0, r=5.0, s0=4.0):
        self.names, self.q, self.p0, self.r, self.s0 = list(names), q, p0, r, s0
        self.k = 3 + len(self.names)   # [home-free intercept, opp skill, humans difference] + players

    def h(self, s):
        names = set(s["present"])
        return np.array([1.0, (s["oppSkill"] - 1450) / 100, (len(names) - s["oppHumans"]) / 2] +
                        [1.0 if n in names else 0.0 for n in self.names])

    def fit(self, specs, ys, sess):
        x = np.zeros(self.k); P = np.eye(self.k) * self.p0; P[0, 0] = P[1, 1] = P[2, 2] = self.s0
        last = None
        for s, y, ss in zip(specs, ys, sess):
            if last is not None and ss != last:   # strengths drift between sessions
                for i in range(3, self.k): P[i, i] += self.q
            last = ss
            H = self.h(s); S = float(H @ P @ H + self.r); K = P @ H / S
            x = x + K * (y - float(H @ x)); P = P - np.outer(K, H @ P)
        self.x, self.P = x, P
        return self

    def predict_gd(self, specs):
        out = []
        for s in specs:
            H = self.h(s); P = self.P.copy()
            for i in range(3, self.k): P[i, i] += self.q
            out.append((float(H @ self.x), float(H @ P @ H + self.r)))
        return out

    def ratings(self):
        return [{"p": n, "strength": round(float(self.x[3 + i]), 2), "sd": round(float(math.sqrt(self.P[3 + i, 3 + i])), 2)}
                for i, n in enumerate(self.names)]


def alt_models(ctx):
    matches, opp_skill, regulars, roster = ctx["matches"], ctx["opp_skill"], ctx["regulars"], ctx["roster"]
    sc = {"oppHumans": ctx["sc_opph"], "oppSkill": ctx["sc_skill"], "playoff": 1, "game": 2}
    specs = [_spec_from_match(m, opp_skill) for m in matches]
    GF = np.array([m["gf"] for m in matches], float); GA = np.array([m["ga"] for m in matches], float); GD = GF - GA
    sess = [m["_sess"] for m in matches]
    sessions = sorted(set(sess))
    all_names = sorted({p["p"] for m in matches for p in m["_present"]} | set(roster))

    # ---- builders for each model, with hyper-parameters picked by walk-forward ----
    def wf_eval(make, predict):
        errs, base, sl, slb, rows = [], [], [], [], []
        for s in sessions:
            tr = [i for i in range(len(matches)) if sess[i] < s]; te = [i for i in range(len(matches)) if sess[i] == s]
            if len(tr) < 10: continue
            mdl = make(tr)
            pr = predict(mdl, [specs[i] for i in te])
            gd = np.array([p["gd"] for p in pr]); b = float(np.mean(GD[tr]))
            errs += list(np.abs(GD[te] - gd)); base += list(np.abs(GD[te] - b))
            if pr and "xgf" in pr[0]:
                sl += list(np.abs(GF[te] - np.array([p["xgf"] for p in pr]))) + list(np.abs(GA[te] - np.array([p["xga"] for p in pr])))
                slb += list(np.abs(GF[te] - GF[tr].mean())) + list(np.abs(GA[te] - GA[tr].mean()))
            rows.append({"session": s, "date": datetime.datetime.fromtimestamp(matches[te[0]]["ts"], datetime.timezone.utc).strftime("%Y-%m-%d"),
                         "games": len(te), "trainGames": len(tr), "modelMAE": round(float(np.mean(np.abs(GD[te] - gd))), 2),
                         "baselineMAE": round(float(np.mean(np.abs(GD[te] - b))), 2), "predGD": round(float(gd.sum()), 1), "actualGD": int(GD[te].sum())})
        return (float(np.mean([w["modelMAE"] for w in rows])) if rows else 9e9), (float(np.mean([w["baselineMAE"] for w in rows])) if rows else None), rows, \
               ({"modelMAE": round(float(np.mean(sl)), 2), "baselineMAE": round(float(np.mean(slb)), 2)} if sl else None)

    # Poisson / NB
    def p_make(alpha):
        return lambda tr: PoissonTeamModel(regulars, alpha).fit([specs[i] for i in tr], GF[tr], GA[tr])
    p_pred = lambda mdl, ss: mdl.predict(ss)
    best_p = min(([a] + list(wf_eval(p_make(a), p_pred)) for a in [0.1, 0.3, 1.0, 3.0]), key=lambda t: t[1])
    # Kalman / Bradley-Terry: GD filter, plus GF and GA filters for the scoreline
    def k_make(q, p0, r, s0, target=GD):
        return lambda tr: KalmanBTModel(all_names, q, p0, r, s0).fit([specs[i] for i in tr], target[tr], [sess[i] for i in tr])
    def k_pred(mdl, ss):
        return [{"gd": mu, "probs": _wdl_gauss(mu, v)} for mu, v in mdl.predict_gd(ss)]
    grid = [(q, p0, r, s0) for q in [0.0, 0.02, 0.1] for p0 in [0.1, 0.3, 1.0] for r in [4.0, 8.0, 15.0] for s0 in [0.5, 2.0]]
    best_k = min(([g] + list(wf_eval(k_make(*g), k_pred)) for g in grid), key=lambda t: t[1])
    q, p0, r, s0 = best_k[0]
    # scoreline walk-forward for Kalman (separate GF and GA filters with the chosen settings)
    ksl, kslb = [], []
    for s in sessions:
        tr = [i for i in range(len(matches)) if sess[i] < s]; te = [i for i in range(len(matches)) if sess[i] == s]
        if len(tr) < 10: continue
        mf = k_make(q, p0, r, s0, GF)(tr); ma = k_make(q, p0, r, s0, GA)(tr)
        pf = [mu for mu, _ in mf.predict_gd([specs[i] for i in te])]; pa = [mu for mu, _ in ma.predict_gd([specs[i] for i in te])]
        ksl += list(np.abs(GF[te] - pf)) + list(np.abs(GA[te] - pa)); kslb += list(np.abs(GF[te] - GF[tr].mean())) + list(np.abs(GA[te] - GA[tr].mean()))
    best_k[4] = {"modelMAE": round(float(np.mean(ksl)), 2), "baselineMAE": round(float(np.mean(kslb)), 2)} if ksl else None

    allidx = list(range(len(matches)))
    P_full = p_make(best_p[0])(allidx)
    K_full = k_make(q, p0, r, s0)(allidx); KF = k_make(q, p0, r, s0, GF)(allidx); KA = k_make(q, p0, r, s0, GA)(allidx)

    def p_predict(ss): return P_full.predict(ss)
    def k_predict(ss):
        g = K_full.predict_gd(ss); f = KF.predict_gd(ss); a = KA.predict_gd(ss)
        return [{"gd": mu, "probs": _wdl_gauss(mu, v), "xgf": max(ff[0], 0), "xga": max(aa[0], 0)} for (mu, v), ff, aa in zip(g, f, a)]

    # ---- team outputs (same shape as the XGBoost section) ----
    def team_outputs(predict, wf, base_mae, wf_rows, scoreline):
        groups = {}
        for m in matches:
            key = tuple(sorted(m["_present"] and [p["p"] for p in m["_present"]] or []))
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
                           "xGF": round(pr.get("xgf", 0), 1), "xGA": round(pr.get("xga", 0), 1), "gamesTogether": n,
                           "record": [W, D, n - W - D], "gf": sum(m["gf"] for m in gms), "ga": sum(m["ga"] for m in gms),
                           "actualPpg": round((3 * W + D) / n, 2), "lastTs": max(m["ts"] for m in gms)})
        actual.sort(key=lambda r: (-(r["gamesTogether"] >= 2), -r["xPts"], -r["gamesTogether"]))
        by_size = {}
        for rr in sorted(actual, key=lambda r: -r["xPts"]):
            by_size.setdefault(str(rr["size"]), [])
            if len(by_size[str(rr["size"])]) < 5: by_size[str(rr["size"])].append(rr)
        proven = [x for x in actual if x["gamesTogether"] >= 2] or actual
        best = max(proven, key=lambda x: (x["xPts"], x["gamesTogether"]))
        # builder: every subset x allowed positions
        specs_c, meta_c = [], []
        R = len(roster)
        for mask in range(1, 1 << R):
            members = [roster[i] for i in range(R) if mask >> i & 1]
            for assign in itertools.product(*[ctx["allowed"][n] for n in members]):
                specs_c.append({**sc, "present": dict(zip(members, assign))}); meta_c.append((mask, members, assign))
        preds = predict(specs_c)
        bestfor = {}
        for (mask, members, assign), pr in zip(meta_c, preds):
            xp = 3 * pr["probs"]["W"] + pr["probs"]["D"]
            score = xp + 0.01 * sum(ctx["pos_rating"][n][a] for n, a in zip(members, assign))
            if mask not in bestfor or score > bestfor[mask][0]: bestfor[mask] = (score, members, assign, pr)
        code = {"defender": "D", "midfielder": "M", "forward": "F"}
        combos = {str(mask): [round(pr["gd"], 2), pr["probs"]["W"], pr["probs"]["D"], pr["probs"]["L"], ctx["together"](members),
                              "".join(code[a] for a in assign), round(pr.get("xgf", 0), 1), round(pr.get("xga", 0), 1)]
                  for mask, (score, members, assign, pr) in bestfor.items()}
        last10 = [m for m in matches[-10:]]
        usual = predict([{**sc, "present": {p["p"]: p["pos"] for p in m["_present"]}} for m in last10])
        u_gd = float(np.mean([u["gd"] for u in usual]))
        u_probs = {k: round(float(np.mean([u["probs"][k] for u in usual])), 3) for k in "WDL"}
        return {"match": {"modelMAE": round(wf, 2), "baselineMAE": round(base_mae, 2) if base_mae else None, "walkForward": wf_rows},
                "scoreline": scoreline, "bestLineup": best, "lineups": by_size,
                "record": {"bestLineup": {"predGD": best["predGD"], "probs": best["probs"]}, "usualSquad": {"predGD": round(u_gd, 2), "probs": u_probs}},
                "lineupTable": {"actual": actual, "combos": combos}}

    # ---- player forecasts ----
    meta = ctx["meta"]; psess = np.array([s_ for s_, _ in meta])
    def eb_player(key, tr_mask):
        """Hierarchical (empirical Bayes) per-player mean: Gamma-Poisson for counts, Normal-Normal for rating."""
        vals = {}
        for (s_, p), ok in zip(meta, tr_mask):
            if ok: vals.setdefault(p["p"], []).append(float(p.get(key) or 0))
        allv = [v for vs in vals.values() for v in vs]
        mu = float(np.mean(allv)) if allv else 0.0
        means = [np.mean(v) for v in vals.values() if len(v) >= 2]
        within = float(np.mean([np.var(v) for v in vals.values() if len(v) >= 2])) if means else 1.0
        tau2 = max(float(np.var(means)) - within / max(1, np.mean([len(v) for v in vals.values()])), 1e-3) if len(means) >= 2 else 1.0
        est = {}
        for n, v in vals.items():
            w = len(v) / max(within, 1e-3); est[n] = (w * np.mean(v) + mu / tau2) / (w + 1 / tau2)
        return est, mu, within
    def kf_player(key, tr_mask, qf):
        seq = {}
        for (s_, p), ok in zip(meta, tr_mask):
            if ok: seq.setdefault(p["p"], []).append(float(p.get(key) or 0))
        allv = [v for vs in seq.values() for v in vs]; mu = float(np.mean(allv)) if allv else 0.0
        within = float(np.mean([np.var(v) for v in seq.values() if len(v) >= 2])) if any(len(v) >= 2 for v in seq.values()) else 1.0
        between = float(np.var([np.mean(v) for v in seq.values()])) if len(seq) >= 2 else within
        est = {}
        for n, v in seq.items():
            x, P = mu, max(between, 1e-3)
            for y in v:
                P += qf * within; Kg = P / (P + within); x += Kg * (y - x); P *= (1 - Kg)
            est[n] = x
        return est, mu, within
    def player_block(fitfn):
        ev, players = {}, {n: {"p": n, "apps": ctx["apps"].get(n, 0), "pos": ctx["main_pos"][n], "pred": {}, "avg": {}, "range": {}} for n in roster}
        for key, label in ctx["targets"]:
            em, eb = [], []
            for s_ in sessions:
                tr, te = psess < s_, psess == s_
                if tr.sum() < 30: continue
                est, mu, _ = fitfn(key, tr)
                for i in np.where(te)[0]:
                    p = meta[i][1]; y = float(p.get(key) or 0)
                    hist = [float(meta[j][1].get(key) or 0) for j in np.where(tr)[0] if meta[j][1]["p"] == p["p"]]
                    em.append(abs(y - est.get(p["p"], mu))); eb.append(abs(y - (np.mean(hist) if hist else mu)))
            ev[key] = {"label": label, "modelMAE": round(float(np.mean(em)), 2) if em else None, "baselineMAE": round(float(np.mean(eb)), 2) if eb else None}
            est, mu, within = fitfn(key, np.ones(len(meta), bool))
            sd = math.sqrt(within)
            for n in roster:
                hist = [float(p.get(key) or 0) for _, p in meta if p["p"] == n]
                v = float(est.get(n, mu))
                players[n]["pred"][key] = round(max(v, 0), 2); players[n]["avg"][key] = round(float(np.mean(hist)), 2) if hist else None
                players[n]["range"][key] = [round(max(v - sd, 0), 2), round(v + sd, 2)]
        return sorted(players.values(), key=lambda r: (-r["apps"] if r["apps"] < 3 else 0, -r["pred"].get("r", 0))), ev
    pl_p, ev_p = player_block(eb_player)
    qbest = 0.1
    pl_k, ev_k = player_block(lambda k, m: kf_player(k, m, qbest))

    def beats(team, ev):
        return {"match": bool(team["match"]["baselineMAE"] and team["match"]["modelMAE"] < team["match"]["baselineMAE"]),
                **{k: bool(v["modelMAE"] is not None and v["modelMAE"] < v["baselineMAE"]) for k, v in ev.items()}}

    tp = team_outputs(p_predict, best_p[1], best_p[2], best_p[3], best_p[4])
    eff = P_full.effects()
    tot = sum(abs(e["attack"]) + abs(e["defence"]) for e in eff) or 1
    tp.update({"key": "poisson", "name": PoissonTeamModel.name, "players": pl_p, "playerEval": ev_p,
               "params": {"alpha": best_p[0], "dispersionFor": round(P_full.rf, 2) if P_full.rf else None, "dispersionAgainst": round(P_full.ra, 2) if P_full.ra else None},
               "effects": eff, "importance": sorted([{"f": e["f"], "share": round((abs(e["attack"]) + abs(e["defence"])) / tot, 3)} for e in eff], key=lambda x: -x["share"])[:10],
               "about": "Goals for and goals against are each modelled as counts (Negative Binomial when the scores are more spread out than Poisson allows). Every player has an attack and a defence effect, pulled towards zero unless the games support it (partial pooling). Win, draw and loss chances come from the two goal distributions. Player forecasts shrink each player's average towards the squad average, more strongly for players with few games."})
    tp["beatsBaseline"] = beats(tp, ev_p)
    tk = team_outputs(k_predict, best_k[1], best_k[2], best_k[3], best_k[4])
    rat = K_full.ratings()
    totk = sum(abs(x["strength"]) for x in rat) or 1
    tk.update({"key": "kalman", "name": KalmanBTModel.name, "players": pl_k, "playerEval": ev_k,
               "params": {"q": q, "p0": p0, "r": r, "s0": s0, "oppSkillPer100": round(float(K_full.x[1]), 2), "humansEdge": round(float(K_full.x[2]), 2), "base": round(float(K_full.x[0]), 2)},
               "ratings": sorted(rat, key=lambda x: -x["strength"]),
               "importance": sorted([{"f": x["p"], "share": round(abs(x["strength"]) / totk, 3)} for x in rat], key=lambda x: -x["share"])[:10],
               "about": "Each player has a strength rating that is updated after every game they play, Kalman-filter style: big surprises move it more, and ratings can drift between sessions so recent form counts more. Our side's strength is the sum of the players on the pitch; the opponent's comes from their EA skill rating. The gap gives the expected goal difference and the win, draw and loss chances (Bradley-Terry style). Player forecasts follow each player's own rating level the same way."})
    tk["beatsBaseline"] = beats(tk, ev_k)
    return {"poisson": tp, "kalman": tk}


def main(db, outp):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    matches = sorted(load(os.path.join(db, "matches")), key=lambda m: m["ts"])
    opps = {int(o["id"]): o for o in load(os.path.join(db, "opponents"))}
    if len(matches) < 12:
        json.dump({"trainedAt": now, "status": "not enough games", "nMatches": len(matches)}, open(outp, "w"))
        return

    # sessions and game number within a session
    sess, prev = 0, None
    for m in matches:
        if prev is None or m["ts"] - prev > SESSION_GAP:
            sess += 1; gi = 0
        gi += 1; m["_sess"], m["_game"], prev = sess, gi, m["ts"]
        m["_present"] = [p for p in m.get("players", []) if (p.get("sec") or 0) > 0]
    # recent form from the previous 5 games only (never the game being predicted); NaN for the first game
    for i, m in enumerate(matches):
        prev5 = matches[max(0, i - 5):i]
        m["_form"] = (float(np.mean([1.0 if g["gf"] > g["ga"] else 0.0 for g in prev5])) if prev5 else float("nan"),
                      float(np.mean([g["gf"] - g["ga"] for g in prev5])) if prev5 else float("nan"))
    last5 = matches[-5:]
    cur_form = (float(np.mean([1.0 if g["gf"] > g["ga"] else 0.0 for g in last5])), float(np.mean([g["gf"] - g["ga"] for g in last5])))

    skills = [o["skillRating"] for o in opps.values() if o.get("skillRating")]
    med_skill = float(np.median(skills)) if skills else 1450.0
    def opp_skill(m):
        o = opps.get(int(m.get("oppId", 0)))
        return float(o["skillRating"]) if o and o.get("skillRating") else med_skill

    apps = {}
    for m in matches:
        for p in m["_present"]:
            apps[p["p"]] = apps.get(p["p"], 0) + 1
    regulars = sorted([n for n, c in apps.items() if c >= MIN_APPS], key=lambda n: -apps[n])
    main_pos = {}
    for n in apps:
        cnt = {}
        for m in matches:
            for p in m["_present"]:
                if p["p"] == n: cnt[p["pos"]] = cnt.get(p["pos"], 0) + 1
        main_pos[n] = max(cnt, key=cnt.get)

    # ---------------- match model: goal difference ----------------
    mf = ["humans", "oppHumans", "oppSkill", "playoff", "sessionGame", "otherHumans", "defenders", "midfielders", "forwards"] + ["with " + n for n in regulars] + ["formWinRate5", "formGD5"]
    def mrow(present, opp_h, skill, playoff, game, form=None):
        form = form or cur_form
        # present: {player name: position}
        names = set(present)
        pos = list(present.values())
        return [len(names), opp_h, skill, playoff, min(game, 6), len([n for n in names if n not in regulars]),
                pos.count("defender") + pos.count("goalkeeper"), pos.count("midfielder"), pos.count("forward")] + \
               [1 if n in names else 0 for n in regulars] + [form[0], form[1]]
    X = np.array([mrow({p["p"]: p["pos"] for p in m["_present"]}, m.get("oppHumans", 3), opp_skill(m),
                       1 if m.get("type") == "playoff" else 0, m["_game"], m["_form"]) for m in matches], float)
    YF = np.array([m["gf"] for m in matches], float); YA = np.array([m["ga"] for m in matches], float)
    y = np.array([m["gf"] - m["ga"] for m in matches], float)

    # walk-forward: predict each session using only earlier sessions
    wf, res, sl_err_m, sl_err_b = [], [], [], []
    for s in sorted(set(m["_sess"] for m in matches)):
        tr = [i for i, m in enumerate(matches) if m["_sess"] < s]
        te = [i for i, m in enumerate(matches) if m["_sess"] == s]
        if len(tr) < 10: continue
        p = fit_predict(X[tr], y[tr], X[te]); base = float(np.mean(y[tr]))
        pf_ = fit_predict(X[tr], YF[tr], X[te]); pa_ = fit_predict(X[tr], YA[tr], X[te])
        sl_err_m.extend(list(np.abs(YF[te] - pf_)) + list(np.abs(YA[te] - pa_)))
        sl_err_b.extend(list(np.abs(YF[te] - YF[tr].mean())) + list(np.abs(YA[te] - YA[tr].mean())))
        for k, i in enumerate(te): res.append(float(y[i] - p[k]))
        wf.append({"session": s, "date": datetime.datetime.fromtimestamp(matches[te[0]]["ts"], datetime.timezone.utc).strftime("%Y-%m-%d"),
                   "games": len(te), "trainGames": len(tr),
                   "modelMAE": round(float(np.mean(np.abs(y[te] - p))), 2),
                   "baselineMAE": round(float(np.mean(np.abs(y[te] - base))), 2),
                   "predGD": round(float(np.sum(p)), 1), "actualGD": int(np.sum(y[te]))})
    sigma = float(np.sqrt(np.mean(np.square(res)))) if res else float(np.std(y))
    m_mae = float(np.mean([w["modelMAE"] for w in wf])) if wf else None
    b_mae = float(np.mean([w["baselineMAE"] for w in wf])) if wf else None

    # scenario for the weekend: median recent opponent strength, typical opponent humans
    recent = matches[-15:]
    sc_skill = float(np.median([opp_skill(m) for m in recent]))
    OX = np.array([[m["_form"][0], m["_form"][1], 1 if m.get("type") == "playoff" else 0] for m in matches], float)
    OY = np.array([opp_skill(m) for m in matches], float)
    o_err_m, o_err_b = [], []
    for s_ in sorted(set(m["_sess"] for m in matches)):
        tr = [i for i, m in enumerate(matches) if m["_sess"] < s_]; te = [i for i, m in enumerate(matches) if m["_sess"] == s_]
        if len(tr) < 10: continue
        pr = fit_predict(OX[tr], OY[tr], OX[te])
        o_err_m += list(np.abs(OY[te] - pr)); o_err_b += list(np.abs(OY[te] - np.median(OY[tr])))
    opp_eval = {"modelMAE": round(float(np.mean(o_err_m)), 1) if o_err_m else None, "baselineMAE": round(float(np.mean(o_err_b)), 1) if o_err_b else None}
    opp_pred = float(fit_predict(OX, OY, np.array([[cur_form[0], cur_form[1], 1]], float))[0])
    opp_eval["predicted"] = round(opp_pred); opp_eval["median"] = round(sc_skill)
    opp_eval["used"] = bool(o_err_m and np.mean(o_err_m) < np.mean(o_err_b))
    if opp_eval["used"]: sc_skill = opp_pred
    sc_opph = int(round(np.median([m.get("oppHumans", 3) for m in recent])))

    # ---------------- club roster (everyone in the latest EA snapshot plus anyone seen in a game) ----------------
    snaps = sorted(load(os.path.join(db, "snapshots")), key=lambda s: s.get("date", ""))
    fav = {}
    for s in snaps:
        for mem in s.get("members", []) or []:
            if mem.get("p"): fav[mem["p"]] = mem.get("fav") or fav.get(mem["p"])
    roster = sorted(set(fav) | set(apps), key=lambda n: (-apps.get(n, 0), n.lower()))[:10]
    def norm_pos(p):
        p = (p or "").lower()
        return p if p in POS else "midfielder"
    # positions each player may be given: the ones they've actually played (top 2), else EA's favourite position
    allowed = {}
    for n in roster:
        cnt = {}
        for m in matches:
            for p in m["_present"]:
                if p["p"] == n: cnt[p["pos"]] = cnt.get(p["pos"], 0) + 1
        ranked_pos = [k for k, _ in sorted(cnt.items(), key=lambda kv: -kv[1])]
        opts = ([ranked_pos[0]] + [k for k in ranked_pos[1:] if cnt[k] >= 2])[:3] if ranked_pos else []
        if not opts: opts = [norm_pos(fav.get(n))]
        allowed[n] = [o for o in opts if o != "goalkeeper"] or ["midfielder"]
        if n not in main_pos: main_pos[n] = allowed[n][0]
    together = lambda c: sum(1 for m in matches if set(c) <= {p["p"] for p in m["_present"]})

    # ---------------- player models ----------------
    pf = ["oppSkill", "oppHumans", "humans", "sessionGame", "playoff", "position"] + ["is " + n for n in regulars] + ["recentRating5"]
    rows, meta = [], []
    hist_r = {}
    for m in matches:
        for p in m["_present"]:
            h = hist_r.get(p["p"], [])[-5:]
            rows.append([opp_skill(m), m.get("oppHumans", 3), len(m["_present"]), min(m["_game"], 6),
                         1 if m.get("type") == "playoff" else 0, POS.get(p["pos"], 2)] + [1 if p["p"] == n else 0 for n in regulars]
                        + [float(np.mean(h)) if h else float("nan")])
            meta.append((m["_sess"], p))
        for p in m["_present"]: hist_r.setdefault(p["p"], []).append(p["r"])
    PX = np.array(rows, float)
    psess = np.array([s for s, _ in meta])
    players_out = {n: {"p": n, "apps": apps.get(n, 0), "pos": main_pos[n], "pred": {}, "avg": {}, "range": {}} for n in roster}
    player_eval = {}
    pimp = None
    TYP_H = 4
    def prow(n, pos):
        h = hist_r.get(n, [])[-5:]
        return [sc_skill, sc_opph, TYP_H, 2, 1, POS.get(pos, 2)] + [1 if n == r else 0 for r in regulars] + [float(np.mean(h)) if h else float("nan")]
    pos_rating = {}
    for key, label in PLAYER_TARGETS:
        py = np.array([float(p.get(key) or 0) for _, p in meta])
        errs_m, errs_b = [], []
        for s in sorted(set(psess)):
            tr, te = psess < s, psess == s
            if tr.sum() < 30: continue
            pr = fit_predict(PX[tr], py[tr], PX[te])
            # baseline: each player's own average so far (squad average if new)
            base = []
            for i in np.where(te)[0]:
                nm = meta[i][1]["p"]; hist = [py[j] for j in np.where(tr)[0] if meta[j][1]["p"] == nm]
                base.append(np.mean(hist) if hist else np.mean(py[tr]))
            errs_m += list(np.abs(py[te] - pr)); errs_b += list(np.abs(py[te] - np.array(base)))
        player_eval[key] = {"label": label, "modelMAE": round(float(np.mean(errs_m)), 2) if errs_m else None,
                            "baselineMAE": round(float(np.mean(errs_b)), 2) if errs_b else None}
        rmse = float(np.sqrt(np.mean(np.square(errs_m)))) if errs_m else float(np.std(py))
        pp = fit_predict(PX, py, np.array([prow(n, main_pos[n]) for n in roster], float))
        for n, v in zip(roster, pp):
            hist = [py[j] for j in range(len(meta)) if meta[j][1]["p"] == n]
            players_out[n]["pred"][key] = round(float(max(v, 0)), 2)
            players_out[n]["avg"][key] = round(float(np.mean(hist)), 2) if hist else None
            players_out[n]["range"][key] = [round(float(max(v - rmse, 0)), 2), round(float(v + rmse), 2)]
        if key == "r":
            pimp = importance(PX, py, pf)
            pairs = [(n, pos) for n in roster for pos in allowed[n]]
            rv = fit_predict(PX, py, np.array([prow(n, pos) for n, pos in pairs], float))
            for (n, pos), v in zip(pairs, rv): pos_rating.setdefault(n, {})[pos] = round(float(v), 2)
    for n in roster:
        recent_r = [p["r"] for m in matches[-12:] for p in m["_present"] if p["p"] == n]
        players_out[n]["recentRating"] = round(float(np.mean(recent_r)), 2) if recent_r else None
        players_out[n]["recentApps"] = len(recent_r)

    # ---------------- actual starting combinations only ----------------
    # A lineup is the exact set of our humans who played a game. We only ever recommend lineups that have
    # really played together, in the positions they actually used - never an untested mix of players.
    groups = {}
    for m in matches:
        key = tuple(sorted(p["p"] for p in m["_present"]))
        if not key: continue
        groups.setdefault(key, []).append(m)
    ranked = []
    for key, gms in groups.items():
        pos = {}
        for n in key:
            cnt = {}
            for m in gms:
                for p in m["_present"]:
                    if p["p"] == n: cnt[p["pos"]] = cnt.get(p["pos"], 0) + 1
            pos[n] = max(cnt, key=cnt.get)
        W = sum(1 for m in gms if m["gf"] > m["ga"]); D = sum(1 for m in gms if m["gf"] == m["ga"])
        ranked.append({"key": key, "pos": pos, "games": gms, "W": W, "D": D, "L": len(gms) - W - D,
                       "gf": sum(m["gf"] for m in gms), "ga": sum(m["ga"] for m in gms)})
    LXr = np.array([mrow(r["pos"], sc_opph, sc_skill, 1, 2) for r in ranked], float)
    LP = fit_predict(X, y, LXr); LGF = fit_predict(X, YF, LXr); LGA = fit_predict(X, YA, LXr)
    out_lineups = []
    for r, gd, xgf, xga in zip(ranked, LP, LGF, LGA):
        pr = wdl(float(gd), sigma); n = len(r["games"])
        out_lineups.append({"players": [{"p": p, "pos": r["pos"][p]} for p in r["key"]], "size": len(r["key"]),
                            "predGD": round(float(gd), 2), "probs": pr, "xPts": round(3 * pr["W"] + pr["D"], 2),
                            "xGF": round(float(max(xgf, 0)), 1), "xGA": round(float(max(xga, 0)), 1), "gamesTogether": n, "record": [r["W"], r["D"], r["L"]], "gf": r["gf"], "ga": r["ga"],
                            "actualPpg": round((3 * r["W"] + r["D"]) / n, 2), "lastTs": max(m["ts"] for m in r["games"])})
    out_lineups.sort(key=lambda r: (-(r["gamesTogether"] >= 2), -r["xPts"], -r["gamesTogether"]))
    best_by_size = {}
    for r in sorted(out_lineups, key=lambda r: -r["xPts"]):
        best_by_size.setdefault(str(r["size"]), [])
        if len(best_by_size[str(r["size"])]) < 5: best_by_size[str(r["size"])].append(r)
    proven = [r for r in out_lineups if r["gamesTogether"] >= 2] or out_lineups
    best = max(proven, key=lambda r: (r["xPts"], r["gamesTogether"]))
    # ---------------- lineup builder: every subset of the roster, best position assignment ----------------
    # Used when the user picks exactly who is available. Scores combine the results model (which knows the
    # team's results with each regular on the pitch) with each player's predicted rating at each position.
    cand_rows, cand_meta = [], []
    R = len(roster)
    for mask in range(1, 1 << R):
        members = [roster[i] for i in range(R) if mask >> i & 1]
        for assign in itertools.product(*[allowed[n] for n in members]):
            cand_rows.append(mrow(dict(zip(members, assign)), sc_opph, sc_skill, 1, 2))
            cand_meta.append((mask, members, assign))
    CX = np.array(cand_rows, float)
    CP = fit_predict(X, y, CX); CGF = fit_predict(X, YF, CX); CGA = fit_predict(X, YA, CX)
    best_for = {}
    for (mask, members, assign), gd, xgf, xga in zip(cand_meta, CP, CGF, CGA):
        pr = wdl(float(gd), sigma); xp = 3 * pr["W"] + pr["D"]
        score = xp + 0.01 * sum(pos_rating[n][a] for n, a in zip(members, assign))
        if mask not in best_for or score > best_for[mask][0]:
            best_for[mask] = (score, members, assign, float(gd), pr, float(xgf), float(xga))
    code = {"defender": "D", "midfielder": "M", "forward": "F"}
    combos = {str(mask): [round(gd, 2), pr["W"], pr["D"], pr["L"], together(members), "".join(code[a] for a in assign),
                          round(max(xgf, 0), 1), round(max(xga, 0), 1)]
              for mask, (score, members, assign, gd, pr, xgf, xga) in best_for.items()}
    # team results with each player on the pitch, and their own rating by position
    with_stats = {}
    for n in roster:
        gms = [m for m in matches if any(p["p"] == n for p in m["_present"])]
        byp = {}
        for m in gms:
            for p in m["_present"]:
                if p["p"] == n: byp.setdefault(p["pos"], []).append(p["r"])
        with_stats[n] = {"games": len(gms), "ppg": round(sum(3 if m["gf"] > m["ga"] else 1 if m["gf"] == m["ga"] else 0 for m in gms) / len(gms), 2) if gms else None,
                         "gd": round(sum(m["gf"] - m["ga"] for m in gms) / len(gms), 2) if gms else None,
                         "ratingByPos": {k: [round(float(np.mean(v)), 2), len(v)] for k, v in byp.items()}}
    avg_row = np.mean(X[-10:], axis=0, keepdims=True); avg_row[0, 2] = sc_skill; avg_row[0, 3] = 1; avg_row[0, 1] = sc_opph
    avg_row[0, -2] = cur_form[0]; avg_row[0, -1] = cur_form[1]
    gd_avg = float(fit_predict(X, y, avg_row)[0])
    record = {"bestLineup": {"predGD": best["predGD"], "probs": best["probs"]},
              "usualSquad": {"predGD": round(gd_avg, 2), "probs": wdl(gd_avg, sigma)}}
    lineup_table = {"roster": [{"p": n, "apps": apps.get(n, 0), "fav": fav.get(n), "main": main_pos[n],
                                "allowed": allowed[n], "posRating": pos_rating.get(n, {}), "regular": n in regulars,
                                "with": with_stats[n]} for n in roster],
                    "actual": out_lineups, "combos": combos, "fields": ["predGD", "W", "D", "L", "gamesTogether", "positions", "xGF", "xGA"]}

    out = {
        "trainedAt": now, "status": "ok", "version": 1,
        "nMatches": len(matches), "nApps": len(meta), "nSessions": int(max(m["_sess"] for m in matches)),
        "lastMatchTs": matches[-1]["ts"],
        "scenario": {"oppSkill": round(sc_skill), "oppHumans": sc_opph, "type": "playoff"},
        "match": {"target": "goal difference", "sigma": round(sigma, 2),
                  "modelMAE": round(m_mae, 2) if m_mae is not None else None,
                  "baselineMAE": round(b_mae, 2) if b_mae is not None else None,
                  "walkForward": wf, "importance": importance(X, y, mf)[:10]},
        "record": record,
        "scoreline": {"modelMAE": round(float(np.mean(sl_err_m)), 2) if sl_err_m else None, "baselineMAE": round(float(np.mean(sl_err_b)), 2) if sl_err_b else None},
        "oppStrength": opp_eval, "form": {"winRate5": round(cur_form[0], 2), "gd5": round(cur_form[1], 2)},
        "bestLineup": best,
        "lineups": best_by_size,
        "lineupTable": lineup_table,
        "beatsBaseline": {"match": bool(m_mae is not None and b_mae is not None and m_mae < b_mae),
                          **{k: bool(v["modelMAE"] is not None and v["modelMAE"] < v["baselineMAE"]) for k, v in player_eval.items()}},
        "players": sorted(players_out.values(), key=lambda r: (-r["apps"] if r["apps"] < MIN_APPS else 0, -r["pred"].get("r", 0))),
        "playerEval": player_eval, "playerImportance": (pimp or [])[:10],
        "notes": ["Opponent strength uses each opponent's current EA skill rating, not the rating on the day of the game.",
                  "Players need %d games to get their own feature; others count as 'other humans' and use their EA favourite position." % MIN_APPS,
                  "Formation and pitch slots are suggested from where each player performs best; EA's match feed doesn't record formations.",
                  "Only lineups that have actually played together are recommended, in the positions they actually used."],
    }
    try:
        out["alt"] = alt_models({"matches": matches, "opp_skill": opp_skill, "regulars": regulars, "roster": roster, "allowed": allowed,
                                 "pos_rating": pos_rating, "together": together, "sc_skill": sc_skill, "sc_opph": sc_opph,
                                 "meta": meta, "targets": PLAYER_TARGETS, "apps": apps, "main_pos": main_pos})
        out["key"], out["name"] = "xgb", "XGBoost"
    except Exception as e:   # the main model must still be saved if an alternative fails
        out["altError"] = str(e)[:300]
    json.dump(out, open(outp, "w"))
    run = {"trainedAt": now, "nMatches": out["nMatches"], "nApps": out["nApps"], "matchMAE": out["match"]["modelMAE"],
           "baselineMAE": out["match"]["baselineMAE"], "ratingMAE": player_eval["r"]["modelMAE"],
           "ratingBaselineMAE": player_eval["r"]["baselineMAE"], "bestLineup": [p["p"] for p in best["players"]],
           "bestProbs": best["probs"],
           **({"poissonMAE": out["alt"]["poisson"]["match"]["modelMAE"], "kalmanMAE": out["alt"]["kalman"]["match"]["modelMAE"]} if "alt" in out else {})}
    json.dump(run, open(os.path.splitext(outp)[0] + "_run.json", "w"))
    print(json.dumps({k: out[k] for k in ("nMatches", "nApps", "nSessions")}), "matchMAE", out["match"]["modelMAE"], "baseline", out["match"]["baselineMAE"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
