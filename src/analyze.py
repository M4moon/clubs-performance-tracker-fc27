#!/usr/bin/env python3
"""Team and player analysis for Al Haramis FC (EA FC 27 Pro Clubs).

Usage: python src/analyze.py data/raw data/processed results

Needs the CSVs from src/build_tables.py, and results/model/review.json from model/review_model.py
for the model section. Writes tables to results/ and charts to results/figures/, plus
results/headline.json with every number quoted in README.md and RESULTS.md.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- chart style: one palette, light surface, direct labels ----------
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRIT, NEUTRAL = "#0ca30c", "#d03b3b", "#a8a79f"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})

MIN_APPS = 10          # games before a player gets his own row in the player tables
PTS = {"W": 3, "D": 1, "L": 0}


def rate(num, den):
    return float(num) / float(den) if den else float("nan")


def r2(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------- team ----------
def team_summary(m):
    rows = []
    for label, d in [("All games", m), ("League", m[m.type == "league"]), ("Playoffs", m[m.type == "playoff"]),
                     ("Season 1 (7-19 Oct)", m[m.season == 1]), ("Season 2 (from 19 Oct)", m[m.season == 2])]:
        rows.append({"split": label, "games": len(d), "W": int((d.result == "W").sum()), "D": int((d.result == "D").sum()),
                     "L": int((d.result == "L").sum()), "points_per_game": r2(d.points.mean()),
                     "goals_for_per_game": r2(d.gf.mean()), "goals_against_per_game": r2(d.ga.mean()),
                     "shots_per_game": r2(d.shots.mean()), "opp_shots_per_game": r2(d.opp_shots.mean()),
                     "shot_conversion_pct": r2(100 * rate(d.gf.sum(), d.shots.sum()), 1),
                     "opp_shot_conversion_pct": r2(100 * rate(d.ga.sum(), d.opp_shots.sum()), 1),
                     "pass_completion_pct": r2(100 * rate(d.passes_made.sum(), d.passes_att.sum()), 1),
                     "opp_pass_completion_pct": r2(100 * rate(d.opp_passes_made.sum(), d.opp_passes_att.sum()), 1),
                     "pass_share_pct": r2(100 * rate(d.passes_att.sum(), d.passes_att.sum() + d.opp_passes_att.sum()), 1)})
    return pd.DataFrame(rows)


def drivers(m):
    med = m.opponent_skill.median()
    m = m.assign(
        opp_strength=np.where(m.opponent_skill >= med, f"Stronger half (skill {med:.0f}+)", f"Weaker half (under {med:.0f})"),
        numbers=np.select([m.humans > m.opp_humans, m.humans == m.opp_humans], ["More humans than them", "Same number"], "Fewer humans than them"),
        session_part=np.select([m.game_in_session <= 2, m.game_in_session <= 4], ["Games 1-2", "Games 3-4"], "Game 5 or later"))
    order = {"opp_strength": [f"Weaker half (under {med:.0f})", f"Stronger half (skill {med:.0f}+)"],
             "numbers": ["Fewer humans than them", "Same number", "More humans than them"],
             "session_part": ["Games 1-2", "Games 3-4", "Game 5 or later"]}
    names = {"opp_strength": "Opponent skill rating", "numbers": "Human players, us vs them", "session_part": "Game number in a session"}
    rows = []
    for f, groups in order.items():
        for g in groups:
            d = m[m[f] == g]
            rows.append({"factor": names[f], "group": g, "games": len(d), "points_per_game": r2(d.points.mean()),
                         "goal_diff_per_game": r2(d.gd.mean()), "win_pct": r2(100 * (d.result == "W").mean(), 1)})
    # one model with all three at once, so each effect is net of the others
    X = np.column_stack([np.ones(len(m)), (m.opponent_skill - med) / 100, m.humans - m.opp_humans, m.game_in_session.clip(upper=6), (m.type == "playoff").astype(int)])
    beta, *_ = np.linalg.lstsq(X, m.gd.values, rcond=None)
    resid = m.gd.values - X @ beta
    s2 = resid @ resid / (len(m) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    reg = pd.DataFrame({"term": ["intercept (median opponent, equal numbers, game 0)", "per +100 opponent skill rating", "per extra human vs opponent",
                                 "per later game in session (capped at 6)", "playoff game"],
                        "goal_diff_effect": np.round(beta, 2), "std_error": np.round(se, 2)})
    return pd.DataFrame(rows), reg, med


# ---------- players ----------
def squad_table(a, m):
    # goal difference after allowing for opponent strength: actual minus what the opponent's skill rating alone predicts
    slope, icpt = np.polyfit(m.opponent_skill, m.gd, 1)
    m = m.assign(gd_adj=m.gd - (icpt + slope * m.opponent_skill))
    rows = []
    for p, d in a.groupby("player"):
        ids = set(d.match_id)
        on, off = m[m.match_id.isin(ids)], m[~m.match_id.isin(ids)]
        pos = d.position.value_counts()
        rows.append({"player": p, "games": len(d), "positions": ", ".join(f"{k} {v}" for k, v in pos.items()),
                     "avg_rating": r2(d.rating.mean()), "goals_per_game": r2(d.goals.mean()), "assists_per_game": r2(d.assists.mean()),
                     "shots_per_game": r2(d.shots.mean()), "shot_conversion_pct": r2(100 * rate(d.goals.sum(), d.shots.sum()), 1),
                     "passes_made_per_game": r2(d.passes_made.mean(), 1), "pass_completion_pct": r2(100 * rate(d.passes_made.sum(), d.passes_att.sum()), 1),
                     "tackles_won_per_game": r2(d.tackles_won.mean()), "tackle_success_pct": r2(100 * rate(d.tackles_won.sum(), d.tackles_att.sum()), 1),
                     "motm": int(d.motm.sum()), "team_ppg_with": r2(on.points.mean()), "team_ppg_without": r2(off.points.mean()) if len(off) else None,
                     "team_gd_with": r2(on.gd.mean()), "team_gd_without": r2(off.gd.mean()) if len(off) else None,
                     "team_gd_vs_expected_with": r2(on.gd_adj.mean()), "team_gd_vs_expected_without": r2(off.gd_adj.mean()) if len(off) else None,
                     # with minus without, and its standard error (Welch); |difference| under ~2 SE is within chance
                     "with_minus_without": r2(on.gd_adj.mean() - off.gd_adj.mean()) if len(off) > 1 else None,
                     "with_minus_without_se": r2(np.sqrt(on.gd_adj.var() / len(on) + off.gd_adj.var() / len(off))) if len(off) > 1 and len(on) > 1 else None,
                     "avg_opponent_skill": r2(on.opponent_skill.mean(), 0)})
    return pd.DataFrame(rows).sort_values("games", ascending=False)


def player_trends(a):
    """Each player's own first half of games against his second half, with how hard the opponents were."""
    rows = []
    for p, d in a.sort_values("ts").groupby("player"):
        if len(d) < MIN_APPS:
            continue
        h = len(d) // 2
        halves = {"first": d.iloc[:h], "second": d.iloc[h:]}
        r = {"player": p, "games": len(d)}
        for k, x in halves.items():
            r[f"{k}_games"] = len(x)
            r[f"{k}_rating"] = r2(x.rating.mean())
            r[f"{k}_ga_per_game"] = r2((x.goals + x.assists).mean())
            r[f"{k}_pass_pct"] = r2(100 * rate(x.passes_made.sum(), x.passes_att.sum()), 1)
            r[f"{k}_tackle_pct"] = r2(100 * rate(x.tackles_won.sum(), x.tackles_att.sum()), 1)
            r[f"{k}_opp_skill"] = r2(x.opponent_skill.mean(), 0)
        rows.append(r)
    return pd.DataFrame(rows)


def pop_pct(q, v):
    """Where v sits among players at the same position (community quantiles at 10/25/50/75/90)."""
    if q is None or v is None or np.isnan(v):
        return float("nan")
    P = [10, 25, 50, 75, 90]
    if v < q[0]:
        return 5.0
    if v > q[4]:
        return 95.0
    eq = [P[i] for i in range(5) if q[i] == v]
    if eq:
        return (eq[0] + eq[-1]) / 2
    for i in range(4):
        if q[i] <= v <= q[i + 1]:
            return P[i] + (P[i + 1] - P[i]) * (v - q[i]) / (q[i + 1] - q[i])
    return float("nan")


# metrics shown in the player profile: (key, label, better direction)
# (style-only counts such as shots off target or share of passes forward are left out: more of them is neither good nor bad)
PROFILE = [("pmade", "Passes completed", 1), ("thru", "Through balls", 1), ("sot", "Shots on target", 1), ("beat", "Players beaten", 1),
           ("tw", "Tackles won", 1), ("int", "Interceptions", 1), ("won", "Possession won", 1), ("lost", "Possession lost", -1),
           ("posPct", "In position (share of prompts)", 1), ("decPct", "Good or best option taken", 1)]


def event_profile(a, bench):
    ev = a[a.has_events == 1]
    rows = []
    for p, d in ev.groupby("player"):
        if len(d) < MIN_APPS:
            continue
        pos = d.position.value_counts().idxmax()
        q = bench["positions"][pos]["q"]
        v = {k[3:]: d[k].mean() for k in d.columns if k.startswith("ev_")}
        v["posPct"] = 100 * rate(d.ev_inPos.sum(), d.ev_inPos.sum() + d.ev_outPos.sum())
        v["decPct"] = 100 * rate(d.ev_good.sum() + d.ev_best.sum(), d.ev_good.sum() + d.ev_best.sum() + d.ev_elsewhere.sum())
        v["fwdShare"] = 100 * rate(d.ev_fwd.sum(), d.ev_pmade.sum())
        for key, label, better in PROFILE:
            pc = pop_pct(q.get(key), v[key])
            rows.append({"player": p, "event_games": len(d), "main_position": pos, "metric": label, "key": key, "better": better,
                         "per_game_or_pct": r2(v[key]), "typical_at_position": r2(q[key][2]) if key in q else None,
                         "raw_percentile": r2(pc, 0), "quality_percentile": r2(pc if better > 0 else 100 - pc, 0)})
    return pd.DataFrame(rows)


def ordinal(x):
    n = int(round(x))
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def coaching_notes(prof):
    """Two strengths and two things to work on per player, from the position percentiles."""
    rows = []
    for p, d in prof.groupby("player"):
        d = d.dropna(subset=["quality_percentile"]).sort_values("quality_percentile")
        weak = d[d.quality_percentile <= 25].head(3)
        strong = d[d.quality_percentile >= 75].sort_values("quality_percentile", ascending=False).head(2)
        where = lambda q: "bottom 10%" if q <= 5 else "top 10%" if q >= 95 else f"{ordinal(q)} percentile"
        unit = lambda r: "%" if r.key.endswith("Pct") else " a game"
        fmt = lambda r: f"{r.metric} {r.per_game_or_pct:.1f}{unit(r)} vs typical {r.typical_at_position:.1f}{unit(r).strip() if unit(r) == '%' else ''} ({where(r.quality_percentile)})"
        rows.append({"player": p, "main_position": d.main_position.iloc[0], "event_games": int(d.event_games.iloc[0]),
                     "strengths": "; ".join(fmt(r) for r in strong.itertuples()) or "none in the top quarter",
                     "work_on": "; ".join(fmt(r) for r in weak.itertuples()) or "none in the bottom quarter"})
    return pd.DataFrame(rows)


def team_event_profile(a, bench):
    """Squad-wide decision and possession numbers against the community median for the same positions."""
    ev = a[a.has_events == 1]
    out = {}
    good = ev.ev_good.sum() + ev.ev_best.sum()
    out["decision_pct"] = 100 * rate(good, good + ev.ev_elsewhere.sum())
    out["in_position_pct"] = 100 * rate(ev.ev_inPos.sum(), ev.ev_inPos.sum() + ev.ev_outPos.sum())
    out["possession_lost_per_game"] = ev.ev_lost.mean()
    out["possession_won_per_game"] = ev.ev_won.mean()
    # the typical player at each of our appearances' positions
    med = lambda k: np.mean([bench["positions"][p]["q"][k][2] for p in ev.position])
    out["typical_decision_pct"] = med("decPct")
    out["typical_in_position_pct"] = med("posPct")
    out["typical_possession_lost"] = med("lost")
    out["typical_possession_won"] = med("won")
    out["event_appearances"] = int(len(ev))
    return {k: r2(v, 1) for k, v in out.items()}


# ---------- missed games ----------
def missed_games(m, snaps):
    """Compare EA's own running record (daily snapshot) with the games captured over the same window."""
    s = snaps.sort_values("takenAt").reset_index(drop=True)
    s["t"] = pd.to_datetime(s.takenAt, utc=True).map(lambda x: int(x.timestamp()))
    base = s.iloc[0]
    rows = []
    for _, r in s.iterrows():
        cap = m[(m.ts > base.t) & (m.ts <= r.t)]
        rows.append({"snapshot": r.takenAt, "ea_games_since_first": int(r.gamesPlayed - base.gamesPlayed),
                     "captured_since_first": len(cap), "ea_W": int(r.wins - base.wins), "ea_D": int(r.ties - base.ties), "ea_L": int(r.losses - base.losses),
                     "cap_W": int((cap.result == "W").sum()), "cap_D": int((cap.result == "D").sum()), "cap_L": int((cap.result == "L").sum())})
    out = pd.DataFrame(rows)
    out["missing"] = out.ea_games_since_first - out.captured_since_first
    return out


# ---------- model ----------
def model_review(review):
    G = [g for g in review["games"] if "pred" in g]
    call = lambda p: "W" if p["W"] >= p["D"] and p["W"] >= p["L"] else ("L" if p["L"] >= p["D"] else "D")
    res = {"games_forecast": len(G),
           "results_called_model": sum(call(g["pred"]) == g["res"] for g in G),
           "results_called_baseline": sum(call(g["pred"]["base"]) == g["res"] for g in G),
           "gd_mae_model": r2(np.mean([abs(g["gf"] - g["ga"] - g["pred"]["gd"]) for g in G])),
           "gd_mae_baseline": r2(np.mean([abs(g["gf"] - g["ga"] - g["pred"]["base"]["gd"]) for g in G])),
           "expected_points": r2(sum(3 * g["pred"]["W"] + g["pred"]["D"] for g in G), 1),
           "actual_points": sum(PTS[g["res"]] for g in G)}
    cal = []
    for label, lo, hi in [("under 30%", 0, .3), ("30-50%", .3, .5), ("50% or more", .5, 1.01)]:
        b = [g for g in G if lo <= g["pred"]["W"] < hi]
        cal.append({"model_win_chance": label, "games": len(b), "average_forecast_pct": r2(100 * np.mean([g["pred"]["W"] for g in b]), 1) if b else None,
                    "actual_win_pct": r2(100 * np.mean([g["res"] == "W" for g in b]), 1) if b else None})
    # player stats: did the model beat each player's own running average?
    per = {}
    for ap in review["apps"]:
        for k, v in ap["pred"].items():
            e, b = abs(ap["act"][k] - v), abs(ap["act"][k] - ap["base"][k])
            s = per.setdefault(k, {"n": 0, "model": 0.0, "own_avg": 0.0})
            s["n"] += 1; s["model"] += e; s["own_avg"] += b
    labels = {"r": "Match rating", "g": "Goals", "a": "Assists", "sh": "Shots", "pm": "Passes made", "tm": "Tackles won"}
    prow = [{"stat": labels[k], "forecasts": s["n"], "model_mae": r2(s["model"] / s["n"]), "own_average_mae": r2(s["own_avg"] / s["n"])} for k, s in per.items()]
    return res, pd.DataFrame(cal), pd.DataFrame(prow)


# ---------- charts ----------
def fig_form(m, path):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(1, len(m) + 1)
    col = m.result.map({"W": GOOD, "D": NEUTRAL, "L": CRIT})
    for i, (_, r) in enumerate(m.iterrows()):
        if r.type == "playoff":
            ax.axvspan(x[i] - .5, x[i] + .5, color="#eda100", alpha=.13, lw=0)
    ax.bar(x, m.gd, color=col, width=.75)
    roll = m.gd.rolling(5, min_periods=3).mean()
    ax.plot(x, roll, color=BLUE, lw=2.2)
    ax.axhline(0, color=AXIS, lw=1)
    s2 = int(m[m.season == 2].index.min()) + 1 if (m.season == 2).any() else None
    if s2:
        ax.axvline(s2 - .5, color=INK2, lw=1, ls=(0, (3, 2)))
        ax.text(s2 - .2, m.gd.max() + .3, "Season 2", color=INK2, fontsize=9, va="bottom")
    ax.text(x[-1] + .8, roll.iloc[-1], "5-game\naverage", color=BLUE, fontsize=9, va="center")
    ax.set_xlim(0, len(m) + 4)
    ax.set_ylabel("Goal difference")
    d0, d1 = pd.to_datetime(m.kickoff_utc.iloc[0]), pd.to_datetime(m.kickoff_utc.iloc[-1])
    ax.set_xlabel(f"Game number ({d0.day} {d0:%b} to {d1.day} {d1:%b %Y})")
    ax.set_title("Every game's goal difference (green win, grey draw, red loss; shaded = playoff)")
    save(fig, path)


def fig_drivers(dr, path):
    facs = list(dict.fromkeys(dr.factor))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, f in zip(axes, facs):
        d = dr[dr.factor == f]
        b = ax.bar(range(len(d)), d.points_per_game, color=BLUE, width=.6)
        for i, r in enumerate(d.itertuples()):
            ax.text(i, r.points_per_game + .05, f"{r.points_per_game:.2f}\n{r.games} games", ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([g.replace(" (", "\n(").replace(" than them", "\nthan them") for g in d.group], fontsize=8.5)
        ax.set_title(f, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Points per game")
    axes[0].set_ylim(0, 2.4)
    fig.suptitle("What goes with winning", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, path)


def fig_shooting(ts, path):
    t = ts.iloc[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1]})
    cats = ["Shots", "Goals"]
    us = [t.shots_per_game, t.goals_for_per_game]; th = [t.opp_shots_per_game, t.goals_against_per_game]
    xs = np.arange(len(cats))
    a1.bar(xs - .18, us, .34, color=BLUE, label="Al Haramis"); a1.bar(xs + .18, th, .34, color=ORANGE, label="Opponents")
    for i in range(len(cats)):
        a1.text(i - .18, us[i] + .1, f"{us[i]:.1f}", ha="center", fontsize=9); a1.text(i + .18, th[i] + .1, f"{th[i]:.1f}", ha="center", fontsize=9)
    a1.set_ylim(0, 8); a1.set_xticks(xs); a1.set_xticklabels(cats); a1.set_title("Per game"); a1.legend(loc="upper right"); a1.grid(axis="x", visible=False)
    cats2 = ["Shots scored", "Passes completed"]
    us2 = [t.shot_conversion_pct, t.pass_completion_pct]; th2 = [t.opp_shot_conversion_pct, t.opp_pass_completion_pct]
    a2.bar(xs - .18, us2, .34, color=BLUE); a2.bar(xs + .18, th2, .34, color=ORANGE)
    for i in range(len(cats2)):
        a2.text(i - .18, us2[i] + 1, f"{us2[i]:.0f}%", ha="center", fontsize=9); a2.text(i + .18, th2[i] + 1, f"{th2[i]:.0f}%", ha="center", fontsize=9)
    a2.set_xticks(xs); a2.set_xticklabels(cats2); a2.set_ylim(0, 100); a2.set_title("Success rate"); a2.grid(axis="x", visible=False)
    fig.suptitle("Same number of shots, fewer goals: the gap is finishing and keeping the ball", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, path)


def fig_player_trends(a, path):
    players = [p for p, n in a.player.value_counts().items() if n >= 15]
    n = len(players); cols = 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows), sharey=True)
    for ax, p in zip(axes.flat, players):
        d = a[a.player == p].sort_values("ts")
        x = np.arange(1, len(d) + 1)
        ax.scatter(x, d.rating, s=10, color=AXIS)
        ax.plot(x, d.rating.rolling(5, min_periods=3).mean(), color=BLUE, lw=2)
        ax.axhline(d.rating.mean(), color=INK2, lw=.8, ls=(0, (3, 2)))
        ax.set_title(f"{p}  ·  avg {d.rating.mean():.1f}", fontsize=10)
        ax.set_ylim(3.5, 10.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("EA match rating")
    for ax in axes[-1]:
        ax.set_xlabel("Player's game number")
    fig.suptitle("Match rating per game, with a 5-game average (dashed = player's average)", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, path)


def fig_percentiles(prof, path):
    from matplotlib.colors import LinearSegmentedColormap
    piv = prof.pivot(index="player", columns="metric", values="quality_percentile")
    order = [l for _, l, _ in PROFILE if l in piv.columns]
    piv = piv[order]
    games = prof.groupby("player").event_games.first()
    piv = piv.loc[games.sort_values(ascending=False).index]
    cmap = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(piv) + 2.2))
    ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color=INK if 20 < v < 80 else "white")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    pos = prof.groupby("player").main_position.first()
    ax.set_yticks(range(len(piv))); ax.set_yticklabels([f"{p} ({pos[p]}, {games[p]} gm)" for p in piv.index], fontsize=9)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Percentile against players at the same position in 51,395 community player-matches\n(higher is better on every column; blue = top, red = bottom)", fontsize=10.5)
    save(fig, path)


def fig_model(runs, path):
    r = runs[runs.get("model") == "poisson"].sort_values("trainedAt") if "model" in runs else runs.iloc[0:0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    x = np.arange(len(r))
    labels = [f"{t[5:10]}\n{n}" for t, n in zip(r.trainedAt, r.nMatches)]
    for ax, (m, b, title) in zip((a1, a2), [("logLoss", "baselineLogLoss", "Win / draw / loss log loss"), ("matchMAE", "baselineMAE", "Average miss on goal difference")]):
        ax.plot(x, r[b], color=MUTED, lw=2, ls=(0, (4, 2)))
        ax.plot(x, r[m], color=BLUE, lw=2.2, marker="o", ms=4)
        ax.text(x[-1] + .25, r[m].iloc[-1], "Model", color=BLUE, fontsize=9, va="center")
        ax.text(x[-1] + .25, r[b].iloc[-1], "Club average", color=MUTED, fontsize=9, va="center")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5); ax.set_xlim(-.5, len(r) + 1.6)
        ax.set_title(title, fontsize=10)
    a1.set_xlabel("Retrain date and games available"); a2.set_xlabel("Retrain date and games available")
    fig.suptitle("Walk-forward error after each retrain of the current model (lower is better)", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, path)


def decisions_under_pressure(a, m):
    """EA's in-game decision and positioning feedback, split by how much the game mattered and how hard it was."""
    ev = a[a.has_events == 1].merge(m[["match_id", "type", "game_in_session"]].rename(columns={"type": "mtype", "game_in_session": "gis"}), on="match_id")
    med = m.opponent_skill.median()
    splits = [("Match type", "League", ev.mtype == "league"), ("Match type", "Playoff", ev.mtype == "playoff"),
              ("Opponent skill", f"Weaker half (under {med:.0f})", ev.opponent_skill < med), ("Opponent skill", f"Stronger half ({med:.0f}+)", ev.opponent_skill >= med),
              ("Game in session", "Games 1-2", ev.gis <= 2), ("Game in session", "Games 3-4", ev.gis.between(3, 4)), ("Game in session", "Game 5 or later", ev.gis >= 5)]
    rows = []
    for factor, label, mask in splits:
        d = ev[mask]
        good = d.ev_good.sum() + d.ev_best.sum()
        rows.append({"factor": factor, "group": label, "appearances": len(d), "games": d.match_id.nunique(),
                     "good_or_best_option_pct": r2(100 * rate(good, good + d.ev_elsewhere.sum()), 1),
                     "in_position_pct": r2(100 * rate(d.ev_inPos.sum(), d.ev_inPos.sum() + d.ev_outPos.sum()), 1),
                     "possession_lost_per_player_game": r2(d.ev_lost.mean())})
    return pd.DataFrame(rows)


def main(raw, proc, out):
    figs = os.path.join(out, "figures")
    os.makedirs(figs, exist_ok=True)
    m = pd.read_csv(os.path.join(proc, "matches.csv")).sort_values("ts").reset_index(drop=True)
    review = json.load(open(os.path.join(out, "model", "review.json")))
    season_of = {g["id"]: g["season"] for g in review["games"]}
    m["season"] = m.match_id.astype(str).map(season_of)
    a = pd.read_csv(os.path.join(proc, "player_appearances.csv"))
    a = a[a.seconds > 0].merge(m[["match_id", "opponent_skill", "gd"]], on="match_id")   # only players who were on the pitch
    snaps = pd.read_csv(os.path.join(proc, "snapshots.csv"))
    bench = json.load(open(os.path.join(raw, "benchmarks", "events.json")))
    runs = pd.DataFrame([json.load(open(os.path.join(raw, "modelruns", f))) for f in sorted(os.listdir(os.path.join(raw, "modelruns")))])
    latest = json.load(open(os.path.join(out, "model", "model_latest.json")))

    ts = team_summary(m); ts.to_csv(os.path.join(out, "team_summary.csv"), index=False)
    dr, reg, med = drivers(m); dr.to_csv(os.path.join(out, "results_drivers.csv"), index=False); reg.to_csv(os.path.join(out, "goal_difference_regression.csv"), index=False)
    sq = squad_table(a, m); sq.to_csv(os.path.join(out, "squad_table.csv"), index=False)
    tr = player_trends(a); tr.to_csv(os.path.join(out, "player_trends.csv"), index=False)
    prof = event_profile(a, bench); prof.to_csv(os.path.join(out, "player_event_profile.csv"), index=False)
    notes = coaching_notes(prof); notes.to_csv(os.path.join(out, "player_coaching_notes.csv"), index=False)
    mg = missed_games(m, snaps); mg.to_csv(os.path.join(out, "missed_games_check.csv"), index=False)
    rv, cal, prow = model_review(review)
    cal.to_csv(os.path.join(out, "model_calibration.csv"), index=False); prow.to_csv(os.path.join(out, "model_player_errors.csv"), index=False)
    runs.drop(columns=["bestLineup", "bestProbs"], errors="ignore").to_csv(os.path.join(out, "model_runs.csv"), index=False)

    fig_form(m, os.path.join(figs, "1_form.png"))
    fig_drivers(dr, os.path.join(figs, "2_what_goes_with_winning.png"))
    fig_shooting(ts, os.path.join(figs, "3_shooting_and_passing.png"))
    fig_player_trends(a, os.path.join(figs, "4_player_ratings.png"))
    fig_percentiles(prof, os.path.join(figs, "5_player_percentiles.png"))
    fig_model(runs, os.path.join(figs, "6_model_error.png"))
    dp = decisions_under_pressure(a, m); dp.to_csv(os.path.join(out, "decisions_under_pressure.csv"), index=False)

    head = {"games": len(m), "first_game": m.kickoff_utc.iloc[0], "last_game": m.kickoff_utc.iloc[-1],
            "sessions": int(m.session.max()), "team": ts.set_index("split").to_dict("index"),
            "drivers": dr.to_dict("records"), "regression": reg.to_dict("records"), "median_opp_skill": med,
            "ai_goals": int(m.ai_goals.sum()), "goals_for": int(m.gf.sum()),
            "squad": sq.to_dict("records"), "trends": tr.to_dict("records"),
            "team_events": team_event_profile(a, bench), "coaching": notes.to_dict("records"),
            "missed_games_final": mg.iloc[-1].to_dict(),
            "model_review": rv, "calibration": cal.to_dict("records"), "player_model": prow.to_dict("records"),
            "decisions_under_pressure": dp.to_dict("records"),
            "latest_model": {"trainedAt": latest["trainedAt"], "name": latest["name"], "nMatches": latest["nMatches"], **latest["match"],
                             "walkForward": None, "importance": None, "scoreline": latest["scoreline"], "playerEval": latest["playerEval"],
                             "oppSkillMethod": latest["oppSkillMethod"], "selection": latest["selection"], "effects": latest["effects"]}}
    json.dump(head, open(os.path.join(out, "headline.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"{len(m)} games, {len(a)} appearances; tables and {len(os.listdir(figs))} charts written to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
