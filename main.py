"""GTA VI trophy-rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

Model (logit space, hierarchical, regime-pooled):

* Refresh detection is GLOBAL. A refresh is a read where players_tracked,
  trophies_earned, or any rate differs from the previous read. Consecutive
  identical reads are stale (Exophase had not refreshed) and collapse to a
  single value. A trophy with 0 earners across many refreshes is genuine
  information (players grew, earners stayed 0) and is kept.
* For each kept refresh and each trophy, earners = rate * players / 100,
  and the smoothed proportion p = (earners + 0.5) / (players + 1) (Jeffreys
  pseudo-count) gives a finite logit at 0 earners and at 100%.
* Time is x = ln(1 + hours since release 2026-11-19T00:00Z), from each read's
  `at`. The players covariate is w = ln(players_tracked).
* Collinearity guard (the key stability point): log-players and log-time
  grow together, so a naive regression of logit(rate) on x and w splits the
  trend unstably between the two and explodes out of sample. Instead w is
  residualized against x (Frisch-Waugh-Lovell): u = w - (alpha + beta*x),
  the part of players growth NOT explained by time, i.e. the refresh jumps.
  The trophy regression is logit(p) = a + b*x + c*u, so b is the pure time
  trend and c is the dilution effect identified only from the jumps. c is
  shrunk toward its theoretical value -1 (earners fixed => rate ~ 1/players)
  in proportion to how much u actually varies. A quadratic term d*x^2 is
  added for rare / delayed-onset trophies, heavily shrunk.
* The players level at the target is extrapolated from the w-on-x fit with
  a slope damped toward zero as the horizon grows (players growth slows),
  then residualized the same way to give u at the target.
* Slopes (and curvature) are pooled toward their regime mean, weighted by
  how many informative refreshes the trophy has (empirical-Bayes shrinkage).
  A negative slope for a Saturated/Diluting trophy is additionally halved
  (SAT_SLOPE_DAMP): a high-rate trophy's decline is a mix shift, not a
  logit-linear process, and extrapolating the steep logit drop from 100%
  linearly overshoots downward.
* When the model has too few refreshes to fit a trend it falls back to
  persistence, but with a launch-spike dilution prior: a flat history
  usually means the data is STALE (Exophase has not refreshed since before
  the release), so by the target new players dilute the rates. The median
  is shifted down by a dilution factor from the projected players growth
  over the unobserved window, and the spreads are wide and low-side-heavy
  (the dilution pushes the truth below the median).
* A runtime backtest splits the refreshes ~60/40, fits on the early part,
  predicts the later refreshes, and picks the damping / shrinkage /
  linear-vs-quadratic setting with the lowest mean logit-pinball loss.
* Quantiles are regime-specific and asymmetric: narrow for mature trophies,
  wide and right-skewed for rare / delayed-onset trophies, widest on the
  side that can move for boundary trophies. The 5%/95% tails are never
  heroically narrow; the 25%/75% inner quartiles are kept tight.

Standard library only. Deterministic. No network. Fast (well under 8 s).
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, and the standard normal quantiles.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release of the game; x = ln(1 + hours since this instant).
RELEASE = datetime.fromisoformat("2026-11-19T00:00:00+00:00")

LOGIT_9999 = math.log(99.99 / 0.01)   # 9.210340...
LOGIT_0001 = -LOGIT_9999

# Grid the internal backtest searches (default first; only a strictly lower
# backtest loss replaces it, so ties keep the conservative default).
DAMP_GRID = [0.5, 0.3, 0.7, 1.0]
SHRINK_GRID = [5, 2, 10, 20]
QUAD_GRID = [False, True]

# Horizon damping for the players extrapolation: the slope is damped toward
# zero as the horizon grows (players growth slows).
DAMP_HORIZON_K = 0.35
# Shrinkage of the dilution coefficient toward -1: c is only identified
# from refresh jumps, so when u barely varies the theoretical value wins.
C_SHRINK_K = 0.02
# Damping of a NEGATIVE logit slope for Saturated/Diluting trophies: a
# high-rate trophy's decline is a mix shift (new players diluting the
# early-adopter share), not a logit-linear process. The logit of a rate near
# 100% is extremely steep, so the fitted decline overstates the forward
# drift and extrapolating it linearly overshoots downward (the 100% trophy's
# true decline decelerates as the new-player share stabilizes). Halve the
# extrapolated negative slope for saturated trophies; rising or flat slopes
# are untouched.
SAT_SLOPE_DAMP = 0.5

# Launch-spike dilution prior (persistence path only, see
# persistence_dilution). When the model has too few refreshes to fit a
# trend it falls back to persistence (last rate). But in launch week a flat
# history usually means the data is STALE (Exophase has not refreshed since
# before the release), not that the player base is static. By the target,
# new players will have joined and diluted the rates: a trophy's rate falls
# as players who have not earned it join. The dilution is estimated from
# the projected players growth over the unobserved window (last distinct
# refresh to the target). Players grow at a launch-spike rate right after
# the midnight release that decays to a sustained launch-week rate, so the
# growth over a long window (the final's 4.5 days) stays sane while the
# first-hours spike is captured. Earners grow at only a fraction of the
# players rate (new players earn slowly).
LAUNCH_SPIKE_PER_DAY = 10.0   # log-players/day at the release spike
LAUNCH_BASE_PER_DAY = 0.5     # sustained log-players/day, launch week
LAUNCH_DECAY_DAYS = 0.5       # spike decay timescale (12 h)
EARNERS_GROWTH_FRAC = 0.2
# Wide, low-side-heavy spreads for the persistence path. The dilution
# pushes the truth BELOW the median, so the low side must be wide enough
# to contain it; the inner quartiles are widened too, because with no
# trend to fit the median itself is highly uncertain. The widths are a
# balance: narrow enough that the pinball loss stays low now that the
# dilution prior puts the median close, wide enough to cover the
# uncertainty in the dilution factor itself.
PERSIST_SIGMAS = (0.30, 1.00, 0.30, 0.80)  # low_in, low_out, high_in, high_out


def damp_saturated_slope(b_s, regime):
    """Damp a negative logit slope for a Saturated/Diluting trophy.

    Returns b_s * SAT_SLOPE_DAMP when the trophy is saturated and its pooled
    slope is negative, else b_s unchanged. Applied in both the backtest and
    the forecast so the two see the same model.
    """
    if regime == "saturated" and b_s < 0.0:
        return b_s * SAT_SLOPE_DAMP
    return b_s


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(l):
    # Stable for extreme logits.
    if l >= 0.0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches JavaScript Math.round)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def log_time(ts):
    """x = ln(1 + hours since release)."""
    hours = max(0.0, (ts - RELEASE.timestamp()) / 3600.0)
    return math.log1p(hours)


# --------------------------------------------------------------------------
# Small linear algebra: least squares via normal equations.
# --------------------------------------------------------------------------

def solve_linear(mat, vec):
    """Solve mat @ x = vec by Gaussian elimination with partial pivoting.
    Returns the solution list, or None when the matrix is (near-)singular."""
    n = len(vec)
    a = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        pv = a[col][col]
        for r in range(col + 1, n):
            f = a[r][col] / pv
            if f == 0.0:
                continue
            for c in range(col, n + 1):
                a[r][c] -= f * a[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = a[i][n]
        for j in range(i + 1, n):
            s -= a[i][j] * x[j]
        x[i] = s / a[i][i]
    return x


def ols(design, ys):
    """Least squares for ys ~ design rows. design is a list of feature
    lists (same length). Returns the coefficient list or None."""
    k = len(design[0])
    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for f, y in zip(design, ys):
        for i in range(k):
            xty[i] += f[i] * y
            fi = f[i]
            row = xtx[i]
            for j in range(k):
                row[j] += fi * f[j]
    for i in range(k):
        xtx[i][i] += 1e-9 * (1.0 + abs(xtx[i][i]))
    return solve_linear(xtx, xty)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((v - m) ** 2 for v in xs) / len(xs)


def fit_trophy(points, quad, w_on_x):
    """Fit one trophy's points: (x, w, y) with w raw log-players.

    w is residualized against x first (u = w - (alpha + beta*x)) so the
    regression logit(p) = a + b*x + c*u is stable under the near-collinearity
    of log-time and log-players. The dilution coefficient c is shrunk toward
    its theoretical value -1 in proportion to how much u varies.

    Returns (a, b, c, d, used_u, used_quad) or None when degenerate.
    """
    if len(points) < 2:
        return None
    ys = [p[2] for p in points]
    # Residualized players covariate.
    us = None
    if w_on_x is not None:
        alpha, beta = w_on_x
        us = [p[1] - (alpha + beta * p[0]) for p in points]
        if max(us) - min(us) < 1e-6:
            us = None
    # Try the full model, then drop the quadratic, then drop u.
    for (q, use_u) in ((quad, us is not None), (False, us is not None),
                       (quad, False), (False, False)):
        need = 2 + (1 if use_u else 0) + (1 if q else 0)
        if len(points) < need:
            continue
        design = []
        for idx, (x, _w, _y) in enumerate(points):
            f = [1.0, x]
            if use_u:
                f.append(us[idx])
            if q:
                f.append(x * x)
            design.append(f)
        sol = ols(design, ys)
        if sol is None:
            continue
        a, b = sol[0], sol[1]
        pos = 2
        c = 0.0
        if use_u:
            c = sol[pos]
            pos += 1
        d = 0.0
        if q:
            d = sol[pos]
        if not all(math.isfinite(v) for v in (a, b, c, d)):
            continue
        if abs(b) > 8.0 or abs(c) > 8.0 or abs(d) > 8.0:
            continue
        # Shrink the dilution coefficient toward -1 by how well u is
        # identified (its variance). Weak identification -> trust theory.
        if use_u:
            var_u = _var(us)
            wgt = var_u / (var_u + C_SHRINK_K)
            c = wgt * c + (1.0 - wgt) * (-1.0)
            c = min(max(c, -3.0), 0.5)
        return (a, b, c, d, use_u, q)
    return None


# --------------------------------------------------------------------------
# Case preparation.
# --------------------------------------------------------------------------

def dedup_refreshes(history):
    """Collapse runs of consecutive identical reads to a single value.

    A refresh is a read where players_tracked, trophies_earned, or any rate
    differs from the previous read. Returns the kept reads, oldest first.
    """
    kept = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), read.get("trophies_earned"),
               tuple(rates))
        if key != prev_key:
            kept.append(read)
            prev_key = key
    return kept


def trophy_points(trophies, refreshes):
    """Per trophy, the list of (refresh_index, x, w, y, rate, earners).

    y is the logit of the Jeffreys-smoothed proportion
    p = (earners + 0.5) / (players + 1). Null rates are skipped for that
    trophy only. w is the RAW log-players (residualized later, per fit).
    """
    n = len(trophies)
    data = [[] for _ in range(n)]
    for r_idx, read in enumerate(refreshes):
        ts = parse_ts(read["at"])
        x = log_time(ts)
        players = read.get("players_tracked") or 0
        if players <= 0:
            continue
        w = math.log(players)
        rates = read.get("rates") or []
        for i in range(n):
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            earners = r * players / 100.0
            p = (earners + 0.5) / (players + 1.0)
            y = math.log(p / (1.0 - p))
            data[i].append((r_idx, x, w, y, float(r), earners))
    return data


def players_fit(refreshes):
    """OLS of ln(players) on x over the refreshes.

    Returns (alpha, beta, x_last, w_last) or None when there are fewer than
    two usable refreshes. The same (alpha, beta) is used both to extrapolate
    players to the target and to residualize w against x.
    """
    pts = []
    for read in refreshes:
        players = read.get("players_tracked") or 0
        if players <= 0:
            continue
        pts.append((log_time(parse_ts(read["at"])), math.log(players)))
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ws = [p[1] for p in pts]
    mx = _mean(xs)
    mw = _mean(ws)
    sxx = sum((v - mx) ** 2 for v in xs)
    if sxx <= 0.0:
        return None
    beta = sum((a - mx) * (b - mw) for a, b in zip(xs, ws)) / sxx
    alpha = mw - beta * mx
    return (alpha, beta, xs[-1], ws[-1])


def extrapolate_w(pfit, x_target, horizon_days, damp):
    """Extrapolate ln(players) to the target with a damped slope.

    The slope is damped toward zero as the horizon grows (players growth
    slows). Returns None when the players fit is unavailable.
    """
    if pfit is None:
        return None
    _alpha, beta, x_last, w_last = pfit
    beta = min(max(beta, 0.0), 2.0)
    damp_eff = damp / (1.0 + DAMP_HORIZON_K * horizon_days)
    return w_last + beta * (x_target - x_last) * damp_eff


def residual_u(w, x, w_on_x):
    """Residualized players covariate at a given (w, x)."""
    if w_on_x is None or w is None:
        return 0.0
    alpha, beta = w_on_x
    return w - (alpha + beta * x)


def persistence_dilution(refreshes, x_target_ts, last_rate):
    """Dilution factor (<= 1) for the persistence-path median.

    Only used when the model has too few refreshes to fit a trend. The
    unobserved window runs from the last DISTINCT refresh (the data may
    have been frozen since then) to the target. Players are projected to
    grow over that window at a launch rate that spikes at the release and
    decays to a sustained launch-week rate:
        g(u) = LAUNCH_BASE_PER_DAY
               + LAUNCH_SPIKE_PER_DAY * exp(-u / LAUNCH_DECAY_DAYS),
    with u = days since release. The log-growth is the integral of g over
    the window; earners grow at only EARNERS_GROWTH_FRAC of that rate, so
    the rate is multiplied by G ** -(1 - EARNERS_GROWTH_FRAC). Boundary
    trophies (0% / 100%) are not diluted: a 100% trophy stays 100%
    (every new player earns it) and a 0-earner trophy stays ~0.
    Returns 1.0 when no dilution applies.
    """
    if last_rate is None or last_rate <= 0.0 or last_rate >= 100.0:
        return 1.0
    if not refreshes:
        return 1.0
    try:
        last_distinct_ts = parse_ts(refreshes[-1]["at"])
    except Exception:
        return 1.0
    u1 = (last_distinct_ts - RELEASE.timestamp()) / 86400.0
    u2 = (x_target_ts - RELEASE.timestamp()) / 86400.0
    if u2 <= u1:
        return 1.0
    tau = LAUNCH_DECAY_DAYS
    # Sustained growth over the whole window; the spike only counts
    # post-release (u >= 0).
    s1 = max(u1, 0.0)
    growth = (LAUNCH_BASE_PER_DAY * (u2 - u1)
              + LAUNCH_SPIKE_PER_DAY * tau
              * (math.exp(-s1 / tau) - math.exp(-u2 / tau)))
    if growth <= 0.0:
        return 1.0
    factor = math.exp(-(1.0 - EARNERS_GROWTH_FRAC) * growth)
    # Floor: dilution never collapses a rate to zero. Over a long window
    # new players catch up and earn, so the factor saturates; the floor
    # only binds for degenerate long-window persistence cases, never for
    # the short-window launch case this prior is built for.
    return min(1.0, max(0.05, factor))


# --------------------------------------------------------------------------
# Regimes and quantile spreads.
# --------------------------------------------------------------------------

def classify(last_rate, last_earners, b):
    """Regime from the trophy's current rate and fitted slope."""
    if last_rate is None:
        return "unknown"
    if last_earners == 0:
        return "boundary_zero"
    if last_rate >= 100.0:
        return "boundary_full"
    if last_rate > 50.0:
        return "saturated"
    if b is not None and b <= 0.0 and last_rate >= 20.0:
        return "saturated"
    if last_rate >= 5.0:
        return "active"
    return "rare"


def sigmas_for(last_rate, regime):
    """(low_inner, low_outer, high_inner, high_outer) in logit units.

    Mature trophies are narrow; rare / delayed-onset trophies are wide and
    right-skewed; boundary trophies are widest on the side that can move.
    """
    if regime == "boundary_zero":
        return (0.25, 0.40, 0.60, 1.60)
    if regime == "boundary_full":
        return (0.60, 1.50, 0.10, 0.20)
    if regime == "unknown" or last_rate is None:
        return (0.50, 2.00, 0.50, 2.00)
    if last_rate >= 15.0:
        return (0.22, 0.48, 0.22, 0.48)
    if last_rate >= 5.0:
        return (0.30, 0.70, 0.30, 0.70)
    if last_rate >= 2.0:
        return (0.30, 0.50, 0.50, 1.10)
    return (0.30, 0.55, 0.60, 1.50)


def widen_outer(sigmas, n_points):
    """Widen the 5%/95% tails for low-refresh trophies (never the inner
    quartiles). Floor the total 5%-to-95% band at ~0.5 logit units and cap
    the outer sigmas at 2.0."""
    low_in, low_out, high_in, high_out = sigmas
    if n_points < 5:
        factor = 1.0 + 0.6 * (5 - n_points) / 5.0
        low_out *= factor
        high_out *= factor
    low_out = min(low_out, 2.0)
    high_out = min(high_out, 2.0)
    band = 1.645 * (low_out + high_out)
    if band < 0.5:
        scale = 0.5 / band
        low_out *= scale
        high_out *= scale
    return (low_in, low_out, high_in, high_out)


def quantiles_from(y_med, sigmas):
    """Five quantiles (percent) from a median logit and regime sigmas."""
    low_in, low_out, high_in, high_out = sigmas
    side = (low_out, low_in, 0.0, high_in, high_out)
    qs = []
    for z, s in zip(Z, side):
        qs.append(clamp_pct(inv_logit_pct(y_med + z * s)))
    qs[2] = clamp_pct(inv_logit_pct(y_med))
    out = [round4(q) for q in qs]
    # Enforce non-decreasing after rounding.
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


# --------------------------------------------------------------------------
# Backtest: split the refreshes ~60/40, fit early, predict late.
# --------------------------------------------------------------------------

def backtest(data, refreshes, horizon_days, x_target):
    """Pick (damp, shrink, quad) by mean logit-pinball on held-out refreshes.

    data: per-trophy lists of (refresh_index, x, w, y, rate, earners).
    Returns the best setting, or the default when there are too few
    refreshes to split.
    """
    default = (DAMP_GRID[0], SHRINK_GRID[0], QUAD_GRID[0])
    n_ref = len(refreshes)
    if n_ref < 6:
        return default
    cut = max(3, int(0.6 * n_ref))
    if n_ref - cut < 2:
        return default

    n_trophies = len(data)
    # Early points and per-trophy early summary.
    early = []
    early_last_rate = []
    early_last_earners = []
    early_n = []
    for pts in data:
        ep = [p for p in pts if p[0] < cut]
        early.append([(x, w, y) for (_ri, x, w, y, _r, _e) in ep])
        early_n.append(len(ep))
        if ep:
            early_last_rate.append(ep[-1][4])
            early_last_earners.append(ep[-1][5])
        else:
            early_last_rate.append(None)
            early_last_earners.append(None)

    pfit_early = players_fit(refreshes[:cut])
    if pfit_early is None:
        return default
    alpha_e, beta_e, x_early_last, w_early_last = pfit_early
    beta_e = min(max(beta_e, 0.0), 2.0)
    w_on_x_e = (alpha_e, beta_e)

    # Late refreshes, subsampled to at most 20, evenly spaced.
    late_idx = list(range(cut, n_ref))
    if len(late_idx) > 20:
        step = len(late_idx) / 20.0
        late_idx = sorted(set(cut + int(i * step) for i in range(20)))
    late_x = []
    late_days = []
    late_actual = []
    for ri in late_idx:
        read = refreshes[ri]
        lx = log_time(parse_ts(read["at"]))
        late_x.append(lx)
        hours = math.expm1(lx)
        hours_last = math.expm1(x_early_last)
        late_days.append(max(0.0, (hours - hours_last) / 24.0))
        rates = read.get("rates") or []
        row = []
        for t in range(n_trophies):
            r = rates[t] if t < len(rates) else None
            row.append(logit_pct(r) if r is not None else None)
        late_actual.append(row)

    best = None
    best_loss = None
    for quad in QUAD_GRID:
        raw = []
        regimes = []
        for t in range(n_trophies):
            pts = early[t]
            lr = early_last_rate[t]
            fit = None
            if len(pts) >= 3:
                fit = fit_trophy(pts, quad and lr is not None and lr < 5.0,
                                 w_on_x_e)
            raw.append(fit)
            b = fit[1] if fit else None
            regimes.append(classify(lr, early_last_earners[t], b))
        # Regime means of slope (and curvature for quad trophies).
        mean_b = {}
        mean_d = {}
        for reg in set(regimes):
            bs = [raw[t][1] for t in range(n_trophies)
                  if regimes[t] == reg and raw[t] is not None]
            ds = [raw[t][3] for t in range(n_trophies)
                  if regimes[t] == reg and raw[t] is not None and raw[t][5]]
            mean_b[reg] = _mean(bs) if bs else 0.0
            mean_d[reg] = _mean(ds) if ds else 0.0
        # Per-trophy centering stats for the intercept refit (on u, not w).
        centers = []
        for t in range(n_trophies):
            pts = early[t]
            if pts:
                us = [residual_u(w, x, w_on_x_e) for (x, w, _y) in pts]
                centers.append((_mean([p[0] for p in pts]),
                                _mean(us),
                                _mean([p[0] ** 2 for p in pts]),
                                _mean([p[2] for p in pts])))
            else:
                centers.append((0.0, 0.0, 0.0, 0.0))
        # Per-trophy quantile sides (independent of damp / shrink).
        sides = []
        for t in range(n_trophies):
            sg = widen_outer(sigmas_for(early_last_rate[t], regimes[t]),
                             early_n[t])
            sides.append((sg[1], sg[0], 0.0, sg[2], sg[3]))
        # Residualized players covariate at the late times, per damp.
        for damp in DAMP_GRID:
            u_pred = []
            for li, lx in enumerate(late_x):
                deff = damp / (1.0 + DAMP_HORIZON_K * late_days[li])
                w_pred = w_early_last + beta_e * (lx - x_early_last) * deff
                u_pred.append(residual_u(w_pred, lx, w_on_x_e))
            for shrink in SHRINK_GRID:
                # Shrunk coefficients per trophy.
                coef = []
                for t in range(n_trophies):
                    fit = raw[t]
                    if fit is None:
                        coef.append(None)
                        continue
                    a, b, c, d, _uu, uq = fit
                    npts = early_n[t]
                    reg = regimes[t]
                    b_s = (npts * b + shrink * mean_b[reg]) / (npts + shrink)
                    b_s = damp_saturated_slope(b_s, reg)
                    d_s = d
                    if uq:
                        kd = 3.0 * shrink
                        d_s = (npts * d + kd * mean_d[reg]) / (npts + kd)
                    mx, mu, mx2, my = centers[t]
                    a_new = my - b_s * mx - c * mu - d_s * mx2
                    coef.append((a_new, b_s, c, d_s, uq))
                # Score.
                tot = 0.0
                cnt = 0
                for li, lx in enumerate(late_x):
                    up = u_pred[li]
                    lx2 = lx * lx
                    for t in range(n_trophies):
                        y_act = late_actual[li][t]
                        if y_act is None:
                            continue
                        cf = coef[t]
                        if cf is None:
                            if early_last_rate[t] is None:
                                continue
                            y_med = logit_pct(early_last_rate[t])
                        else:
                            a_new, b_s, c, d_s, uq = cf
                            y_med = a_new + b_s * lx + c * up
                            if uq:
                                y_med += d_s * lx2
                        side = sides[t]
                        for tau, z, s in zip(LEVELS, Z, side):
                            d = y_act - (y_med + z * s)
                            tot += 2.0 * max(tau * d, (tau - 1.0) * d)
                            cnt += 1
                if cnt == 0:
                    continue
                loss = tot / cnt
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best = (damp, shrink, quad)
    return best if best is not None else default


# --------------------------------------------------------------------------
# Forecast one case.
# --------------------------------------------------------------------------

def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    try:
        target_ts = parse_ts(case["target"])
        x_target = log_time(target_ts)
    except Exception:
        target_ts = parse_ts(history[-1]["at"]) if history else 0.0
        x_target = log_time(target_ts) if history else 0.0

    refreshes = dedup_refreshes(history)
    data = trophy_points(trophies, refreshes)
    n_trophies = len(trophies)

    # Fallback medians when there is no usable history at all.
    all_last_rates = []
    for pts in data:
        if pts:
            all_last_rates.append(pts[-1][4])
    global_median_rate = sorted(all_last_rates)[len(all_last_rates) // 2] \
        if all_last_rates else 1.0

    damp, shrink, quad = backtest(data, refreshes, horizon_days, x_target)
    pfit = players_fit(refreshes)
    w_target = extrapolate_w(pfit, x_target, horizon_days, damp)
    if w_target is None:
        if refreshes and (refreshes[-1].get("players_tracked") or 0) > 0:
            w_target = math.log(refreshes[-1]["players_tracked"])
        else:
            w_target = 0.0
    w_on_x = (pfit[0], pfit[1]) if pfit is not None else None
    u_target = residual_u(w_target, x_target, w_on_x)

    # Raw full-data fits and regimes.
    raw = []
    regimes = []
    for t in range(n_trophies):
        pts = [(x, w, y) for (_ri, x, w, y, _r, _e) in data[t]]
        last_rate = data[t][-1][4] if data[t] else None
        last_earners = data[t][-1][5] if data[t] else None
        fit = None
        if len(pts) >= 3:
            fit = fit_trophy(pts, quad and last_rate is not None and last_rate < 5.0,
                             w_on_x)
        raw.append(fit)
        b = fit[1] if fit else None
        regimes.append(classify(last_rate, last_earners, b))

    # Regime means for pooling.
    mean_b = {}
    mean_d = {}
    for reg in set(regimes):
        bs = [raw[t][1] for t in range(n_trophies)
              if regimes[t] == reg and raw[t] is not None]
        ds = [raw[t][3] for t in range(n_trophies)
              if regimes[t] == reg and raw[t] is not None and raw[t][5]]
        mean_b[reg] = _mean(bs) if bs else 0.0
        mean_d[reg] = _mean(ds) if ds else 0.0

    rows = []
    for t, trophy in enumerate(trophies):
        pts = data[t]
        npts = len(pts)
        last_rate = pts[-1][4] if pts else None
        last_earners = pts[-1][5] if pts else None
        regime = regimes[t]
        fit = raw[t]

        if fit is None or npts < 3:
            # Persistence with a launch-spike dilution prior and wide,
            # low-side-heavy spreads (see persistence_dilution). With no
            # trend to fit, the last rate is a stale snapshot; by the
            # target, launch influx dilutes it, and the uncertainty is
            # dominated by that dilution, not the regime.
            med_rate = last_rate if last_rate is not None else global_median_rate
            dil = persistence_dilution(refreshes, target_ts, med_rate)
            med_rate = clamp_pct(med_rate * dil)
            y_med = logit_pct(med_rate)
            sg = PERSIST_SIGMAS
            rows.append({"id": trophy["id"],
                         "q": quantiles_from(y_med, sg)})
            continue

        # Shrink slope (and curvature) toward the regime mean, refit the
        # intercept on the residualized covariate so the line stays centred
        # on the data.
        a, b, c, d, _uu, uq = fit
        reg = regime
        b_s = (npts * b + shrink * mean_b[reg]) / (npts + shrink)
        b_s = damp_saturated_slope(b_s, reg)
        d_s = d
        if uq:
            kd = 3.0 * shrink
            d_s = (npts * d + kd * mean_d[reg]) / (npts + kd)
        us = [residual_u(w, x, w_on_x) for (_ri, x, w, _y, _r, _e) in pts]
        # data[t] rows are (r_idx, x, w, y, rate, earners): x is [1], y [3].
        mx = _mean([p[1] for p in pts])
        mu = _mean(us)
        mx2 = _mean([p[1] ** 2 for p in pts])
        my = _mean([p[3] for p in pts])
        a_new = my - b_s * mx - c * mu - d_s * mx2

        y_med = a_new + b_s * x_target + c * u_target
        if uq:
            y_med += d_s * x_target * x_target

        # Boundary correction: the Jeffreys smoothing pulls a rate stuck at
        # 0% or 100% away from the boundary. When the fitted drift is small
        # (the rate is essentially stuck), put the median back on the
        # boundary instead of reporting the smoothing artefact.
        y_last = pts[-1][3]  # y is element [3] of (r_idx, x, w, y, rate, earners)
        drift = y_med - y_last
        if last_rate == 100.0 and drift > -0.5:
            y_med = max(y_med, LOGIT_9999)
        elif last_rate == 0.0 and drift < 0.5:
            y_med = min(y_med, LOGIT_0001)

        sg = widen_outer(sigmas_for(last_rate, regime), npts)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from(y_med, sg)})

    return {"trophies": rows}


def fallback_output(case):
    """A always-valid output: last rate per trophy, widest spread."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    rows = []
    for idx, trophy in enumerate(trophies):
        rate = 1.0
        for read in reversed(history):
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                rate = rates[idx]
                break
        rows.append({"id": trophy["id"],
                     "q": quantiles_from(logit_pct(rate), (0.5, 2.0, 0.5, 2.0))})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            forecast = forecast_case(case)
            # Validate before writing: one row per trophy, finite,
            # 0..100, non-decreasing.
            if len(forecast["trophies"]) != len(case.get("trophies") or []):
                raise ValueError("row count mismatch")
            for row in forecast["trophies"]:
                q = row["q"]
                if len(q) != 5 or not all(isinstance(v, (int, float))
                                          and math.isfinite(v)
                                          and 0.0 <= v <= 100.0 for v in q):
                    raise ValueError("bad quantiles")
                if any(q[i] > q[i + 1] for i in range(4)):
                    raise ValueError("quantiles not non-decreasing")
        except Exception:
            forecast = fallback_output(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
