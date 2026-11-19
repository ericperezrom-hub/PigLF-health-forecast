"""Helper functions for the structure analysis of the PigLF dataset.

Usage in the notebook:
    import sys
    sys.path.insert(0, "../src")
    from utils import (load_data, pig_summary, status_counts, withdrawals_by_pen,
                       plot_pig_timeline, plot_withdrawals_by_pen)
"""

import warnings
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

STATUS_COLORS = {"Complete": "#2a9d8f", "Withdrawn": "#e76f51"}


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV, parsing the Date column."""
    return pd.read_csv(path, parse_dates=["Date"])


def pig_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per pig with its stay, pen and status.

    Columns: Animal_ID, pen (initial pen), first_date, last_date, n_days,
    n_pens, span_days, has_gaps, transferred, status ('Complete' or 'Withdrawn').
    """
    end = df["Date"].max()

    pig = (
        df.sort_values("Date")
        .groupby("Animal_ID")
        .agg(
            pen=("Pen", "first"),
            first_date=("Date", "min"),
            last_date=("Date", "max"),
            n_days=("Date", "nunique"),
            n_pens=("Pen", "nunique"),
        )
        .reset_index()
    )
    pig["span_days"] = (pig["last_date"] - pig["first_date"]).dt.days + 1
    pig["has_gaps"] = pig["n_days"] < pig["span_days"]
    pig["transferred"] = pig["n_pens"] > 1
    pig["status"] = pig["last_date"].lt(end).map({True: "Withdrawn", False: "Complete"})
    return pig


def status_counts(pig: pd.DataFrame) -> pd.DataFrame:
    """Number and percentage of pigs per status."""
    counts = pig["status"].value_counts()
    return pd.DataFrame({"n_pigs": counts, "pct": (100 * counts / counts.sum()).round(1)})


def withdrawals_by_pen(pig: pd.DataFrame) -> pd.DataFrame:
    """Table per pen: complete pigs, withdrawn pigs, total and % withdrawn."""
    t = (
        pig.groupby("pen")["status"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=list(STATUS_COLORS), fill_value=0)
    )
    t["total"] = t.sum(axis=1)
    t["pct_withdrawn"] = (100 * t["Withdrawn"] / t["total"]).round(1)
    return t


def plot_withdrawals_by_pen(pig: pd.DataFrame, save_path: str | None = None):
    """Stacked bars per pen: complete and withdrawn pigs."""
    t = withdrawals_by_pen(pig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = pd.Series(0, index=t.index)
    for status, color in STATUS_COLORS.items():
        ax.bar(t.index.astype(str), t[status], bottom=bottom, color=color, label=status)
        for x, (b, v) in enumerate(zip(bottom, t[status])):
            if v > 0:
                ax.text(x, b + v / 2, int(v), ha="center", va="center", color="white", fontweight="bold")
        bottom = bottom + t[status]

    ax.set_xlabel("Pen")
    ax.set_ylabel("Number of pigs")
    ax.set_title("Complete and withdrawn pigs per pen")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_pig_timeline(pig: pd.DataFrame, by_pen: bool = False, save_path: str | None = None):
    """Gantt-style chart: one bar per pig (first to last day).

    by_pen=False: sorted by status.
    by_pen=True: grouped by pen, with separators and a label for each pen.
    """
    sort_cols = ["pen", "status", "last_date", "first_date"] if by_pen else ["status", "last_date", "first_date"]
    p = pig.sort_values(sort_cols).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 0.12 * len(p) + 2))
    for i, r in p.iterrows():
        ax.barh(
            i,
            (r.last_date - r.first_date).days + 1,
            left=mdates.date2num(r.first_date),
            color=STATUS_COLORS[r.status],
            height=0.8,
        )

    if by_pen:
        ticks, labels = [], []
        for pen, g in p.groupby("pen"):
            ticks.append((g.index.min() + g.index.max()) / 2)
            labels.append(f"Pen {pen}")
            ax.axhline(g.index.max() + 0.5, color="gray", lw=0.6)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    else:
        ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_ylabel(f"Pigs (n={len(p)})")
    ax.xaxis_date()
    ax.set_title("Stay of each pig in the dataset" + (" (by pen)" if by_pen else ""))
    n = p["status"].value_counts()
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in STATUS_COLORS.values()],
        labels=[f"{s} ({n.get(s, 0)})" for s in STATUS_COLORS],
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax

def outlier_summary(s):
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = ((s < lo) | (s > hi)).sum()
    return pd.Series({"min": s.min(), "max": s.max(),
                      "outlier_threshold_inf": lo, "outlier_threshold_sup": hi,
                      "n_outliers": n, "%_outliers": 100 * n / len(s)})

def iqr_fences(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

def is_outlier(s, k=1.5):
    lo, hi = iqr_fences(s.dropna(), k)
    return (s < lo) | (s > hi)

def weekly_outlier_stats(s, data, group="Week"):
    s = s.dropna()
    g = data.loc[s.index, group]                                     # global rule
    week_mild = s.groupby(g).transform(lambda x: is_outlier(x, 1.5))     # per-week rule
    return pd.Series({
        "n": len(s),
        "n_weekly_outliers": week_mild.sum(),
        "pct_weekly_mild": 100 * week_mild.mean(),
    })

def weekly_breakdown(data, col, group="Week", k=1.5):
    rows = []
    for w, s in data.groupby(group)[col]:
        s = s.dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_low = int((s < lo).sum())
        n_high = int((s > hi).sum())
        rows.append({
            group: w, "n": len(s), "Q1": q1, "Q3": q3,
            "lower_fence": lo, "upper_fence": hi,
            "n_low": n_low, "n_high": n_high,
            "n_outliers": n_low + n_high,
            "pct_outliers": 100 * (n_low + n_high) / len(s),
        })
    return pd.DataFrame(rows).set_index(group)

def outlier_flags(data, cols, k=1.5):
    q1, q3 = data[cols].quantile(0.25), data[cols].quantile(0.75)
    iqr = q3 - q1
    return (data[cols] < q1 - k * iqr) | (data[cols] > q3 + k * iqr)


def animal_outlier_stats(data, cols, id_col="Animal_ID"):
    flags = outlier_flags(data, cols)
    by = flags.groupby(data[id_col])

    out = by.mean().mul(100)                                   # % outliers per variable
    out["pct_days_with_outlier"] = flags.any(axis=1).groupby(data[id_col]).mean() * 100
    out["outliers_per_100_obs"] = by.sum().sum(axis=1) / (by.size() * len(cols)) * 100
    return out.sort_values("pct_days_with_outlier", ascending=False)

def group_outlier_stats(data, cols, group_col):
    flags = outlier_flags(data, cols)
    by = flags.groupby(data[group_col])

    out = by.mean().mul(100)                                 
    out["pct_days_with_outlier"] = flags.any(axis=1).groupby(data[group_col]).mean() * 100
    out["outliers_per_100_obs"] = by.sum().sum(axis=1) / (by.size() * len(cols)) * 100
    return out.sort_values("pct_days_with_outlier", ascending=False)

def plot_time_series(x, y, title=None, color="tab:blue", ylim=None,
                     xlabel="Date", ylabel=None, label=None, marker="o",
                     ax=None, figsize=(12, 4)):
    """Plot a time series with one point per day.

    x: dates (e.g. series.index). y: values. Missing days are not
    interpolated: the line is broken where there is a gap.
    ylim: tuple (min, max) for the y axis, e.g. (0, 100). None = automatic.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    s = pd.Series(list(y), index=pd.to_datetime(list(x))).sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))

    ax.plot(s.index, s.values, color=color, marker=marker, markersize=3,
            label=label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if label is not None:
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else getattr(y, "name", None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return ax

import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_time_series(x, y, title=None, color="tab:blue", ylim=None,
                     xlabel="Date", ylabel=None, label=None, marker="o",
                     ax=None, figsize=(12, 4)):
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=figsize)

    s = pd.Series(list(y), index=pd.to_datetime(list(x))).sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))

    ax.plot(s.index, s.values, color=color, marker=marker, markersize=3,
            label=label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if label is not None:
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else getattr(y, "name", None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(alpha=0.3)
    if own_fig:
        plt.tight_layout()
    return ax


def is_pen_level(df, col, pen_col="Pen", date_col="Date"):
    return bool((df.groupby([pen_col, date_col])[col].nunique(dropna=False)
                 <= 1).all())



def plot_time_series(x, y, title=None, color="tab:blue", ylim=None,
                     xlabel="Date", ylabel=None, label=None, marker="o",
                     ax=None, figsize=(12, 4), xlim=None, show_missing=True,
                     missing_color="tab:red", missing_alpha=0.18):
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=figsize)

    s = pd.Series(list(y), index=pd.to_datetime(list(x))).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if xlim is None:
        start, end = s.index.min(), s.index.max()
    else:
        start, end = pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1])
    s = s.reindex(pd.date_range(start, end, freq="D"))

    ax.plot(s.index, s.values, color=color, marker=marker, markersize=3,
            label=label)

    if show_missing:
        missing = s.isna().to_numpy()
        half = pd.Timedelta(hours=12)
        i, n = 0, len(s)
        while i < n:
            if missing[i]:
                j = i
                while j + 1 < n and missing[j + 1]:
                    j += 1
                ax.axvspan(s.index[i] - half, s.index[j] + half,
                           color=missing_color, alpha=missing_alpha,
                           linewidth=0)
                i = j + 1
            else:
                i += 1

    ax.set_xlim(start - pd.Timedelta(hours=12), end + pd.Timedelta(hours=12))
    if ylim is not None:
        ax.set_ylim(*ylim)
    if label is not None:
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else getattr(y, "name", None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(alpha=0.3)
    if own_fig:
        plt.tight_layout()
    return ax


def is_pen_level(df, col, pen_col="Pen", date_col="Date"):
    return bool((df.groupby([pen_col, date_col])[col].nunique(dropna=False)
                 <= 1).all())


def plot_2D_grid(df, col, level="auto", ylim=None, color="tab:blue",
                 date_col="Date", pen_col="Pen", id_col="Animal_ID",
                 ncols=None, cell_size=(3.6, 2.6), sharey=True):
    if level == "auto":
        level = "pen" if is_pen_level(df, col, pen_col, date_col) else "animal"
    if level not in ("pen", "animal"):
        raise ValueError("level must be 'auto', 'pen' or 'animal'")

    if ylim is None and sharey:
        lo, hi = df[col].min(), df[col].max()
        pad = 0.05 * (hi - lo if hi > lo else 1)
        ylim = (lo - pad, hi + pad)

    def make_grid(n, suptitle):
        nc = ncols or (5 if level == "pen" else 4)
        nc = min(nc, n)
        nr = math.ceil(n / nc)
        fig, axes = plt.subplots(nr, nc, figsize=(cell_size[0] * nc,
                                                  cell_size[1] * nr),
                                 squeeze=False)
        fig.suptitle(suptitle, fontsize=13)
        flat = axes.ravel()
        for extra in flat[n:]:
            extra.axis("off")
        return fig, flat

    xlim = (df[date_col].min(), df[date_col].max())
    missing_patch = Patch(color="tab:red", alpha=0.18, label="Missing data")

    figs = []
    pens = sorted(df[pen_col].unique())

    if level == "pen":
        fig, axes = make_grid(len(pens), f"{col} per pen")
        for ax, pen in zip(axes, pens):
            s = df[df[pen_col] == pen].groupby(date_col)[col].mean()
            plot_time_series(s.index, s, title=f"Pen {pen}", color=color,
                             ylim=ylim, xlim=xlim, xlabel="", ylabel="",
                             ax=ax)
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.legend(handles=[missing_patch], loc="upper right")
        figs.append(fig)
    else:
        for pen in pens:
            sub = df[df[pen_col] == pen]
            ids = sorted(sub[id_col].unique())
            fig, axes = make_grid(len(ids), f"{col} - Pen {pen} "
                                            f"({len(ids)} pigs)")
            for ax, a in zip(axes, ids):
                s = sub[sub[id_col] == a].set_index(date_col)[col]
                plot_time_series(s.index, s, title=str(a), color=color,
                                 ylim=ylim, xlim=xlim, xlabel="", ylabel="",
                             ax=ax)
                ax.tick_params(labelsize=7)
            fig.tight_layout()
            fig.legend(handles=[missing_patch], loc="upper right")
            figs.append(fig)
    return figs

def summary_stats(df, cols):
    d = df[cols]
    out = d.describe().T            
    out["var"] = d.var()
    out["skew"] = d.skew()
    out["kurt"] = d.kurt()
    out["missing_%"] = d.isna().mean() * 100
    return out[["count", "missing_%", "mean", "std", "var",
                "min", "25%", "50%", "75%", "max", "skew", "kurt"]].round(2)


def _fit_ss(y, factors, n_iter=200, tol=1e-10):
    """Sum of squares explained by an additive model y ~ f1 + f2 + ...
    (each factor = array of group labels). Solved by backfitting, so it
    also works with unbalanced data (missing days, pigs removed...)."""
    yc = np.asarray(y, dtype=float)
    yc = yc - yc.mean()
    if not factors:
        return 0.0
    codes = [pd.factorize(f)[0] for f in factors]
    counts = [np.bincount(c) for c in codes]
    comps = [np.zeros_like(yc) for _ in codes]
    fitted = np.zeros_like(yc)
    for _ in range(n_iter):
        old = fitted
        for k, c in enumerate(codes):
            partial = yc - (fitted - comps[k])      # remove the other factors
            comps[k] = (np.bincount(c, weights=partial) / counts[k])[c]
            fitted = sum(comps)
        if np.max(np.abs(fitted - old)) < tol:
            break
    return float(np.sum(yc ** 2) - np.sum((yc - fitted) ** 2))


def detect_level(df, col, date_col="Date", pen_col="Pen"):
    """'batch' (1 value per day), 'pen' (1 per pen and day) or 'animal'."""
    if (df.groupby(date_col)[col].nunique() <= 1).all():
        return "batch"
    if (df.groupby([pen_col, date_col])[col].nunique() <= 1).all():
        return "pen"
    return "animal"


def decompose_variance(df, col, date_col="Date", pen_col="Pen",
                       id_col="Animal_ID", week_col="Week"):
    """Sequential decomposition of the variance of `col`.

    Order: Week -> day (within week) -> Pen -> Animal (within pen) -> residual.
    Shares add up to 100%. The variable's level is detected automatically
    and the data is de-duplicated accordingly (a pen-level variable is not
    counted once per pig).
    Residual = what is left after removing those effects, i.e. pen/pig
    x time interaction + noise.
    """
    d = df[[c for c in {date_col, pen_col, id_col, week_col, col}
            if c in df.columns]].dropna(subset=[col])
    level = detect_level(d, col, date_col, pen_col)
    if level == "batch":
        d = d.groupby(date_col, as_index=False).agg(
            {col: "mean", **({week_col: "first"} if week_col in d else {})})
    elif level == "pen":
        d = d.groupby([pen_col, date_col], as_index=False).agg(
            {col: "mean", **({week_col: "first"} if week_col in d else {})})

    y = d[col].to_numpy(dtype=float)
    n = len(y)
    total_ss = float(np.sum((y - y.mean()) ** 2))
    total_var = total_ss / n

    # Cumulative models, in order; each component = SS gain over the previous
    steps = []   # (label, factor array to add)
    use_week = week_col in d.columns
    if use_week:
        steps.append(("Time: week (trend)", d[week_col].to_numpy()))
    if level != "batch":
        steps.append(("Time: day (shared by all)", d[date_col].to_numpy()))
    if level in ("pen", "animal"):
        steps.append(("Between pens", d[pen_col].to_numpy()))
    if level == "animal":
        steps.append(("Between pigs (within pen)", d[id_col].to_numpy()))

    factors, prev, rows = [], 0.0, {}
    for label, f in steps:
        factors.append(f)
        ss = _fit_ss(y, factors)
        rows[label] = (ss - prev) / total_ss * 100
        prev = ss
    rest_label = {"batch": "Day-to-day (within week)" if use_week
                  else "Residual",
                  "pen": "Residual (pen x time + noise)",
                  "animal": "Residual (pig x time + noise)"}[level]
    rows[rest_label] = (total_ss - prev) / total_ss * 100
    out = {"level": level, "n_obs": n, "total_var": total_var, **rows}
    return out

def variance_table(df, cols, **kwargs):
    res = pd.DataFrame({c: decompose_variance(df, c, **kwargs)
                        for c in cols}).T
    pct = [c for c in res.columns if c not in ("level", "n_obs", "total_var")]
    res[pct] = res[pct].astype(float).round(1)
    res["Residual %"] = res[[c for c in pct if c.startswith(("Residual",
                                                              "Day-to-day"))]
                            ].bfill(axis=1).iloc[:, 0]
    res = res.drop(columns=[c for c in pct if c.startswith(("Residual",
                                                            "Day-to-day"))])
    res = res.rename(columns={c: c + " %" for c in res.columns
                              if c.startswith(("Time", "Between"))})
    res.insert(2, "total_var", res.pop("total_var").astype(float).round(3))
    return res.fillna("-")


def plot_variance_table(table, figsize=(10, None)):
    comp = [c for c in table.columns if c.endswith("%")]
    t = table[comp].replace("-", 0).astype(float)
    h = figsize[1] or 0.6 * len(t) + 1.5
    ax = t.plot.barh(stacked=True, figsize=(figsize[0], h), width=0.7,
                     colormap="viridis")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of total variance")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=8)
    plt.tight_layout()
    return ax

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _fit_ss(y, factors, n_iter=200, tol=1e-10):
    yc = np.asarray(y, dtype=float)
    yc = yc - yc.mean()
    if not factors:
        return 0.0
    codes = [pd.factorize(f)[0] for f in factors]
    counts = [np.bincount(c) for c in codes]
    comps = [np.zeros_like(yc) for _ in codes]
    fitted = np.zeros_like(yc)
    for _ in range(n_iter):
        old = fitted
        for k, c in enumerate(codes):
            partial = yc - (fitted - comps[k])   
            comps[k] = (np.bincount(c, weights=partial) / counts[k])[c]
            fitted = sum(comps)
        if np.max(np.abs(fitted - old)) < tol:
            break
    return float(np.sum(yc ** 2) - np.sum((yc - fitted) ** 2))


def detect_level(df, col, date_col="Date", pen_col="Pen"):
    if (df.groupby(date_col)[col].nunique() <= 1).all():
        return "batch"
    if (df.groupby([pen_col, date_col])[col].nunique() <= 1).all():
        return "pen"
    return "animal"


def decompose_variance(df, col, date_col="Date", pen_col="Pen",
                       id_col="Animal_ID", week_col="Week"):
    d = df[[c for c in {date_col, pen_col, id_col, week_col, col}
            if c in df.columns]].dropna(subset=[col])
    level = detect_level(d, col, date_col, pen_col)
    if level == "batch":
        d = d.groupby(date_col, as_index=False).agg(
            {col: "mean", **({week_col: "first"} if week_col in d else {})})
    elif level == "pen":
        d = d.groupby([pen_col, date_col], as_index=False).agg(
            {col: "mean", **({week_col: "first"} if week_col in d else {})})

    y = d[col].to_numpy(dtype=float)
    n = len(y)
    total_ss = float(np.sum((y - y.mean()) ** 2))
    total_var = total_ss / n


    steps = []  
    use_week = week_col in d.columns
    if use_week:
        steps.append(("Time: week (trend)", d[week_col].to_numpy()))
    if level != "batch":
        steps.append(("Time: day (shared by all)", d[date_col].to_numpy()))
    if level in ("pen", "animal"):
        steps.append(("Between pens", d[pen_col].to_numpy()))
    if level == "animal":
        steps.append(("Between pigs (within pen)", d[id_col].to_numpy()))

    factors, prev, rows = [], 0.0, {}
    for label, f in steps:
        factors.append(f)
        ss = _fit_ss(y, factors)
        rows[label] = (ss - prev) / total_ss * 100
        prev = ss
    rest_label = {"batch": "Day-to-day (within week)" if use_week
                  else "Residual",
                  "pen": "Residual (pen x time + noise)",
                  "animal": "Residual (pig x time + noise)"}[level]
    rows[rest_label] = (total_ss - prev) / total_ss * 100
    out = {"level": level, "n_obs": n, "total_var": total_var, **rows}
    return out


def variance_table(df, cols, **kwargs):
    res = pd.DataFrame({c: decompose_variance(df, c, **kwargs)
                        for c in cols}).T
    pct = [c for c in res.columns if c not in ("level", "n_obs", "total_var")]
    res[pct] = res[pct].astype(float).round(1)
    res["Residual %"] = res[[c for c in pct if c.startswith(("Residual",
                                                              "Day-to-day"))]
                            ].bfill(axis=1).iloc[:, 0]
    res = res.drop(columns=[c for c in pct if c.startswith(("Residual",
                                                            "Day-to-day"))])
    res = res.rename(columns={c: c + " %" for c in res.columns
                              if c.startswith(("Time", "Between"))})
    res.insert(2, "total_var", res.pop("total_var").astype(float).round(3))
    return res.fillna("-")


def plot_variance_table(table, figsize=(10, None)):
    comp = [c for c in table.columns if c.endswith("%")]
    t = table[comp].replace("-", 0).astype(float)
    h = figsize[1] or 0.6 * len(t) + 1.5
    ax = t.plot.barh(stacked=True, figsize=(figsize[0], h), width=0.7,
                     colormap="viridis")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of total variance")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=8)
    plt.tight_layout()
    return ax


def _residuals(y, factors, n_iter=200, tol=1e-10):
    yc = np.asarray(y, dtype=float)
    yc = yc - yc.mean()
    if not factors:
        return yc
    codes = [pd.factorize(f)[0] for f in factors]
    counts = [np.bincount(c) for c in codes]
    comps = [np.zeros_like(yc) for _ in codes]
    fitted = np.zeros_like(yc)
    for _ in range(n_iter):
        old = fitted
        for k, c in enumerate(codes):
            partial = yc - (fitted - comps[k])
            comps[k] = (np.bincount(c, weights=partial) / counts[k])[c]
            fitted = sum(comps)
        if np.max(np.abs(fitted - old)) < tol:
            break
    return yc - fitted


def residual_autocorr(df, cols, lags=(1, 2, 3, 7), date_col="Date",
                      pen_col="Pen", id_col="Animal_ID", week_col="Week"):
    rows = {}
    for col in cols:
        d = df[[c for c in {date_col, pen_col, id_col, week_col, col}
                if c in df.columns]].dropna(subset=[col])
        level = detect_level(d, col, date_col, pen_col)
        agg = {col: "mean", **({week_col: "first"} if week_col in d else {})}
        if level == "batch":
            d = d.groupby(date_col, as_index=False).agg(agg)
            d["_unit"] = 0
            factors = [d[week_col]] if week_col in d else []
        elif level == "pen":
            d = d.groupby([pen_col, date_col], as_index=False).agg(agg)
            d["_unit"] = d[pen_col]
            factors = ([d[week_col]] if week_col in d else []) + \
                      [d[date_col], d["_unit"]]
        else:
            d["_unit"] = d[id_col]
            factors = ([d[week_col]] if week_col in d else []) + \
                      [d[date_col], d["_unit"]]
        d = d.assign(_r=_residuals(d[col].to_numpy(),
                                   [f.to_numpy() for f in factors]))
        wide = d.pivot(index=date_col, columns="_unit", values="_r")
        wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max()))
        out = {"level": level}
        for k in lags:
            a, b = wide.iloc[:-k].to_numpy().ravel(), wide.iloc[k:].to_numpy().ravel()
            ok = ~np.isnan(a) & ~np.isnan(b)
            out[f"lag{k}"] = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 \
                else np.nan
        rows[col] = out
    res = pd.DataFrame(rows).T
    res[[c for c in res.columns if c != "level"]] = \
        res[[c for c in res.columns if c != "level"]].astype(float).round(2)
    return res

def _rsd(x):
    """Robust standard deviation (IQR / 1.349): not inflated by outliers."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return np.nan
    q1, q3 = np.percentile(x, [25, 75])
    return (q3 - q1) / 1.349


def weekly_variability(df, col, date_col="Date", pen_col="Pen",
                       id_col="Animal_ID", week_col="Week"):
    """Weekly table of level and spread for one variable.

    level   : weekly median of the variable.
    between : spread ACROSS units (pigs, or pens for pen-level variables)
              each day, averaged over the days of the week.
              (NaN for batch-level variables: 1 value per day.)
    within  : day-to-day spread INSIDE each unit during the week
              (median over units).
    All spreads are robust (IQR/1.349).
    """
    d = df[[c for c in {date_col, pen_col, id_col, week_col, col}
            if c in df.columns]].dropna(subset=[col])
    level = detect_level(d, col, date_col, pen_col)
    if level == "batch":
        d = d.groupby(date_col, as_index=False).agg(
            {col: "mean", week_col: "first"})
        d["_u"] = 0
    elif level == "pen":
        d = d.groupby([pen_col, date_col], as_index=False).agg(
            {col: "mean", week_col: "first"})
        d["_u"] = d[pen_col]
    else:
        d["_u"] = d[id_col]

    out = pd.DataFrame({"level": d.groupby(week_col)[col].median()})
    if level != "batch":
        daily = d.groupby([week_col, date_col])[col].agg(_rsd)
        out["between"] = daily.groupby(week_col).mean()
    else:
        out["between"] = np.nan
    w = d.groupby(["_u", week_col])[col].agg(_rsd)
    out["within"] = w.groupby(week_col).median()
    return out


def variability_summary(df, cols, **kwargs):
    """One row per variable: how much does the spread change over time?

    ratio_* : mean spread in the last third of weeks / first third
              (1 = stable, >1 grows, <1 shrinks).
    taylor  : slope of log(within spread) vs log(weekly level).
              ~0 constant variance; ~1 spread proportional to the level
              (constant CV, a log transform would stabilise it).
    """
    rows = {}
    for c in cols:
        t = weekly_variability(df, c, **kwargs)
        k = max(len(t) // 3, 1)

        def ratio(s):
            a, b = s.iloc[:k].mean(), s.iloc[-k:].mean()
            return b / a if a and a > 0 else np.nan

        ok = (t["level"] > 0) & (t["within"] > 0)
        slope = (np.polyfit(np.log(t.loc[ok, "level"]),
                            np.log(t.loc[ok, "within"]), 1)[0]
                 if ok.sum() > 3 else np.nan)
        rows[c] = {"ratio_between": ratio(t["between"]),
                   "ratio_within": ratio(t["within"]),
                   "taylor": slope}
    return pd.DataFrame(rows).T.astype(float).round(2)


def plot_variability(df, cols, ncols=3, **kwargs):
    """Grid: weekly spread across units and within units, per variable."""
    n = len(cols)
    nr = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nr, ncols, figsize=(4.2 * ncols, 2.8 * nr),
                             squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        t = weekly_variability(df, c, **kwargs)
        ax.plot(t.index, t["between"], marker="o", ms=3,
                label="between units")
        ax.plot(t.index, t["within"], marker="o", ms=3,
                label="within unit (day-to-day)")
        ax.set_ylim(bottom=0)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("Week", fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7, loc="best")
    fig.suptitle("Weekly spread (robust SD)")
    fig.tight_layout()
    return fig

TESTS = ["trend", "seasonality", "stationarity", "age_dependence"]

def mann_kendall(x, correct=True):
    """Mann-Kendall trend test.

    correct=True applies the Hamed & Rao (1998) modification: the variance
    of S is inflated according to the significant autocorrelations of the
    (detrended) ranks, so autocorrelated series are not flagged as trending
    just because neighbouring days look alike.
    Returns (p_value, sen_slope_per_step).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = np.triu(np.sign(x[None, :] - x[:, None]), 1).sum()
    _, cnt = np.unique(x, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5)
             - np.sum(cnt * (cnt - 1) * (2 * cnt + 5))) / 18
    i, j = np.triu_indices(n, 1)
    sen = np.median((x[j] - x[i]) / (j - i))
    factor = 1.0
    if correct:
        ranks = stats.rankdata(x - sen * np.arange(n))
        r = acf(ranks, nlags=n - 1, fft=True)[1:]
        k = np.arange(1, n)
        sig = np.abs(r) > 1.96 / np.sqrt(n)     # 95% band, as in common
                                                # implementations (pymannkendall)
        factor = 1 + 2 / (n * (n - 1) * (n - 2)) * np.sum(
            ((n - k) * (n - k - 1) * (n - k - 2) * r)[sig])
    factor = max(factor, 0.1)   # strong negative autocorrelation could make
                                # the corrected variance <= 0
    z = (s - np.sign(s)) / np.sqrt(var_s * factor)
    return 2 * (1 - stats.norm.cdf(abs(z))), sen


def seasonality_test(y, period=7):
    """Seasonality of a given period in a regular daily series.

    Removes the slow component with a centred moving average of one period
    (this cancels the seasonal cycle itself, so what is left = seasonal +
    noise) and tests whether the mean differs by position in the cycle
    (e.g. day of the week). The test is an F-test on the cycle dummies with
    HAC (Newey-West) errors, so autocorrelation does not inflate it.
    Returns (p_value, strength in [0, 1]).
    """
    ma = y.rolling(period, center=True).mean()
    if period % 2 == 0:
        ma = ma.rolling(2).mean().shift(-1)
    d = (y - ma).dropna()
    if len(d) < 3 * period or d.std() == 0:
        return np.nan, np.nan
    phase = pd.Series(((d.index - y.index[0]).days % period), index=d.index)
    X = sm.add_constant(pd.get_dummies(phase, drop_first=True).astype(float))
    fit = sm.OLS(d.to_numpy(), X.to_numpy()).fit(
        cov_type="HAC", cov_kwds={"maxlags": period})
    R = np.eye(X.shape[1])[1:]
    p = float(fit.f_test(R).pvalue)
    strength = max(0.0, 1 - np.var(fit.resid) / np.var(d.to_numpy()))
    return p, strength


def stationarity_tests(x):
    """ADF (H0: non-stationary) and KPSS (H0: stationary), level version."""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in (lambda: adfuller(x, autolag="AIC")[1],
                  lambda: kpss(x, regression="c", nlags="auto")[1]):
            try:
                out.append(float(f()))
            except Exception:
                out.append(np.nan)
    return out[0], out[1]


def age_dependence_test(x, age):
    """Spearman correlation between the series and age, with a p-value
    corrected for autocorrelation (effective sample size, Bretherton 1999).
    Returns (rho, p_value)."""
    x, age = np.asarray(x, float), np.asarray(age, float)
    n = len(x)
    if n < 10 or np.std(age) == 0 or np.std(x) == 0:
        return np.nan, np.nan
    rho = stats.spearmanr(x, age)[0]
    r1x = acf(stats.rankdata(x), nlags=1, fft=True)[1]
    r1y = acf(stats.rankdata(age), nlags=1, fft=True)[1]
    n_eff = float(np.clip(n * (1 - r1x * r1y) / (1 + r1x * r1y), 4, n))
    t = rho * np.sqrt((n_eff - 2) / max(1 - rho ** 2, 1e-12))
    return rho, 2 * stats.t.sf(abs(t), n_eff - 2)


# ------------------------------------------------------- series handling
def _prepare(s, max_gap=3, min_obs=30):
    """Regular daily series: gaps <= max_gap days are interpolated, longer
    gaps split the series and the longest continuous stretch is kept."""
    s = s.dropna().sort_index()
    if s.empty:
        return None
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    isn = full.isna()
    run = (isn != isn.shift()).cumsum()
    long_gap = isn & (isn.groupby(run).transform("sum") > max_gap)
    full = full.interpolate(limit_area="inside")
    full[long_gap] = np.nan
    ok = full.notna()
    seg_id = (~ok).cumsum()[ok]
    if seg_id.empty:
        return None
    best = seg_id.value_counts().idxmax()
    seg = full[ok][seg_id == best]
    return seg if len(seg) >= min_obs else None


def _series_tests(s, age_s, alpha, period, max_gap, min_obs):
    seg = _prepare(s, max_gap, min_obs)
    nan = dict(n_obs=0, trend_p=np.nan, trend_p_raw=np.nan, slope=np.nan,
               season_p=np.nan, season_strength=np.nan, adf_p=np.nan,
               kpss_p=np.nan, age_rho=np.nan, age_p=np.nan)
    if seg is None or seg.std() == 0:
        return {**nan, "trend": np.nan, "seasonality": np.nan,
                "stationarity": np.nan, "age_dependence": np.nan}
    x = seg.to_numpy()
    p_corr, sen = mann_kendall(x, correct=True)
    p_raw, _ = mann_kendall(x, correct=False)
    sp, sstr = seasonality_test(seg, period)
    adf_p, kpss_p = stationarity_tests(x)
    if age_s is not None:
        age = age_s.reindex(seg.index).interpolate(limit_area=None) \
                   .bfill().ffill()
        rho, ap = age_dependence_test(x, age.to_numpy())
    else:
        rho, ap = np.nan, np.nan
    res = dict(n_obs=len(x), trend_p=p_corr, trend_p_raw=p_raw, slope=sen,
               season_p=sp, season_strength=sstr, adf_p=adf_p, kpss_p=kpss_p,
               age_rho=rho, age_p=ap)
    flag = lambda c: np.nan if pd.isna(c) else float(c)
    res["trend"] = flag(p_corr < alpha)
    res["seasonality"] = flag(sp < alpha) if not pd.isna(sp) else np.nan
    res["stationarity"] = (np.nan if pd.isna(adf_p) or pd.isna(kpss_p)
                           else float(adf_p < alpha and kpss_p >= alpha))
    res["age_dependence"] = flag(ap < alpha) if not pd.isna(ap) else np.nan
    return res


# ------------------------------------------------------------- panel API
def temporal_properties(df, col, alpha=0.05, period=7, age_col="Fattening_day",
                        date_col="Date", pen_col="Pen", id_col="Animal_ID",
                        min_obs=30, max_gap=3, clip_quantiles=None):
    """Trend, seasonality, stationarity and age dependence of a variable.

    Tests (p-value < alpha means the property is present, except for
    stationarity):
      trend          Mann-Kendall with Hamed-Rao autocorrelation correction
                     (trend_p_raw = uncorrected, for comparison).
      seasonality    weekly (period) cycle, F-test with HAC errors.
      stationarity   ADF rejects non-stationarity (p < alpha) AND KPSS does
                     not reject stationarity (p >= alpha).
      age_dependence Spearman vs age_col, autocorrelation-corrected p-value.
                     NB: if all animals enter on the same day (as in this
                     batch), age and calendar time coincide and this is
                     the same question as the trend.

    Series: batch-level variable -> 1 series; pen-level -> 1 per pen;
    pig-level -> 1 per pig. The "mean series" is the daily mean over all
    pens / pigs. For pig-level variables the tests are also run per pen.

    Returns dict:
      level, per_series (p-values and flags per pen/pig), summary
      (% of series with the property + mean-series result) and, for
      pig-level variables, by_pen (% of pigs per pen + pen-mean series).
    """
    d = df.copy()
    if clip_quantiles:
        lo, hi = d[col].quantile(clip_quantiles)
        d[col] = d[col].clip(lo, hi)
    level = detect_level(d.dropna(subset=[col]), col, date_col, pen_col)
    has_age = age_col in d.columns
    cols = {date_col, pen_col, col} | ({id_col} if level == "animal" else set())
    if has_age:
        cols.add(age_col)
    d = d[[c for c in cols if c in d.columns]].dropna(subset=[col])

    def age_of(sub):
        return sub.groupby(date_col)[age_col].mean() if has_age else None

    def run(sub):
        return _series_tests(sub.groupby(date_col)[col].mean(), age_of(sub),
                             alpha, period, max_gap, min_obs)

    if level == "batch":
        units, key = {"batch": d}, None
    elif level == "pen":
        units, key = {p: g for p, g in d.groupby(pen_col)}, pen_col
    else:
        units, key = {a: g for a, g in d.groupby(id_col)}, id_col
    per_series = pd.DataFrame({u: run(g) for u, g in units.items()}).T
    per_series.index.name = key or "series"
    if level == "animal":
        pen_of = d.drop_duplicates(id_col).set_index(id_col)[pen_col]
        per_series.insert(0, pen_col, pen_of.reindex(per_series.index))

    mean_res = run(d)   # daily mean over all pens / pigs
    rows = {}
    for t, pcol in zip(TESTS, ["trend_p", "season_p", None, "age_p"]):
        flags = per_series[t].dropna()
        rows[t] = {"n_series": len(flags),
                   "pct_series": round(100 * flags.mean(), 1)
                   if len(flags) else np.nan,
                   "mean_series_pass": mean_res[t],
                   "mean_series_p": mean_res[pcol] if pcol else np.nan}
    summary = pd.DataFrame(rows).T
    summary.loc["stationarity", "mean_series_p"] = np.nan  # two p-values
    # Sen slope of the mean series (units of the variable per day)
    summary["mean_slope_per_day"] = np.nan
    summary.loc["trend", "mean_slope_per_day"] = mean_res["slope"]
    out = {"level": level, "per_series": per_series.round(4),
           "summary": summary, "mean_series": mean_res}

    if level == "animal":
        parts = {}
        for pen, g in d.groupby(pen_col):
            ps = per_series[per_series[pen_col] == pen]
            pen_mean = run(g)
            row = {}
            for t in TESTS:
                row[(t, "% pigs")] = round(100 * ps[t].dropna().mean(), 1)
                row[(t, "pen mean")] = pen_mean[t]
            parts[pen] = row
        by_pen = pd.DataFrame(parts).T
        by_pen.columns = pd.MultiIndex.from_tuples(by_pen.columns)
        out["by_pen"] = by_pen
    return out


def temporal_properties_table(df, cols, **kwargs):
    rows = {}
    for c in cols:
        r = temporal_properties(df, c, **kwargs)
        s = r["summary"]
        row = {"level": r["level"], "n_series": int(s["n_series"].max())}
        for t in TESTS:
            row[f"{t} %"] = s.loc[t, "pct_series"]
            row[f"{t} (mean)"] = s.loc[t, "mean_series_pass"]
            if t == "trend":
                row["mean slope/day"] = s.loc[t, "mean_slope_per_day"]
        rows[c] = row
    return pd.DataFrame(rows).T

def acf_pacf_panel(df, col, nlags=14, diff=False, date_col="Date",
                   pen_col="Pen", id_col="Animal_ID", min_obs=30, max_gap=3,
                   lb_lags=10):
    d = df[[c for c in {date_col, pen_col, id_col, col} if c in df.columns]]
    d = d.dropna(subset=[col])
    level = detect_level(d, col, date_col, pen_col)
    if level == "batch":
        units = {"batch": d.groupby(date_col)[col].mean()}
    elif level == "pen":
        units = {p: g.groupby(date_col)[col].mean()
                 for p, g in d.groupby(pen_col)}
    else:
        units = {a: g.set_index(date_col)[col]
                 for a, g in d.groupby(id_col)}

    A, P, N, LB = {}, {}, {}, {}
    for u, s in units.items():
        seg = _prepare(s, max_gap, min_obs)
        if seg is None:
            continue
        if diff:
            seg = seg.diff().dropna()
        x = seg.to_numpy()
        if len(x) < 2 * nlags + 2 or np.std(x) == 0:
            continue
        A[u] = acf(x, nlags=nlags, fft=True)[1:]
        P[u] = pacf(x, nlags=nlags, method="ywm")[1:]
        N[u] = len(x)
        LB[u] = float(acorr_ljungbox(x, lags=[lb_lags])["lb_pvalue"].iloc[0])
    lags = range(1, nlags + 1)
    acf_df = pd.DataFrame(A, index=lags).T
    pacf_df = pd.DataFrame(P, index=lags).T
    n = pd.Series(N)
    band = 1.96 / np.sqrt(n)
    out_acf = acf_df.abs().gt(band, axis=0)
    out_pacf = pacf_df.abs().gt(band, axis=0)
    summary = pd.DataFrame({
        "acf_median": acf_df.median(), "acf_%_sig": 100 * out_acf.mean(),
        "pacf_median": pacf_df.median(), "pacf_%_sig": 100 * out_pacf.mean()
    }).round(2)
    summary.index.name = "lag"
    lb = pd.Series(LB)
    return {"level": level, "acf": acf_df, "pacf": pacf_df, "n": n,
            "band": band, "lb_p": lb, "diff": diff, "summary": summary,
            "pct_dependent": round(100 * (lb < 0.05).mean(), 1)}


def plot_acf_pacf(res, title=None, figsize=(12, 3.6), show_units=True):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    bound = float(res["band"].median())
    for ax, key, name in zip(axes, ["acf", "pacf"], ["ACF", "PACF"]):
        m = res[key]
        lags = m.columns.to_numpy()
        if len(m) == 1:
            ax.bar(lags, m.iloc[0].to_numpy(), width=0.4, color="tab:blue")
        else:
            if show_units:
                ax.plot(lags, m.T.to_numpy(), color="gray", alpha=0.12, lw=0.8)
            ax.fill_between(lags, m.quantile(0.25), m.quantile(0.75),
                            color="tab:blue", alpha=0.25, label="IQR")
            ax.plot(lags, m.median(), color="tab:blue", marker="o", ms=3,
                    label="median")
            ax.legend(fontsize=8, loc="upper right")
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(bound, color="r", ls="--", lw=0.8)
        ax.axhline(-bound, color="r", ls="--", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("Lag (days)")
        ax.grid(alpha=0.3)
    suffix = " (1st difference)" if res["diff"] else ""
    fig.suptitle((title or "") + suffix +
                 f"  |  {len(res['acf'])} series, "
                 f"Ljung-Box dependent: {res['pct_dependent']}%", fontsize=11)
    fig.tight_layout()
    return fig

def detrended_wide(df, col, id_col="Animal_ID", date_col="Date", deg=1, min_obs=30):
    w = df.pivot(index=date_col, columns=id_col, values=col)
    x = np.arange(len(w)); out = {}
    for a in w:
        s = w[a]; m = s.notna().to_numpy()
        if m.sum() < min_obs: continue
        b = np.polyfit(x[m], s[m], deg); out[a] = s - np.polyval(b, x)
    return pd.DataFrame(out)

def synchrony(df, col, id_col="Animal_ID", pen_col="Pen", **kw):
    W = detrended_wide(df, col, id_col=id_col, **kw)
    pen_of = df.drop_duplicates(id_col).set_index(id_col)[pen_col]
    C = W.corr(min_periods=30)
    same = pen_of.reindex(C.index).to_numpy()
    same = same[:, None] == same[None, :]
    iu = np.triu_indices(len(C), 1); v, s = C.values[iu], same[iu]
    r2 = [pd.concat([W[a], W.drop(columns=a).mean(axis=1)], axis=1).dropna().corr().iloc[0, 1] ** 2
          for a in W]
    Z = ((W - W.mean()) / W.std()).fillna(0)
    ev = np.linalg.svd(Z.values, compute_uv=False) ** 2; ev /= ev.sum()
    return {"corr same pen": np.nanmean(v[s]), "corr diff pen": np.nanmean(v[~s]),
            "median R² vs farm": np.median(r2), "PC1 var share": ev[0]}