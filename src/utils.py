"""Helper functions for the structure analysis of the PigLF dataset.

Usage in the notebook:
    import sys
    sys.path.insert(0, "../src")
    from utils import (load_data, pig_summary, status_counts, withdrawals_by_pen,
                       plot_pig_timeline, plot_withdrawals_by_pen)
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

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
    """Outlier rates per group (animal, pen...), sorted by % of days with any outlier."""
    flags = outlier_flags(data, cols)
    by = flags.groupby(data[group_col])

    out = by.mean().mul(100)                                   # % outliers per variable
    out["pct_days_with_outlier"] = flags.any(axis=1).groupby(data[group_col]).mean() * 100
    out["outliers_per_100_obs"] = by.sum().sum(axis=1) / (by.size() * len(cols)) * 100
    return out.sort_values("pct_days_with_outlier", ascending=False)