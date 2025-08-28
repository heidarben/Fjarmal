# app.py
# Run in a browser (Streamlit). Upload CSVs, choose dates, click Generate → download PDF.
# Currency is fixed to ISK. Colors/labels match your desktop script.

import io
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader

# --- Visual palette & constants ---
POSITIVE_DELTA_COLOR = "#9fc8c8"   # light minty green (above zero)
NEGATIVE_DELTA_COLOR = "#298c8c"   # deeper teal-green (below zero)
CUMULATIVE_LINE_COLOR = "#ffc000"  # golden yellow
GRID_COLOR = "#d9d9d9"             # soft gridline
ZERO_LINE_COLOR = "#3a3a3a"        # zero axis
CURRENCY = "ISK"

st.set_page_config(page_title="Finances Dashboard", layout="centered")

# ------------- Helpers -------------
def money_formatter():
    def _fmt(x, pos):
        return f"{int(x):,} {CURRENCY}" if abs(x) >= 1 else f"{x:.2f} {CURRENCY}"
    return FuncFormatter(_fmt)

def _coerce_to_date(series, dayfirst=False):
    s = series.astype(str).str.strip()
    d = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=dayfirst)
    mask = d.isna()
    if mask.any():
        d2 = pd.to_datetime(s[mask], format="%Y-%m", errors="coerce", dayfirst=dayfirst)
        d.loc[mask] = d2
    return d

def monthly_from_transactions(income_tx, expense_tx):
    inc = pd.DataFrame(columns=["date","income"])
    exp = pd.DataFrame(columns=["date","expenses"])
    if not income_tx.empty:
        inc = income_tx.copy()
        inc["date"] = inc["date"].dt.to_period("M").dt.to_timestamp()
        inc = inc.groupby("date", as_index=False)["amount"].sum().rename(columns={"amount":"income"})
    if not expense_tx.empty:
        exp = expense_tx.copy()
        exp["date"] = exp["date"].dt.to_period("M").dt.to_timestamp()
        exp = exp.groupby("date", as_index=False)["amount"].sum().rename(columns={"amount":"expenses"})
    df = pd.merge(inc, exp, on="date", how="outer").fillna(0.0)
    return df.sort_values("date")

def compute_monthly_and_deltas(income_tx, expense_tx):
    monthly = monthly_from_transactions(income_tx, expense_tx)
    if monthly.empty:
        return monthly
    monthly["delta"] = monthly["income"] - monthly["expenses"]
    monthly["cumulative"] = monthly["delta"].cumsum()
    return monthly

def filter_transactions_by_range(df, start_dt, end_dt):
    if df.empty or (start_dt is None and end_dt is None):
        return df
    mask = pd.Series(True, index=df.index)
    if start_dt is not None:
        mask &= df["date"] >= pd.Timestamp(start_dt)
    if end_dt is not None:
        mask &= df["date"] <= pd.Timestamp(end_dt)
    return df.loc[mask].copy()

# ------------- Plots (return PNG bytes) -------------
def plot_delta_and_cumulative(monthly) -> bytes:
    df = monthly.copy()
    deltas = df["delta"].to_numpy()
    cum = df["cumulative"].to_numpy()
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=140)

    # Bars
    bar_colors = [POSITIVE_DELTA_COLOR if v >= 0 else NEGATIVE_DELTA_COLOR for v in deltas]
    ax.bar(x, deltas, color=bar_colors, edgecolor="none")

    # Left limits (ensure 0 inside)
    if deltas.size:
        dmin = float(np.nanmin(deltas)); dmax = float(np.nanmax(deltas))
    else:
        dmin = dmax = 0.0
    span = dmax - dmin
    pad = max(span * 0.10, 1.0)
    y0, y1 = (dmin - pad, dmax + pad)
    if y0 >= 0: y0 = -1.0
    if y1 <= 0: y1 =  1.0
    ax.set_ylim(y0, y1)

    # Scale cumulative to left axis (0 aligns), keep within ~85% height
    cum_min = float(np.nanmin(cum)) if cum.size else 0.0
    cum_max = float(np.nanmax(cum)) if cum.size else 0.0
    pos_cap = y1 * 0.85 if y1 > 0 else 1.0
    neg_cap = -y0 * 0.85 if y0 < 0 else 1.0
    scale_pos = pos_cap / cum_max if cum_max > 0 else np.inf
    scale_neg = neg_cap / (-cum_min) if cum_min < 0 else np.inf
    scale = 1.0 if (np.isinf(scale_pos) and np.isinf(scale_neg)) else min(scale_pos, scale_neg)
    cum_scaled = cum * scale

    ax.plot(x, cum_scaled, linewidth=2.8, marker="o", markersize=5.5, markeredgewidth=0, color=CUMULATIVE_LINE_COLOR)
    ax.axhline(0, linewidth=1.2, color=ZERO_LINE_COLOR)

    # Right axis shows TRUE cumulative via inverse transform
    def left_to_true(y_left):  return y_left / scale if scale != 0 else y_left
    def true_to_left(y_true):  return y_true * scale
    try:
        secax = ax.secondary_yaxis('right', functions=(left_to_true, true_to_left))
        secax.set_ylabel(f"Söfnun ({CURRENCY})")
        secax.yaxis.set_major_formatter(money_formatter())
    except AttributeError:
        ax2 = ax.twinx()
        L0, L1 = ax.get_ylim()
        ax2.set_ylim(left_to_true(L0), left_to_true(L1))
        ax2.set_ylabel(f"Söfnun ({CURRENCY})")
        ax2.yaxis.set_major_formatter(money_formatter())

    # X ticks (every 2nd month)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([pd.to_datetime(d).strftime("%b-%y") for d in df["date"].iloc[::2]], rotation=30, ha="right")

    ax.set_title("Þróun söfnunar", pad=14)
    ax.set_ylabel(f"Mánaðarleg Δ ({CURRENCY})")
    ax.yaxis.set_major_formatter(money_formatter())

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def plot_income_vs_expenses(monthly) -> bytes:
    df = monthly.copy()
    x = np.arange(len(df))
    width = 0.6

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=140)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linewidth=1, linestyle="-", color=GRID_COLOR)
    ax.set_axisbelow(True)

    ax.bar(x - width/4, df["income"].values,  width=width/2, label="Innkoma", color=POSITIVE_DELTA_COLOR, edgecolor="none")
    ax.bar(x + width/4, df["expenses"].values, width=width/2, label="Útgjöld", color=NEGATIVE_DELTA_COLOR, edgecolor="none")

    ax.axhline(0, linewidth=1.2, color=ZERO_LINE_COLOR)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([pd.to_datetime(d).strftime("%b-%y") for d in df["date"].iloc[::2]], rotation=30, ha="right")
    ax.set_title("Innkoma vs Útgjöld", pad=14)
    ax.set_ylabel(f"Upphæð ({CURRENCY})")
    ax.legend()
    ax.yaxis.set_major_formatter(money_formatter())

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def plot_income_breakdown(income_tx) -> bytes:
    if income_tx.empty:
        # Nothing to plot
        fig, ax = plt.subplots(figsize=(11, 6.5), dpi=140)
        ax.text(0.5, 0.5, "No income data", ha="center", va="center")
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
        return buf.getvalue()

    priority_order = ["Laun Marel", "Dagpeningar", "Laun Eirr", "Laun TRI", "Hjördís"]
    df = income_tx.copy()
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["source"] = df["source"].apply(lambda s: s if s in priority_order else "Other")
    piv = (
        df.groupby(["date", "source"], as_index=False)["amount"].sum()
          .pivot(index="date", columns="source", values="amount")
          .fillna(0.0).sort_index()
    )
    col_order = [c for c in priority_order if c in piv.columns]
    if "Other" in piv.columns: col_order.append("Other")
    piv = piv[col_order]

    color_map = {
        "Laun Marel": "#1f3b73",
        "Dagpeningar": "#7d7d7d",
        "Laun Eirr": NEGATIVE_DELTA_COLOR,
        "Laun TRI": "#e67e22",
        "Hjördís": POSITIVE_DELTA_COLOR,
        "Other": "#f7c6d0",
    }

    x = np.arange(len(piv.index))
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=140)
    bottom = np.zeros(len(piv.index))
    for col in piv.columns:
        ax.bar(x, piv[col].values, bottom=bottom, label=str(col),
               color=color_map.get(col, "#cccccc"), edgecolor="none")
        bottom += piv[col].values

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linewidth=1, linestyle="-", color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([d.strftime("%b-%y") for d in piv.index[::2]], rotation=30, ha="right")
    ax.set_title("Dreifing Innkomu", pad=14)
    ax.set_ylabel(f"Upphæð ({CURRENCY})")
    ax.legend(ncol=3, fontsize=9)
    ax.yaxis.set_major_formatter(money_formatter())
    ax.axhline(0, linewidth=1.2, color=ZERO_LINE_COLOR)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def build_single_page_pdf(png1: bytes, png2: bytes, png3: bytes) -> bytes:
    # A4 single page, three rows with spacing
    page_w, page_h = A4
    margin = 0.5 * cm
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin
    n = 3
    row_gap = 0.75 * cm
    total_gaps = row_gap * (n - 1)
    each_h = (usable_h - total_gaps) / n

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    y_top = page_h - margin

    for this_png in [png1, png2, png3]:
        ir = ImageReader(io.BytesIO(this_png))
        w_px, h_px = ir.getSize()
        scale = min(usable_w / w_px, each_h / h_px)
        draw_w, draw_h = w_px * scale, h_px * scale
        x = margin + (usable_w - draw_w) / 2
        y = y_top - draw_h
        c.drawImage(ir, x, y, width=draw_w, height=draw_h,
                    preserveAspectRatio=True, mask="auto")
        y_top = y - row_gap

    c.showPage()
    c.save()
    return buf.getvalue()

# ------------- UI -------------
st.title("📊 Finances Dashboard (ISK)")
st.write("Upload your CSVs, set an optional date range, and generate plots + a single-page PDF.")

col1, col2 = st.columns(2)
with col1:
    dayfirst = st.toggle("Dates are D/M/Y (day first)", value=False)
with col2:
    st.caption("Currency is fixed to ISK")

inc_file = st.file_uploader("income_transactions.csv", type=["csv"])
exp_file = st.file_uploader("expense_transactions.csv", type=["csv"])

with st.expander("Optional: Date range (inclusive)"):
    c1, c2 = st.columns(2)
    with c1:
        from_dt = st.date_input("From", value=None)
    with c2:
        to_dt = st.date_input("To", value=None)

gen = st.button("Generate")

if gen:
    if inc_file is None or exp_file is None:
        st.error("Please upload both CSV files.")
        st.stop()

    # --- Load/clean inputs ---
    income_tx = pd.read_csv(inc_file)
    expense_tx = pd.read_csv(exp_file)

    # Validate minimum columns
    for need in ["date", "source", "amount"]:
        if need not in income_tx.columns:
            st.error(f"`income_transactions.csv` must contain columns: date, source, amount")
            st.stop()
    for need in ["date", "category", "amount"]:
        if need not in expense_tx.columns:
            st.error(f"`expense_transactions.csv` must contain columns: date, category, amount")
            st.stop()

    income_tx["date"] = _coerce_to_date(income_tx["date"], dayfirst=dayfirst)
    expense_tx["date"] = _coerce_to_date(expense_tx["date"], dayfirst=dayfirst)
    income_tx["amount"] = pd.to_numeric(income_tx["amount"], errors="coerce").fillna(0.0)
    expense_tx["amount"] = pd.to_numeric(expense_tx["amount"], errors="coerce").fillna(0.0)
    income_tx = income_tx.dropna(subset=["date"])
    expense_tx = expense_tx.dropna(subset=["date"])

    # Date filter (inclusive)
    start = pd.Timestamp(from_dt) if isinstance(from_dt, date) else None
    end   = pd.Timestamp(to_dt)   if isinstance(to_dt, date) else None
    income_tx = filter_transactions_by_range(income_tx, start, end)
    expense_tx = filter_transactions_by_range(expense_tx, start, end)

    # Compute monthly + plots
    monthly = compute_monthly_and_deltas(income_tx, expense_tx)
    if monthly.empty:
        st.warning("No data in the selected period.")
        st.stop()

    p1 = plot_delta_and_cumulative(monthly)
    p2 = plot_income_vs_expenses(monthly)
    p3 = plot_income_breakdown(income_tx)

    st.image(p1, caption="Þróun söfnunar", use_column_width=True)
    st.image(p2, caption="Innkoma vs Útgjöld", use_column_width=True)
    st.image(p3, caption="Dreifing Innkomu", use_column_width=True)

    pdf_bytes = build_single_page_pdf(p1, p2, p3)
    st.download_button("⬇️ Download PDF", data=pdf_bytes,
                       file_name="Fjármálayfirlit.pdf",
                       mime="application/pdf")
