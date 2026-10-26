"""
Real Exchange Rate App
======================
Streamlit app that pulls monthly CPI and exchange-rate data from the IMF
SDMX 3.0 API and computes a bilateral real exchange rate (RER).

Definitions
-----------
E_i(t)  : IMF exchange rate, units of country i's currency per 1 USD
S(t)    : units of Country 2 currency per 1 unit of Country 1 currency
          S(t) = E_2(t) / E_1(t)
P1, P2  : monthly CPI of Country 1 and Country 2
RER(t)  = S(t) * P1(t) / P2(t)
RER*    = mean of RER(t) over valid months in the selected window
Real Index(t) = 100 * RER(t) / RER*

A rise in the Real Index means Country 1's currency has appreciated in real
terms against Country 2's (Country 1 goods got relatively more expensive).
"""

from __future__ import annotations

import io
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
BASE_URL = "https://api.imf.org/external/sdmx/3.0"
AGENCY = "IMF.STA"
CPI_FLOW = "CPI"
ER_FLOW = "ER"
CPI_KEY = "{iso}.CPI._T.IX.M"          # all-items CPI index, monthly
ER_KEY = "{iso}.XDC_USD.PA_RT.M"       # domestic currency per USD, period average
START_MONTH = pd.Period("1976-01", freq="M")
EURO_AREA = "G163"
USA = "USA"
TIMEOUT = 60

USER_AGENT = "real-exchange-rate-app/1.0 (Streamlit; educational use)"
JSON_ACCEPT = "application/vnd.sdmx.data+json;version=2.0.0"
CSV_ACCEPT = "application/vnd.sdmx.data+csv;version=2.0.0"

# Used only if the IMF country list cannot be downloaded.
FALLBACK_COUNTRIES = {
    "USA": "United States", "G163": "Euro area", "JPN": "Japan",
    "GBR": "United Kingdom", "CAN": "Canada", "AUS": "Australia",
    "CHE": "Switzerland", "CHN": "China", "IND": "India", "BRA": "Brazil",
    "MEX": "Mexico", "KOR": "Korea, Republic of", "ZAF": "South Africa",
    "TUR": "Türkiye", "SWE": "Sweden", "NOR": "Norway", "DNK": "Denmark",
    "NZL": "New Zealand", "SGP": "Singapore", "IDN": "Indonesia",
    "THA": "Thailand", "PHL": "Philippines", "MYS": "Malaysia",
    "CHL": "Chile", "COL": "Colombia", "PER": "Peru", "ARG": "Argentina",
    "POL": "Poland", "HUN": "Hungary", "CZE": "Czech Republic",
    "ISR": "Israel", "SAU": "Saudi Arabia", "EGY": "Egypt", "NGA": "Nigeria",
    "DEU": "Germany", "FRA": "France", "ITA": "Italy", "ESP": "Spain",
    "NLD": "Netherlands", "PRT": "Portugal",
}


# --------------------------------------------------------------------------
# Parsing helpers (pure functions, no Streamlit)
# --------------------------------------------------------------------------
def parse_month(value) -> pd.Period | None:
    """Turn IMF period strings like '1976-M01', '1976-01', '1976M1' into a Period."""
    if value is None:
        return None
    s = str(value).strip()
    m = re.match(r"^(\d{4})[-]?M?(\d{1,2})(?:-\d{1,2})?(?:T.*)?$", s)
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        return None
    return pd.Period(year=year, month=month, freq="M")


def _label(obj: dict) -> str:
    """Extract an English name from an SDMX-JSON component value."""
    name = obj.get("name")
    if isinstance(name, dict):
        name = name.get("en") or next(iter(name.values()), None)
    if not name:
        names = obj.get("names")
        if isinstance(names, dict):
            name = names.get("en") or next(iter(names.values()), None)
    return name or obj.get("id") or ""


def _to_float(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if np.isfinite(f) else np.nan


COUNTRY_DIM_IDS = ("COUNTRY", "REF_AREA", "JURISDICTION", "ECONOMY")


def parse_sdmx_json(payload: dict) -> tuple[pd.DataFrame, dict]:
    """Parse an SDMX-JSON (1.0 or 2.0) data message.

    Returns (DataFrame[country, period, value], {country_code: country_name}).
    """
    d = payload.get("data", payload)
    structs = d.get("structures") or ([d["structure"]] if "structure" in d else [])
    if not structs:
        raise ValueError("SDMX-JSON response has no structure section.")
    dims = structs[0].get("dimensions", {})
    series_dims = dims.get("series", []) or []
    obs_dims = dims.get("observation", []) or []

    rows, names = [], {}
    for ds in d.get("dataSets", []) or []:
        # Case 1: series-level keys + observation keys (dimensionAtObservation=TIME_PERIOD)
        for skey, s in (ds.get("series") or {}).items():
            sidx = [int(i) for i in skey.split(":")] if skey else []
            code, cname = None, None
            for pos, dim in enumerate(series_dims):
                if dim.get("id") in COUNTRY_DIM_IDS and pos < len(sidx):
                    val = dim["values"][sidx[pos]]
                    code, cname = val.get("id"), _label(val)
            for okey, ov in (s.get("observations") or {}).items():
                oidx = [int(i) for i in okey.split(":")]
                period = None
                for pos, dim in enumerate(obs_dims):
                    if dim.get("id") in ("TIME_PERIOD", "TIME") and pos < len(oidx):
                        tv = dim["values"][oidx[pos]]
                        period = tv.get("id") or tv.get("value") or tv.get("start")
                    elif dim.get("id") in COUNTRY_DIM_IDS and pos < len(oidx):
                        val = dim["values"][oidx[pos]]
                        code, cname = val.get("id"), _label(val)
                value = ov[0] if isinstance(ov, list) and ov else ov
                rows.append((code, period, _to_float(value)))
            if code:
                names[code] = cname or code
        # Case 2: flat observations (dimensionAtObservation=AllDimensions)
        for okey, ov in (ds.get("observations") or {}).items():
            oidx = [int(i) for i in okey.split(":")]
            code, cname, period = None, None, None
            for pos, dim in enumerate(obs_dims):
                if pos >= len(oidx):
                    continue
                val = dim["values"][oidx[pos]]
                if dim.get("id") in COUNTRY_DIM_IDS:
                    code, cname = val.get("id"), _label(val)
                elif dim.get("id") in ("TIME_PERIOD", "TIME"):
                    period = val.get("id") or val.get("value") or val.get("start")
            value = ov[0] if isinstance(ov, list) and ov else ov
            rows.append((code, period, _to_float(value)))
            if code:
                names[code] = cname or code

    return _tidy(rows), names


def parse_sdmx_csv(text: str) -> tuple[pd.DataFrame, dict]:
    """Parse an SDMX-CSV data message (codes or 'code: label' values)."""
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    col = {c.split(":")[0].strip().upper(): c for c in raw.columns}
    ccol = next((col[c] for c in COUNTRY_DIM_IDS if c in col), None)
    tcol = col.get("TIME_PERIOD") or col.get("TIME")
    vcol = col.get("OBS_VALUE")
    if not (ccol and tcol and vcol):
        raise ValueError("SDMX-CSV response is missing COUNTRY/TIME_PERIOD/OBS_VALUE columns.")
    rows, names = [], {}
    for c, t, v in zip(raw[ccol], raw[tcol], raw[vcol]):
        code, _, label = str(c).partition(":")
        code = code.strip()
        names[code] = label.strip() or names.get(code, code)
        rows.append((code, str(t).split(":")[0].strip(), _to_float(v)))
    return _tidy(rows), names


def _tidy(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["country", "period", "value"])
    df["period"] = df["period"].map(parse_month)
    df = df.dropna(subset=["country", "period"])
    df = df.drop_duplicates(subset=["country", "period"], keep="first")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# IMF API access
# --------------------------------------------------------------------------
def _data_urls(flow: str, key: str) -> list[str]:
    # '+' = latest stable version (SDMX 3.0); '~' and '*' as fallbacks.
    return [f"{BASE_URL}/data/dataflow/{AGENCY}/{flow}/{v}/{key}" for v in ("+", "~", "*")]


def fetch_imf(flow: str, key: str, last_n: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Download one IMF dataflow/key. Tries JSON then CSV, with and without filters."""
    param_sets = []
    base = {"dimensionAtObservation": "TIME_PERIOD", "attributes": "none"}
    if last_n:
        param_sets.append({**base, "lastNObservations": str(last_n)})
    param_sets.append({**base, "c[TIME_PERIOD]": f"ge:{START_MONTH.strftime('%Y-%m')}"})
    param_sets.append(base)
    param_sets.append({})

    errors = []
    for url in _data_urls(flow, key):
        for params in param_sets:
            for accept, parser in ((JSON_ACCEPT, "json"), (CSV_ACCEPT, "csv")):
                try:
                    r = requests.get(
                        url, params=params, timeout=TIMEOUT,
                        headers={"Accept": accept, "User-Agent": USER_AGENT},
                    )
                    if r.status_code == 404:
                        # SDMX uses 404 for "no data for this key": stop trying variants.
                        if "NoResults" in r.text or "No Results" in r.text or "no data" in r.text.lower():
                            return pd.DataFrame(columns=["country", "period", "value"]), {}
                        errors.append(f"{r.status_code} {url}")
                        continue
                    if r.status_code != 200:
                        errors.append(f"{r.status_code} {url} {params}")
                        continue
                    if parser == "json":
                        df, names = parse_sdmx_json(r.json())
                    else:
                        df, names = parse_sdmx_csv(r.text)
                    return df, names
                except (requests.ConnectionError, requests.Timeout) as exc:
                    # The server is unreachable: retrying other variants won't help.
                    raise RuntimeError(f"Cannot reach the IMF API ({type(exc).__name__}).") from exc
                except Exception as exc:  # parse error: try next variant
                    errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError("IMF API request failed. Last errors: " + " | ".join(errors[-3:]))


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_country_list() -> tuple[dict, bool]:
    """Countries (incl. Euro area) that have IMF monthly CPI data.

    Returns ({code: name}, from_imf_flag).
    """
    try:
        cpi_df, cpi_names = fetch_imf(CPI_FLOW, CPI_KEY.format(iso="*"), last_n=1)
        try:
            er_df, er_names = fetch_imf(ER_FLOW, ER_KEY.format(iso="*"), last_n=1)
        except Exception:
            er_df, er_names = pd.DataFrame(columns=["country"]), {}
        cpi_codes = set(cpi_df["country"])
        er_codes = set(er_df["country"]) | {USA}
        codes = cpi_codes & er_codes if len(er_codes) > 1 else cpi_codes
        names = {**er_names, **cpi_names}
        out = {c: names.get(c, c) for c in codes}
        out.setdefault(EURO_AREA, "Euro area")
        out.setdefault(USA, "United States")
        if len(out) > 2:
            return out, True
    except Exception:
        pass
    return dict(FALLBACK_COUNTRIES), False


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_series(flow: str, iso: str) -> pd.Series:
    key = (CPI_KEY if flow == CPI_FLOW else ER_KEY).format(iso=iso)
    df, _ = fetch_imf(flow, key)
    df = df[df["country"] == iso] if not df.empty else df
    s = pd.Series(df["value"].values, index=pd.PeriodIndex(df["period"], freq="M"), dtype=float)
    s = s[~s.index.duplicated()].sort_index()
    return s[s.index >= START_MONTH]


def exchange_rate_per_usd(iso: str, er: pd.Series, like_index: pd.PeriodIndex) -> tuple[pd.Series, str]:
    """Return (rate per USD, note). USD per USD is 1 by definition (not published by the IMF).

    If the IMF has no euro-area aggregate exchange rate, use Germany's rate from
    Jan 1999 onward, which is the euro per USD rate.
    """
    if iso == USA and er.dropna().empty:
        return pd.Series(1.0, index=like_index), "1 by definition (USD)"
    if iso == EURO_AREA and er.dropna().empty:
        try:
            deu = load_series(ER_FLOW, "DEU")
            deu = deu[deu.index >= pd.Period("1999-01", freq="M")]
            if not deu.dropna().empty:
                return deu, ("Euro area aggregate not published; using EUR per USD "
                             "(Germany's series) from Jan 1999: " + coverage(deu))
        except Exception:
            pass
    return er, coverage(er)


# --------------------------------------------------------------------------
# Economics (pure functions)
# --------------------------------------------------------------------------
def build_panel(e1: pd.Series, e2: pd.Series, p1: pd.Series, p2: pd.Series) -> pd.DataFrame:
    """Align the four series monthly from Jan-1976 and compute S and RER."""
    df = pd.concat({"E1": e1, "E2": e2, "P1": p1, "P2": p2}, axis=1).sort_index()
    df = df[df.index >= START_MONTH]
    if not df.empty:  # make the monthly calendar explicit so gaps show as NaN
        full = pd.period_range(df.index.min(), df.index.max(), freq="M")
        df = df.reindex(full)
    for c in ["E1", "E2", "P1", "P2"]:
        df[c] = df[c].where(df[c] > 0)  # non-positive values are invalid
    df["S"] = df["E2"] / df["E1"]               # Country-2 currency per 1 Country-1 currency
    df["RER"] = df["S"] * df["P1"] / df["P2"]
    df["valid"] = np.isfinite(df["RER"])
    return df


def apply_window(panel: pd.DataFrame, start: pd.Period, end: pd.Period) -> tuple[pd.DataFrame, float]:
    """Slice to [start, end], compute RER* over valid months and the Real Index."""
    w = panel.loc[(panel.index >= start) & (panel.index <= end)].copy()
    valid_rer = w.loc[w["valid"], "RER"]
    rer_star = float(valid_rer.mean()) if not valid_rer.empty else np.nan
    w["RER_star"] = rer_star
    w["Real_Index"] = 100.0 * w["RER"] / rer_star
    return w, rer_star


def to_csv(w: pd.DataFrame, c1: str, c2: str) -> bytes:
    out = pd.DataFrame({
        "month": w.index.strftime("%Y-%m"),
        f"E1_{c1}_per_USD": w["E1"].values,
        f"E2_{c2}_per_USD": w["E2"].values,
        f"S_{c2}_per_{c1}": w["S"].values,
        f"P1_CPI_{c1}": w["P1"].values,
        f"P2_CPI_{c2}": w["P2"].values,
        "RER": w["RER"].values,
        "valid_month": w["valid"].values,
        "RER_star": w["RER_star"].values,
        "Real_Index": w["Real_Index"].values,
    })
    return out.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
NEON_CYAN = "#00f0ff"
NEON_PINK = "#ff2a6d"
NEON_YELLOW = "#f9f002"
TEXT = "#d1f7ff"
GRID = "rgba(0, 240, 255, 0.10)"
MONO = "Share Tech Mono, Consolas, monospace"


def _neon(x, y, name, color, hover, showlegend=True):
    """A line with a soft glow: two wide translucent copies under a crisp core."""
    glow = [go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=wd),
                       opacity=op, hoverinfo="skip", showlegend=False, connectgaps=False)
            for wd, op in ((10, 0.08), (5, 0.18))]
    core = go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=1.8),
                      hovertemplate=hover, connectgaps=False, showlegend=showlegend)
    return glow + [core]


def make_chart(w: pd.DataFrame, mode: str, s_label: str, log_nominal: bool) -> go.Figure:
    x = w.index.to_timestamp()
    real = _neon(x, w["Real_Index"], "REAL INDEX (AVG = 100)", NEON_CYAN,
                 "Real Index: %{y:.2f}<extra></extra>")
    nom = _neon(x, w["S"], "NOMINAL S", NEON_PINK,
                "S: %{y:,.4f}<extra></extra>")
    nom_type = "log" if log_nominal else "linear"

    if mode == "Both":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for t in real:
            fig.add_trace(t, secondary_y=False)
        for t in nom:
            fig.add_trace(t, secondary_y=True)
        fig.update_yaxes(title_text="REAL INDEX", secondary_y=False,
                         title_font_color=NEON_CYAN, tickfont_color=NEON_CYAN)
        fig.update_yaxes(title_text=s_label.upper(), secondary_y=True, type=nom_type,
                         title_font_color=NEON_PINK, tickfont_color=NEON_PINK, showgrid=False)
        fig.add_hline(y=100, line_dash="dot", line_color=NEON_YELLOW, line_width=1,
                      opacity=0.7, secondary_y=False)
    elif mode == "Real":
        fig = go.Figure(real)
        fig.update_yaxes(title_text="REAL INDEX (SAMPLE AVG = 100)")
        fig.add_hline(y=100, line_dash="dot", line_color=NEON_YELLOW, line_width=1, opacity=0.7)
    else:
        fig = go.Figure(nom)
        fig.update_yaxes(title_text=s_label.upper(), type=nom_type)

    fig.update_layout(
        template="plotly_dark",
        height=540, hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5, 5, 20, 0.6)",
        font=dict(family=MONO, color=TEXT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(family=MONO, size=12)),
        hoverlabel=dict(bgcolor="#0b0b1e", bordercolor=NEON_CYAN,
                        font=dict(family=MONO, color=TEXT)),
    )
    fig.update_xaxes(rangeslider_visible=False, gridcolor=GRID, zeroline=False,
                     linecolor=NEON_CYAN, showline=True, hoverformat="%b %Y",
                     showspikes=True, spikecolor=NEON_YELLOW, spikethickness=1, spikedash="dot")
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


CYBERPUNK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');

:root {
  --cyan: #00f0ff; --pink: #ff2a6d; --yellow: #f9f002; --violet: #7a04eb;
  --bg: #05050f; --panel: rgba(12, 12, 32, 0.85); --text: #d1f7ff; --muted: #7fa7b5;
}

/* Base: dark grid + faint scanlines */
html, body, [data-testid="stApp"] { background: var(--bg) !important; color: var(--text); }
[data-testid="stApp"] {
  background:
    repeating-linear-gradient(0deg, rgba(255,255,255,0.025) 0 1px, transparent 1px 3px),
    linear-gradient(rgba(0,240,255,0.05) 1px, transparent 1px) 0 0 / 40px 40px,
    linear-gradient(90deg, rgba(0,240,255,0.05) 1px, transparent 1px) 0 0 / 40px 40px,
    radial-gradient(ellipse at top, #1a0533 0%, var(--bg) 60%) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
html, body, p, li, label, span, div, input, textarea, button, [data-testid="stMarkdownContainer"] {
  font-family: 'Share Tech Mono', Consolas, monospace !important;
}
/* Keep Streamlit's icon font for expander arrows etc. */
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }

/* Headings */
h1, h2, h3, h4 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 0.08em; text-transform: uppercase; }
h2, h3 { color: var(--cyan) !important; text-shadow: 0 0 8px rgba(0,240,255,0.6); }

/* Glitch title */
.cp-title {
  font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: clamp(1.8rem, 4.5vw, 3.2rem);
  letter-spacing: 0.06em; text-transform: uppercase; color: #fff; margin: 0.2rem 0 0;
  text-shadow: -2px 0 var(--pink), 2px 0 var(--cyan), 0 0 18px rgba(255,42,109,0.7), 0 0 36px rgba(0,240,255,0.35);
  animation: cp-flicker 6s infinite;
}
.cp-title span { color: var(--yellow); text-shadow: 0 0 12px rgba(249,240,2,0.8); }
.cp-sub {
  color: var(--cyan); font-size: 0.95rem; letter-spacing: 0.12em; text-transform: uppercase;
  border-left: 3px solid var(--pink); padding-left: 0.7rem; margin: 0.4rem 0 1.4rem;
}
.cp-sub b { color: var(--yellow); font-weight: normal; }
@keyframes cp-flicker {
  0%, 92%, 100% { opacity: 1; } 93% { opacity: 0.55; } 94% { opacity: 1; } 96% { opacity: 0.75; } 97% { opacity: 1; }
}
@media (prefers-reduced-motion: reduce) { .cp-title { animation: none; } }

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a0a1f 0%, #12051f 100%) !important;
  border-right: 1px solid var(--pink); box-shadow: 4px 0 24px rgba(255,42,109,0.25);
}
[data-testid="stSidebar"] h2 { color: var(--pink) !important; text-shadow: 0 0 8px rgba(255,42,109,0.7); font-size: 1rem; }
[data-testid="stWidgetLabel"] p { color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.8rem !important; }

/* Inputs */
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
  background: rgba(0, 240, 255, 0.04) !important; border: 1px solid rgba(0,240,255,0.45) !important; border-radius: 0 !important;
}
[data-baseweb="select"] > div:hover { box-shadow: 0 0 10px rgba(0,240,255,0.5); }
[data-baseweb="popover"] ul { background: #0b0b1e !important; border: 1px solid var(--cyan); }

/* Metric cards: clipped corner, neon edge */
[data-testid="stMetric"] {
  background: var(--panel); border: 1px solid rgba(0,240,255,0.5); padding: 0.9rem 1rem;
  clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 14px 100%, 0 calc(100% - 14px));
  box-shadow: inset 0 0 18px rgba(0,240,255,0.12);
}
[data-testid="stMetricLabel"] p { color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] {
  font-family: 'Orbitron', sans-serif !important; color: var(--yellow) !important;
  text-shadow: 0 0 10px rgba(249,240,2,0.6);
}
[data-testid="stColumn"]:nth-child(even) [data-testid="stMetric"] { border-color: rgba(255,42,109,0.6); box-shadow: inset 0 0 18px rgba(255,42,109,0.14); }

/* Chart frame */
[data-testid="stPlotlyChart"] {
  border: 1px solid rgba(0,240,255,0.35); background: rgba(5,5,20,0.55); padding: 0.4rem;
  box-shadow: 0 0 24px rgba(0,240,255,0.12), inset 0 0 40px rgba(122,4,235,0.12);
}

/* Buttons */
.stDownloadButton button, .stButton button {
  background: transparent !important; color: var(--yellow) !important; border: 1px solid var(--yellow) !important;
  border-radius: 0 !important; text-transform: uppercase; letter-spacing: 0.15em;
  box-shadow: 0 0 10px rgba(249,240,2,0.35); transition: all .15s ease;
}
.stDownloadButton button:hover, .stButton button:hover {
  background: var(--yellow) !important; color: #05050f !important; box-shadow: 0 0 22px rgba(249,240,2,0.8);
}
.stDownloadButton button p, .stButton button p { color: inherit !important; }

/* Expanders, alerts, tables */
[data-testid="stExpander"] details { border: 1px solid rgba(255,42,109,0.45) !important; border-radius: 0 !important; background: var(--panel); }
[data-testid="stExpander"] summary p { color: var(--pink) !important; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="stAlert"] { border-radius: 0 !important; border-left: 3px solid var(--yellow); background: rgba(249,240,2,0.06) !important; }
[data-testid="stDataFrame"] { border: 1px solid rgba(0,240,255,0.3); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
strong { color: var(--cyan); }
code { color: var(--pink) !important; background: rgba(255,42,109,0.1) !important; }
hr { border-color: rgba(0,240,255,0.25) !important; }
</style>
"""


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def coverage(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "no data"
    return f"{s.index.min().strftime('%b %Y')} – {s.index.max().strftime('%b %Y')} ({len(s)} months)"


def main() -> None:
    st.set_page_config(page_title="Real Exchange Rate App", page_icon="⚡", layout="wide")
    st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="cp-title">Real Exchange <span>//</span> Rate</div>'
        '<div class="cp-sub">IMF data feed <b>▸</b> monthly CPI + FX <b>▸</b> Jan 1976 → now</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading IMF country list…"):
        countries, from_imf = load_country_list()
    if not from_imf:
        st.warning("Could not download the IMF country list; showing a built-in list of major economies.")

    codes = sorted(countries, key=lambda c: (c != EURO_AREA, countries[c].lower()))
    fmt = lambda c: f"{countries[c]} ({c})"

    with st.sidebar:
        st.header("Countries")
        c1 = st.selectbox("Country 1", codes, index=codes.index(USA) if USA in codes else 0, format_func=fmt)
        c2 = st.selectbox("Country 2", codes, index=codes.index(EURO_AREA) if EURO_AREA in codes else 1, format_func=fmt)

    if c1 == c2:
        st.info("Choose two different countries.")
        st.stop()

    n1, n2 = countries[c1], countries[c2]
    try:
        with st.spinner(f"Downloading IMF data for {n1} and {n2}…"):
            p1 = load_series(CPI_FLOW, c1)
            p2 = load_series(CPI_FLOW, c2)
            e1 = load_series(ER_FLOW, c1) if c1 != USA else pd.Series(dtype=float)
            e2 = load_series(ER_FLOW, c2) if c2 != USA else pd.Series(dtype=float)
    except Exception as exc:
        st.error(f"Could not retrieve IMF data: {exc}")
        st.stop()

    idx = p1.index.union(p2.index).union(e1.index).union(e2.index)
    e1, note1 = exchange_rate_per_usd(c1, e1, idx)
    e2, note2 = exchange_rate_per_usd(c2, e2, idx)
    panel = build_panel(e1, e2, p1, p2)

    with st.expander("Data coverage", expanded=False):
        st.markdown(
            f"- **CPI, {n1}:** {coverage(p1)}\n"
            f"- **CPI, {n2}:** {coverage(p2)}\n"
            f"- **Exchange rate, {n1} (per USD):** {note1}\n"
            f"- **Exchange rate, {n2} (per USD):** {note2}"
        )

    valid_months = panel.index[panel["valid"]]
    if len(valid_months) == 0:
        st.error("No month has all four inputs (both CPIs and both exchange rates). "
                 "The IMF may not publish monthly data for one of these countries.")
        st.stop()

    first, last = valid_months.min(), valid_months.max()
    months = list(pd.period_range(first, last, freq="M"))

    with st.sidebar:
        st.header("Sample period")
        start, end = st.select_slider(
            "Start and end month", options=months, value=(months[0], months[-1]),
            format_func=lambda p: p.strftime("%b %Y"),
        )
        st.header("Chart")
        mode = st.radio("Show", ["Real", "Nominal", "Both"], index=2, horizontal=True)
        log_nominal = st.checkbox("Log scale for nominal S", value=False,
                                  help="Useful when S changes by orders of magnitude (high-inflation currencies).")

    w, rer_star = apply_window(panel, start, end)
    n_valid = int(w["valid"].sum())
    n_total = len(w)
    if n_valid == 0:
        st.warning("The selected period contains no month with complete data. Widen the period.")
        st.stop()

    s_label = f"{n2} currency per 1 {n1} currency"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RER* (sample average)", f"{rer_star:,.4g}")
    m2.metric(f"Valid months (of {n_total})", f"{n_valid}")
    last_valid = w.loc[w["valid"]].iloc[-1]
    m3.metric(f"Real Index, {w.index[w['valid']][-1].strftime('%b %Y')}", f"{last_valid['Real_Index']:.1f}")
    m4.metric("Nominal S, same month", f"{last_valid['S']:,.4f}")

    if n_valid < n_total:
        st.caption(f"{n_total - n_valid} month(s) in the window lack at least one input; "
                   "they are excluded from RER* and shown as gaps.")

    st.plotly_chart(make_chart(w, mode, s_label, log_nominal), width="stretch")

    st.markdown(
        f"**How to read it.** S(t) = units of {n2} currency per 1 unit of {n1} currency "
        f"(= IMF rate of {c2} per USD ÷ IMF rate of {c1} per USD). "
        f"RER(t) = S(t) × P1(t) / P2(t). The Real Index is RER rescaled so its average over "
        f"{start.strftime('%b %Y')}–{end.strftime('%b %Y')} equals 100. "
        f"A **higher** index means {n1}'s currency is **stronger in real terms** against {n2}'s "
        f"(goods in {n1} are relatively more expensive than over the sample on average)."
    )

    st.download_button(
        "Download CSV (all months in the selected period)",
        data=to_csv(w, c1, c2),
        file_name=f"rer_{c1}_{c2}_{start.strftime('%Y%m')}_{end.strftime('%Y%m')}.csv",
        mime="text/csv",
    )

    with st.expander("Show data table"):
        show = w.copy()
        show.index = show.index.strftime("%Y-%m")
        st.dataframe(show, width="stretch")

    st.caption(f"Source: IMF SDMX 3.0 API ({BASE_URL}), dataflows {AGENCY}:{CPI_FLOW} and {AGENCY}:{ER_FLOW}. "
               f"Data cached for 24 hours. Retrieved {date.today():%Y-%m-%d}.")


if __name__ == "__main__":
    main()
