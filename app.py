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
def make_chart(w: pd.DataFrame, mode: str, s_label: str, log_nominal: bool) -> go.Figure:
    x = w.index.to_timestamp()
    real = go.Scatter(x=x, y=w["Real_Index"], name="Real Index (avg = 100)",
                      mode="lines", line=dict(color="#1f5fa8", width=2),
                      hovertemplate="%{x|%b %Y}<br>Real Index: %{y:.2f}<extra></extra>",
                      connectgaps=False)
    nom = go.Scatter(x=x, y=w["S"], name=f"Nominal S ({s_label})",
                     mode="lines", line=dict(color="#c4572b", width=2),
                     hovertemplate="%{x|%b %Y}<br>S: %{y:,.4f}<extra></extra>",
                     connectgaps=False)

    if mode == "Both":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(real, secondary_y=False)
        fig.add_trace(nom, secondary_y=True)
        fig.update_yaxes(title_text="Real Index (sample avg = 100)", secondary_y=False)
        fig.update_yaxes(title_text=s_label, secondary_y=True,
                         type="log" if log_nominal else "linear")
        fig.add_hline(y=100, line_dash="dot", line_color="gray", secondary_y=False)
    elif mode == "Real":
        fig = go.Figure(real)
        fig.update_yaxes(title_text="Real Index (sample avg = 100)")
        fig.add_hline(y=100, line_dash="dot", line_color="gray")
    else:
        fig = go.Figure(nom)
        fig.update_yaxes(title_text=s_label, type="log" if log_nominal else "linear")

    fig.update_layout(
        height=520, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def coverage(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "no data"
    return f"{s.index.min().strftime('%b %Y')} – {s.index.max().strftime('%b %Y')} ({len(s)} months)"


def main() -> None:
    st.set_page_config(page_title="Real Exchange Rate App", layout="wide")
    st.title("Real Exchange Rate App")
    st.caption("Monthly IMF CPI and exchange-rate data (SDMX 3.0 API), January 1976 onward where available.")

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
    m2.metric("Valid months", f"{n_valid} of {n_total}")
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
