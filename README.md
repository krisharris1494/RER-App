# Real Exchange Rate App

An interactive Streamlit app that computes and plots the bilateral **real exchange rate** between any two IMF countries (or the Euro area) using monthly IMF CPI and exchange-rate data from January 1976 onward, where available.

**Live app:** _add your Streamlit URL here after deploying_

## What it does

1. Pick **Country 1** and **Country 2** from the IMF country list (the Euro area is IMF code `G163`).
2. The app downloads monthly CPI and exchange-rate data from the IMF SDMX 3.0 API.
3. Choose a **start and end month**. The app computes the sample-average real exchange rate over that window and rescales the series to an index.
4. View the **Real** index, the **Nominal** exchange rate, or **Both** (separate y-axes).
5. **Download a CSV** with every month in the selected period: inputs, intermediate calculations, and the index.

## Economics

| Symbol | Definition |
|---|---|
| E₁(t), E₂(t) | IMF exchange rate: units of each country's currency per 1 US dollar |
| S(t) | Units of **Country 2** currency per 1 unit of **Country 1** currency = E₂(t) / E₁(t) |
| P₁(t), P₂(t) | Monthly all-items CPI for Country 1 and Country 2 |
| RER(t) | S(t) × P₁(t) / P₂(t) |
| RER* | Average of RER(t) over the valid months in the selected window |
| Real Index(t) | 100 × RER(t) / RER* |

- The IMF quotes every currency against the US dollar, so the app converts to the cross rate S(t) = E₂ / E₁. For the United States, E = 1 by definition.
- A **valid month** has all four inputs (E₁, E₂, P₁, P₂) present and positive. Months with any missing input are excluded from RER* and appear as gaps in the chart.
- RER* is recalculated whenever the start or end month changes, so the Real Index always averages 100 over the chosen window.
- Nominal S(t) is plotted in actual currency units and is **not** normalized. A log-scale option helps when S moves by orders of magnitude.
- **Interpretation:** a higher Real Index means Country 1's currency is stronger in real terms against Country 2's; Country 1's goods are relatively more expensive than their average over the window.
- CPI base years differ across countries. That does not matter for the Real Index, because rescaling by RER* removes the arbitrary level.

## Data

- Source: IMF SDMX 3.0 API, `https://api.imf.org/external/sdmx/3.0`
- CPI: dataflow `IMF.STA:CPI`, key `{ISO3}.CPI._T.IX.M` (all-items index, monthly)
- Exchange rate: dataflow `IMF.STA:ER`, key `{ISO3}.XDC_USD.PA_RT.M` (domestic currency per USD, period average, monthly)
- Responses are cached for 24 hours.

### Data caveats

- Coverage varies by country. Many series start after 1976, and some countries have no monthly data. The app's "Data coverage" panel shows what was found.
- If the IMF does not publish a Euro-area aggregate exchange rate, the app uses Germany's series from January 1999 (euros per USD) and says so in the coverage panel.
- Currency redenominations (for example, countries that dropped zeros from their currency) and the 1999 switch to the euro for member countries can create level breaks. The IMF usually restates history in current currency units, but check the nominal chart for jumps.
- Monthly exchange rates are period averages, not end-of-month rates.

## Run it online (no local Python needed)

1. Put `app.py`, `requirements.txt`, and `README.md` in a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **Create app**.
3. Choose the repository, branch `main`, and main file `app.py`. Under **Advanced settings**, pick Python 3.12.
4. Click **Deploy**.

To run locally instead: `pip install -r requirements.txt` then `streamlit run app.py`.

## Files

- `app.py` — the Streamlit app (data download, calculations, charts, CSV export)
- `requirements.txt` — Python dependencies (Python 3.12 compatible)
- `README.md` — this file

## AI assistance

The code was written with an LLM. The author wrote the prompts and verified the economics.

- Model: _model name_
- Date: _date_
- Prompts: _paste or link your prompts here_
