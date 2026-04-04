# Nifty Regime Tool

A Streamlit app to score Nifty 50 or Nifty Next 50 into:
- Extreme Buy
- Buy
- Accumulate
- Neutral
- Reduce
- Sell
- Extreme Sell

## What it uses
1. **Valuation history** (optional but recommended)
   - P/E
   - P/B
   - Dividend Yield
2. **Technical regime**
   - Drawdown from ATH
   - 200 DMA distance
   - RSI(14)
3. **Macro overlays**
   - S&P 500 trend
   - Brent crude
   - USD/INR
   - US 10Y yield
   - VIX

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repo.
2. Add `app.py`, `requirements.txt`, and optionally the `data/` folder.
3. In Streamlit Cloud, pick `app.py` as the entry point.
4. If you have official NSE valuation history CSVs, add them under:
   - `data/nifty50_valuation.csv`
   - `data/nifty_next50_valuation.csv`

## CSV format expected
Valuation file should contain:
- `Date`
- `P/E`
- `P/B`
- `Div Yield %`

Optional:
- `Close`
- `IndexName`

## Notes
- The app works without valuation CSVs, but then it runs in **price + macro mode** only.
- Best practice is to use official NSE valuation history exports.