# Stochastic Optimization of EV Charging Station Allocation Under Uncertain Demand

Two-stage stochastic programming with Monte Carlo scenarios to decide where to
build new EV chargers when demand is uncertain. Compares a stochastic plan with
a deterministic (average-demand) baseline and reports the Value of the
Stochastic Solution (VSS).

## Run locally
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tech
Python, Streamlit, Plotly, PuLP (CBC), SciPy, NumPy, Pandas