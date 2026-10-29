import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
import streamlit as st
from scipy.optimize import linprog

st.set_page_config(page_title="EV Charging Stochastic Optimization", layout="wide")
st.title("⚡ Stochastic Optimization of EV Charging Station Allocation Under Uncertain Demand")

# ----------------------------------------------------------------------------
# 1. SIDEBAR: PARAMETERS
# ----------------------------------------------------------------------------
st.sidebar.header("Model parameters")
cap = st.sidebar.number_input("Sessions served per charger per day", 5, 100, 30)
install = st.sidebar.number_input("Install cost per new charger (per day)", 1.0, 1000.0, 60.0)
penalty = st.sidebar.number_input("Penalty per unmet session (lost profit)", 1.0, 500.0, 25.0)
rate = st.sidebar.number_input("Transfer cost per km per session", 0.0, 20.0, 1.5)
max_dist = st.sidebar.number_input("Max distance EV will travel to another zone (km)", 0.0, 50.0, 6.0)
budget = st.sidebar.number_input("Max total new chargers (budget)", 1, 200, 16)
max_new = st.sidebar.number_input("Max new chargers per zone", 1, 50, 8)

st.sidebar.header("Demand uncertainty")
sig_c = st.sidebar.slider("City-wide volatility (common shock)", 0.0, 0.6, 0.15, 0.01)
sig_i = st.sidebar.slider("Zone-level volatility (local noise)", 0.0, 0.8, 0.25, 0.01)
n_opt = st.sidebar.slider("Scenarios used for optimization", 20, 300, 100, 10)
n_eval = st.sidebar.slider("Monte Carlo scenarios for evaluation", 1000, 10000, 1000, 500)
seed = st.sidebar.number_input("Random seed", 0, 10000, 42)

if penalty <= rate * max_dist:
    st.sidebar.warning("Penalty should be larger than the max transfer cost "
                       "(rate × max distance), otherwise leaving demand unmet is cheaper.")

# ----------------------------------------------------------------------------
# 2. ZONE DATA (editable table)
# ----------------------------------------------------------------------------
st.subheader("1. Zone data (editable)")
default_zones = pd.DataFrame({
    "Zone": ["Downtown", "University", "Mall", "Airport", "Residential", "Tech Park"],
    "Mean_Demand": [120, 90, 100, 70, 80, 110],   # expected charging sessions per day
    "X_km": [5, 2, 8, 12, 1, 9],
    "Y_km": [5, 8, 2, 9, 2, 8],
    "Existing": [2, 1, 1, 1, 1, 2],               # chargers already installed
})
zones = st.data_editor(default_zones, num_rows="fixed", use_container_width=True)

P = dict(
    names=zones["Zone"].tolist(),
    mean=zones["Mean_Demand"].to_numpy(float),
    exist=zones["Existing"].to_numpy(float),
    X=zones["X_km"].to_numpy(float),
    Y=zones["Y_km"].to_numpy(float),
    cap=cap, install=install, penalty=penalty, rate=rate,
    max_dist=max_dist, budget=budget, max_new=max_new,
    sig_c=sig_c, sig_i=sig_i,
)
P["dist"] = np.sqrt((P["X"][:, None] - P["X"][None, :]) ** 2 +
                    (P["Y"][:, None] - P["Y"][None, :]) ** 2)


# ----------------------------------------------------------------------------
# 3. CORE FUNCTIONS
# ----------------------------------------------------------------------------
def generate_scenarios(P, n, rng, sig_i=None):
    """Monte Carlo demand scenarios (n x zones).
    demand = mean * common city-wide shock * local zone noise  (both lognormal, mean = 1)."""
    sig_i = P["sig_i"] if sig_i is None else sig_i
    k = len(P["mean"])
    common = rng.lognormal(-P["sig_c"] ** 2 / 2, P["sig_c"], size=(n, 1))
    local = rng.lognormal(-sig_i ** 2 / 2, sig_i, size=(n, k)) if sig_i > 0 else np.ones((n, k))
    return P["mean"] * common * local


def build_recourse(P):
    """Pre-build the second-stage LP structure (reused for every scenario)."""
    n = len(P["mean"])
    allowed = P["dist"] <= P["max_dist"]
    pairs = [(i, j) for i in range(n) for j in range(n) if allowed[i, j]]  # demand zone i -> charger zone j
    unit_cost = P["rate"] * P["dist"]                                       # 0 when i == j
    k = len(pairs)
    c = np.array([unit_cost[i, j] for i, j in pairs] + [P["penalty"]] * n)  # [transfers..., unmet...]
    A_eq = np.zeros((n, k + n))   # every zone's demand = served (anywhere) + unmet
    A_ub = np.zeros((n, k + n))   # sessions served at zone j <= capacity of zone j
    for idx, (i, j) in enumerate(pairs):
        A_eq[i, idx] = 1
        A_ub[j, idx] = 1
    A_eq[np.arange(n), k + np.arange(n)] = 1
    return dict(n=n, pairs=pairs, k=k, unit_cost=unit_cost, c=c, A_eq=A_eq, A_ub=A_ub)


def second_stage(d, capacity, M):
    """Solve the recourse LP for ONE demand scenario. Returns (cost, unmet per zone)."""
    res = linprog(M["c"], A_ub=M["A_ub"], b_ub=capacity, A_eq=M["A_eq"], b_eq=d,
                  bounds=(0, None), method="highs")
    return res.fun, res.x[M["k"]:]


def solve_two_stage(scen, P, M, budget):
    """Extensive-form two-stage stochastic program (MILP, solved with PuLP/CBC).
    With a single scenario (the mean demand) this IS the deterministic baseline."""
    n, S, pairs = M["n"], len(scen), M["pairs"]
    prob = pulp.LpProblem("EV_Charging", pulp.LpMinimize)

    # First-stage decision: number of new chargers per zone (integer)
    x = [pulp.LpVariable(f"x_{j}", 0, P["max_new"], cat="Integer") for j in range(n)]
    # Second-stage decisions, one set per scenario
    t = {(s, i, j): pulp.LpVariable(f"t_{s}_{i}_{j}", 0) for s in range(S) for i, j in pairs}
    u = {(s, i): pulp.LpVariable(f"u_{s}_{i}", 0) for s in range(S) for i in range(n)}

    # Objective: install cost + average (expected) recourse cost over scenarios
    prob += (P["install"] * pulp.lpSum(x) +
             (1.0 / S) * pulp.lpSum(M["unit_cost"][i, j] * t[s, i, j] for s in range(S) for i, j in pairs) +
             (1.0 / S) * pulp.lpSum(P["penalty"] * u[s, i] for s in range(S) for i in range(n)))

    prob += pulp.lpSum(x) <= budget                               # budget constraint
    for s in range(S):
        for i in range(n):                                        # demand balance
            prob += (pulp.lpSum(t[s, i, j] for (a, j) in pairs if a == i) + u[s, i]) == float(scen[s][i])
        for j in range(n):                                        # capacity
            prob += (pulp.lpSum(t[s, i, j] for (i, b) in pairs if b == j)
                     <= P["cap"] * (P["exist"][j] + x[j]))

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    x_val = np.array([int(round(v.value() or 0)) for v in x])
    return x_val, pulp.value(prob.objective)


def evaluate(x_new, scen, P, M):
    """Monte Carlo evaluation of a fixed first-stage plan on many scenarios."""
    capacity = P["cap"] * (P["exist"] + x_new)
    costs, unmet_tot, unmet_zone = [], [], []
    for d in scen:
        val, um = second_stage(d, capacity, M)
        costs.append(P["install"] * x_new.sum() + val)
        unmet_tot.append(um.sum())
        unmet_zone.append(um)
    costs, unmet_tot = np.array(costs), np.array(unmet_tot)
    total_dem = scen.sum(axis=1)
    return dict(
        costs=costs,
        mean_cost=costs.mean(),
        std_cost=costs.std(),
        p95_cost=np.percentile(costs, 95),
        service=1 - unmet_tot.sum() / total_dem.sum(),
        prob_unmet=(unmet_tot > 1e-6).mean(),
        unmet_zone=np.mean(unmet_zone, axis=0),
    )


# ----------------------------------------------------------------------------
# 4. RUN MAIN EXPERIMENT
# ----------------------------------------------------------------------------
st.subheader("2. Run optimization")
if st.button("▶ Run stochastic optimization", type="primary"):
    with st.spinner("Generating scenarios and solving..."):
        M = build_recourse(P)
        opt_scen = generate_scenarios(P, n_opt, np.random.default_rng(seed))
        eval_scen = generate_scenarios(P, n_eval, np.random.default_rng(seed + 1))  # fresh scenarios

        x_det, _ = solve_two_stage(P["mean"][None, :], P, M, budget)   # deterministic (mean demand)
        x_sto, obj_sto = solve_two_stage(opt_scen, P, M, budget)       # stochastic (RP)

        res_det = evaluate(x_det, eval_scen, P, M)
        res_sto = evaluate(x_sto, eval_scen, P, M)
        st.session_state["main"] = dict(x_det=x_det, x_sto=x_sto, res_det=res_det, res_sto=res_sto,
                                        eval_scen=eval_scen, obj_sto=obj_sto, P=P)

if "main" in st.session_state:
    R = st.session_state["main"]
    x_det, x_sto, rd, rs = R["x_det"], R["x_sto"], R["res_det"], R["res_sto"]
    names = R["P"]["names"]

    # ---- Key metrics ----
    vss = rd["mean_cost"] - rs["mean_cost"]   # Value of the Stochastic Solution
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected cost - Deterministic", f"{rd['mean_cost']:,.1f}")
    c2.metric("Expected cost - Stochastic", f"{rs['mean_cost']:,.1f}")
    c3.metric("VSS (savings)", f"{vss:,.1f}", f"{100 * vss / rd['mean_cost']:.2f}%")
    c4.metric("Service level (Stochastic)", f"{100 * rs['service']:.1f}%",
              f"{100 * (rs['service'] - rd['service']):+.2f} pts vs deterministic")

    tab1, tab2, tab3, tab4 = st.tabs(["Allocation", "Cost analysis", "Scenarios", "Sensitivity"])

    # ---- Tab 1: allocation ----
    with tab1:
        alloc = pd.DataFrame({
            "Zone": names,
            "Existing": R["P"]["exist"].astype(int),
            "New (Deterministic)": x_det,
            "New (Stochastic)": x_sto,
            "Avg unmet/day (Det)": rd["unmet_zone"].round(1),
            "Avg unmet/day (Sto)": rs["unmet_zone"].round(1),
        })
        st.dataframe(alloc, use_container_width=True)
        fig = go.Figure()
        fig.add_bar(x=names, y=R["P"]["exist"], name="Existing")
        fig.add_bar(x=names, y=x_det, name="New - Deterministic")
        fig.add_bar(x=names, y=x_sto, name="New - Stochastic")
        fig.update_layout(barmode="group", title="Chargers per zone", yaxis_title="Chargers")
        st.plotly_chart(fig, use_container_width=True)

        map_df = pd.DataFrame({"Zone": names, "X": R["P"]["X"], "Y": R["P"]["Y"],
                               "Total chargers (Stochastic)": R["P"]["exist"] + x_sto,
                               "Mean demand": R["P"]["mean"]})
        fig_map = px.scatter(map_df, x="X", y="Y", size="Total chargers (Stochastic)", color="Mean demand",
                             text="Zone", title="Zone map (bubble size = chargers, color = mean demand)",
                             size_max=45)
        fig_map.update_traces(textposition="top center")
        st.plotly_chart(fig_map, use_container_width=True)

    # ---- Tab 2: cost analysis ----
    with tab2:
        summary = pd.DataFrame({
            "Metric": ["Mean cost", "Std dev of cost", "95th percentile cost (risk)",
                       "Service level", "Prob. of any unmet demand"],
            "Deterministic": [rd["mean_cost"], rd["std_cost"], rd["p95_cost"],
                              rd["service"], rd["prob_unmet"]],
            "Stochastic": [rs["mean_cost"], rs["std_cost"], rs["p95_cost"],
                           rs["service"], rs["prob_unmet"]],
        })
        st.dataframe(summary.round(3), use_container_width=True)
        fig_h = go.Figure()
        fig_h.add_histogram(x=rd["costs"], name="Deterministic", opacity=0.6, nbinsx=50)
        fig_h.add_histogram(x=rs["costs"], name="Stochastic", opacity=0.6, nbinsx=50)
        fig_h.update_layout(barmode="overlay", title=f"Cost distribution over {len(R['eval_scen'])} Monte Carlo scenarios",
                            xaxis_title="Total daily cost")
        st.plotly_chart(fig_h, use_container_width=True)
        st.download_button("Download allocation CSV", alloc.to_csv(index=False), "allocation.csv")

    # ---- Tab 3: scenarios ----
    with tab3:
        sc = pd.DataFrame(R["eval_scen"], columns=names)
        st.write("Summary of simulated demand (sessions/day):")
        st.dataframe(sc.describe().round(1), use_container_width=True)
        st.plotly_chart(px.box(sc, title="Demand distribution by zone"), use_container_width=True)

    # ---- Tab 4: sensitivity ----
    with tab4:
        st.write("Re-solves the model for different budgets and volatility levels "
                 "(uses fewer scenarios so it finishes quickly).")
        if st.button("Run sensitivity analysis"):
            Pm = R["P"]
            M = build_recourse(Pm)
            s_opt = min(n_opt, 60)
            rows_b, rows_v = [], []
            prog = st.progress(0.0)

            # (a) vary budget
            budgets = [8, 12, 16, 20, 24]
            for k, B in enumerate(budgets):
                rng = np.random.default_rng(seed)
                sc_o = generate_scenarios(Pm, s_opt, rng)
                sc_e = generate_scenarios(Pm, 500, np.random.default_rng(seed + 1))
                xs, _ = solve_two_stage(sc_o, Pm, M, B)
                xd, _ = solve_two_stage(Pm["mean"][None, :], Pm, M, B)
                es, ed = evaluate(xs, sc_e, Pm, M), evaluate(xd, sc_e, Pm, M)
                rows_b.append(dict(Budget=B, Stochastic=es["mean_cost"], Deterministic=ed["mean_cost"],
                                   VSS=ed["mean_cost"] - es["mean_cost"], Service=es["service"]))
                prog.progress((k + 1) / (len(budgets) + 4))

            # (b) vary zone-level volatility
            for k, sg in enumerate([0.1, 0.2, 0.3, 0.4]):
                sc_o = generate_scenarios(Pm, s_opt, np.random.default_rng(seed), sig_i=sg)
                sc_e = generate_scenarios(Pm, 500, np.random.default_rng(seed + 1), sig_i=sg)
                xs, _ = solve_two_stage(sc_o, Pm, M, budget)
                xd, _ = solve_two_stage(Pm["mean"][None, :], Pm, M, budget)
                es, ed = evaluate(xs, sc_e, Pm, M), evaluate(xd, sc_e, Pm, M)
                rows_v.append(dict(Volatility=sg, Stochastic=es["mean_cost"], Deterministic=ed["mean_cost"],
                                   VSS=ed["mean_cost"] - es["mean_cost"], Service=es["service"]))
                prog.progress((len(budgets) + k + 1) / (len(budgets) + 4))
            st.session_state["sens"] = (pd.DataFrame(rows_b), pd.DataFrame(rows_v))

        if "sens" in st.session_state:
            db, dv = st.session_state["sens"]
            st.write("**Effect of budget**")
            st.dataframe(db.round(2), use_container_width=True)
            st.plotly_chart(px.line(db, x="Budget", y=["Stochastic", "Deterministic"], markers=True,
                                    title="Expected cost vs budget"), use_container_width=True)
            st.write("**Effect of zone-level volatility**")
            st.dataframe(dv.round(2), use_container_width=True)
            st.plotly_chart(px.line(dv, x="Volatility", y=["VSS"], markers=True,
                                    title="Value of Stochastic Solution vs demand volatility"),
                            use_container_width=True)
else:
    st.info("Set parameters in the sidebar and click **Run stochastic optimization**.")