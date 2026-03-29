"""
Ising Model Selection via ℓ₁-Regularized Logistic Regression
=============================================================
Reproduces experiments from:
  Ravikumar, Wainwright & Lafferty (2010)
  "High-Dimensional Ising Model Selection Using ℓ₁-Regularized Logistic Regression"
  Annals of Statistics, Vol. 38, No. 3, pp. 1287–1319

Install & run:
  pip install streamlit plotly matplotlib networkx scikit-learn numpy pandas
  streamlit run ising_model_selection.py

TO-DO:
- beta range -> n/(10 d log p) for each graph
- graph previews for multiple betas
- change notation from grid side to p
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LogisticRegression


# ══════════════════════════════════════════════════════════════
#  Graph construction
# ══════════════════════════════════════════════════════════════

def build_grid_graph(p_side: int, connectivity: int = 4):
    """Return (p, edge_list) for a 4-NN or 8-NN 2-D grid."""
    p = p_side * p_side
    edges = []
    for i in range(p_side):
        for j in range(p_side):
            node = i * p_side + j
            if connectivity >= 4:
                if j + 1 < p_side: edges.append((node, i * p_side + j + 1))
                if i + 1 < p_side: edges.append((node, (i + 1) * p_side + j))
            if connectivity >= 8:
                if i + 1 < p_side and j + 1 < p_side:
                    edges.append((node, (i + 1) * p_side + j + 1))
                if i + 1 < p_side and j - 1 >= 0:
                    edges.append((node, (i + 1) * p_side + j - 1))
    return p, edges


def build_star_graph(p: int, degree_type: str = "logarithmic"):
    """Hub (node 0) connected to d leaves.  d = ⌈log p⌉ or ⌈0.1 p⌉."""
    d = (max(1, int(np.ceil(np.log(p)))) if degree_type == "logarithmic"
         else max(1, int(np.ceil(0.1 * p))))
    d = min(d, p - 1)
    return p, [(0, i) for i in range(1, d + 1)], d


# ══════════════════════════════════════════════════════════════
#  Ising model utilities
# ══════════════════════════════════════════════════════════════

def make_theta_matrix(p: int, edges: list, omega: float,
                      coupling_type: str = "mixed") -> np.ndarray:
    """Build p×p coupling matrix: ±ω (mixed) or +ω (attractive)."""
    theta = np.zeros((p, p))
    rng = np.random.RandomState(0)           # fixed seed → reproducible graph
    for s, t in edges:
        w = omega if coupling_type == "attractive" else omega * rng.choice([-1, 1])
        theta[s, t] = theta[t, s] = w
    return theta


def gibbs_sample_ising(theta: np.ndarray, n_samples: int,
                        n_burnin: int = 300, seed: int = None) -> np.ndarray:
    """Sequential Gibbs sampler for the {−1,+1} Ising model."""
    rng = np.random.RandomState(seed)
    p   = theta.shape[0]
    x   = rng.choice(np.array([-1.0, 1.0]), size=p)
    out = []
    for step in range(n_burnin + n_samples):
        for i in range(p):
            prob = 1.0 / (1.0 + np.exp(-2.0 * float(theta[i] @ x)))
            x[i] = 1.0 if rng.rand() < prob else -1.0
        if step >= n_burnin:
            out.append(x.copy())
    return np.array(out)


def exact_sample_star(theta: np.ndarray, n_samples: int,
                      seed: int = None) -> np.ndarray:
    """Exact ancestral sampling: hub first, then leaves conditionally independent."""
    rng = np.random.RandomState(seed)
    p   = theta.shape[0]
    out = []
    for _ in range(n_samples):
        x    = np.zeros(p)
        x[0] = 1.0 if rng.rand() < 0.5 else -1.0
        for i in range(1, p):
            if theta[i, 0] != 0:
                prob = 1.0 / (1.0 + np.exp(-2.0 * theta[i, 0] * x[0]))
                x[i] = 1.0 if rng.rand() < prob else -1.0
            else:
                x[i] = 1.0 if rng.rand() < 0.5 else -1.0
        out.append(x)
    return np.array(out)


# ══════════════════════════════════════════════════════════════
#  ℓ₁-regularised logistic regression  (c = 1, fixed)
# ══════════════════════════════════════════════════════════════
#
#  Paper:   min_θ  −(1/n) Σᵢ log P_θ(xᵣ|x_{−r})  + λₙ ‖θ‖₁
#  λₙ = √(log p / n)                       c = 1 fixed
#
#  sklearn liblinear minimises: (1/C)‖w‖₁ + Σᵢ log(1+exp(−yᵢwᵀxᵢ))
#  Matching (multiply paper obj by n):   C = 1/(n·λₙ) = 1/√(n·log p)
#
#  fit_intercept=False: Ising model is zero-mean (no external field)
#  AND rule: include edge (s,t) iff t∈N̂(s) AND s∈N̂(t)

def nbhd_logistic(X: np.ndarray, r: int, lambda_n: float) -> set:
    n, p = X.shape
    y    = X[:, r].astype(float)
    Xr   = np.delete(X, r, axis=1).astype(float)
    if len(np.unique(y)) < 2:
        return set()
    C_val = 1.0 / max(lambda_n * n, 1e-10)
    clf = LogisticRegression(
        penalty="l1", C=C_val, solver="liblinear",
        max_iter=2000, fit_intercept=False, tol=1e-4,
    )
    clf.fit(Xr, y)
    nbhd = set()
    for idx in np.where(np.abs(clf.coef_[0]) > 1e-8)[0]:
        nbhd.add(int(idx) if idx < r else int(idx) + 1)
    return nbhd


def estimate_graph(X: np.ndarray, lambda_n: float) -> set:
    """AND-rule symmetrisation."""
    p     = X.shape[1]
    nbhds = [nbhd_logistic(X, r, lambda_n) for r in range(p)]
    edges = set()
    for s in range(p):
        for t in nbhds[s]:
            if s in nbhds[t]:
                edges.add((min(s, t), max(s, t)))
    return edges


# ══════════════════════════════════════════════════════════════
#  Experiment runners
# ══════════════════════════════════════════════════════════════

def run_grid_experiment(p_side, connectivity, coupling_type, omega,
                        beta_values, n_trials, progress_bar=None):
    p, edges = build_grid_graph(p_side, connectivity)
    d        = connectivity
    true_set = {(min(s, t), max(s, t)) for s, t in edges}
    theta    = make_theta_matrix(p, edges, omega, coupling_type)
    log_p    = np.log(p)
    # burnin scales with p to ensure good mixing
    burnin   = max(300, p * 5)

    results, done, total = [], 0, len(beta_values) * n_trials
    for beta in beta_values:
        n       = max(30, int(np.ceil(beta * 10 * d * log_p)))
        lam_n   = np.sqrt(log_p / n)          # c = 1
        success = 0
        for trial in range(n_trials):
            seed = trial * 997 + p_side * 31
            X    = gibbs_sample_ising(theta, n, n_burnin=burnin, seed=seed)
            est  = estimate_graph(X, lam_n)
            if est == true_set:
                success += 1
            done += 1
            if progress_bar is not None:
                progress_bar.progress(done / total)
        results.append(success / n_trials)
    return beta_values, results


def run_star_experiment(p, degree_type, omega, beta_values, n_trials,
                        progress_bar=None):
    _, edges, d = build_star_graph(p, degree_type)
    true_set    = {(min(s, t), max(s, t)) for s, t in edges}
    theta       = make_theta_matrix(p, edges, omega, "attractive")
    log_p       = np.log(p)

    succ_list, disagree_list = [], []
    done, total = 0, len(beta_values) * n_trials
    for beta in beta_values:
        n         = max(30, int(np.ceil(beta * 10 * d * log_p)))
        lam_n     = np.sqrt(log_p / n)        # c = 1
        success   = 0
        disagrees = []
        for trial in range(n_trials):
            X   = exact_sample_star(theta, n, seed=trial * 997 + p * 13)
            est = estimate_graph(X, lam_n)
            if est == true_set:
                success += 1
            disagrees.append(len(true_set.symmetric_difference(est)))
            done += 1
            if progress_bar is not None:
                progress_bar.progress(done / total)
        succ_list.append(success / n_trials)
        disagree_list.append(float(np.mean(disagrees)))
    return beta_values, succ_list, disagree_list


# ══════════════════════════════════════════════════════════════
#  Plotly chart builders  (light theme)
# ══════════════════════════════════════════════════════════════

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#f8f9fa",
    font=dict(color="#1a1a2e", family="Inter, sans-serif", size=12),
    legend=dict(bgcolor="white", bordercolor="#dee2e6", borderwidth=1),
    margin=dict(l=60, r=30, t=55, b=55),
)
_AXIS = dict(
    gridcolor="#dee2e6",
    zerolinecolor="#adb5bd",
    linecolor="#adb5bd",
    tickcolor="#6c757d",
)


def plotly_phase_transition(all_results: dict, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=0.5, line_dash="dot", line_color="#adb5bd", line_width=1.5,
                  annotation_text="50 %", annotation_font_color="#6c757d",
                  annotation_position="right")
    for ci, (label, vals) in enumerate(all_results.items()):
        betas, succs = vals[0], vals[1]
        col = PALETTE[ci % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=betas, y=succs, mode="lines+markers", name=label,
            line=dict(color=col, width=2.2),
            marker=dict(size=7, color=col, line=dict(color="white", width=1.2)),
            hovertemplate=f"<b>{label}</b><br>β=%{{x:.2f}}<br>P[success]=%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14, color="#1a1a2e")),
        xaxis=dict(title="β = n / (10 d log p)", **_AXIS),
        yaxis=dict(title="P[N̂(r) = N(r), ∀r]", range=[-0.05, 1.05], **_AXIS),
        legend_title="Graph size",
        height=440, **_LAYOUT,
    )
    return fig


def plotly_star_dual(all_results: dict, title: str) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["P[success] vs β", "Mean edge disagreements vs β"],
        horizontal_spacing=0.12,
    )
    for ci, (label, vals) in enumerate(all_results.items()):
        betas, succs, disagrees = vals
        col = PALETTE[ci % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=betas, y=succs, mode="lines+markers", name=label,
            line=dict(color=col, width=2.2),
            marker=dict(size=7, color=col, line=dict(color="white", width=1.2)),
            hovertemplate=f"<b>{label}</b><br>β=%{{x:.2f}}<br>P=%{{y:.2f}}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=betas, y=disagrees, mode="lines+markers",
            name=label, showlegend=False,
            line=dict(color=col, width=2.2, dash="dash"),
            marker=dict(size=7, symbol="square", color=col,
                        line=dict(color="white", width=1.2)),
            hovertemplate=f"<b>{label}</b><br>β=%{{x:.2f}}<br>Disagree=%{{y:.2f}}<extra></extra>",
        ), row=1, col=2)

    fig.add_hline(y=0.5, line_dash="dot", line_color="#adb5bd",
                  line_width=1.5, row=1, col=1)

    axis_kw = dict(gridcolor="#dee2e6", zerolinecolor="#adb5bd",
                   linecolor="#adb5bd", color="#1a1a2e")
    fig.update_xaxes(title_text="β = n / (10 d log p)", **axis_kw)
    fig.update_yaxes(title_text="P[success]", range=[-0.05, 1.05], row=1, col=1, **axis_kw)
    fig.update_yaxes(title_text="FP + FN edges", row=1, col=2, **axis_kw)
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14, color="#1a1a2e")),
        height=420,
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        font=dict(color="#1a1a2e", family="Inter, sans-serif", size=12),
        legend_title="Graph size",
        legend=dict(bgcolor="white", bordercolor="#dee2e6", borderwidth=1),
        margin=dict(l=60, r=30, t=55, b=55),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#1a1a2e"
        ann.font.size  = 12
    return fig


# ══════════════════════════════════════════════════════════════
#  Graph visualisation (matplotlib / networkx, light theme)
# ══════════════════════════════════════════════════════════════

_NODE_COLOR  = "#4C72B0"
_HUB_COLOR   = "#e377c2"
_TP_COLOR    = "#2ca02c"
_FP_COLOR    = "#d62728"
_FN_COLOR    = "#ff7f0e"

_LEGEND_HANDLES = [
    mpatches.Patch(color=_TP_COLOR, label="True positive"),
    mpatches.Patch(color=_FP_COLOR, label="False positive (spurious)"),
    mpatches.Patch(color=_FN_COLOR, label="False negative (missed)"),
]


def _edge_partition(true_set: set, est_set: set):
    return true_set & est_set, est_set - true_set, true_set - est_set


def draw_grid_graph(p_side: int, true_set: set, est_set: set, ax, title: str = ""):
    p   = p_side * p_side
    pos = {i * p_side + j: (j, -i) for i in range(p_side) for j in range(p_side)}
    G   = nx.Graph()
    G.add_nodes_from(range(p))
    tp, fp, fn = _edge_partition(true_set, est_set)
    for s, t in tp: G.add_edge(s, t, et="tp")
    for s, t in fp: G.add_edge(s, t, et="fp")
    for s, t in fn: G.add_edge(s, t, et="fn")
    tp_e = [(s, t) for s, t, d in G.edges(data=True) if d["et"] == "tp"]
    fp_e = [(s, t) for s, t, d in G.edges(data=True) if d["et"] == "fp"]
    fn_e = [(s, t) for s, t, d in G.edges(data=True) if d["et"] == "fn"]
    ax.set_facecolor("white")
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=_NODE_COLOR,
                           linewidths=1, edgecolors="white", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=tp_e,
                           edge_color=_TP_COLOR, width=2.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=fp_e,
                           edge_color=_FP_COLOR, width=2.5, style="dashed", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=fn_e,
                           edge_color=_FN_COLOR, width=2.5, style="dotted", ax=ax)
    ax.set_title(title, fontsize=9, color="#1a1a2e", pad=6)
    ax.axis("off")


def draw_star_graph(true_set: set, est_set: set, ax, p: int, title: str = ""):
    G = nx.Graph()
    G.add_nodes_from(range(p))
    for s, t in (true_set | est_set):
        G.add_edge(s, t)
    pos    = nx.spring_layout(G, seed=42, k=1.5)
    pos[0] = np.array([0.0, 0.0])
    tp, fp, fn = _edge_partition(true_set, est_set)
    ax.set_facecolor("white")
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color=_HUB_COLOR,
                           node_size=600, linewidths=1, edgecolors="white", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=list(range(1, p)),
                           node_color=_NODE_COLOR, node_size=100,
                           linewidths=1, edgecolors="white", ax=ax)
    if tp: nx.draw_networkx_edges(G, pos, edgelist=list(tp),
                                  edge_color=_TP_COLOR, width=2.5, ax=ax)
    if fp: nx.draw_networkx_edges(G, pos, edgelist=list(fp),
                                  edge_color=_FP_COLOR, width=2.5, style="dashed", ax=ax)
    if fn: nx.draw_networkx_edges(G, pos, edgelist=list(fn),
                                  edge_color=_FN_COLOR, width=2.5, style="dotted", ax=ax)
    ax.set_title(title, fontsize=9, color="#1a1a2e", pad=6)
    ax.axis("off")


def make_graph_preview(is_star: bool, p_side_or_p, edges, theta,
                       d, coupling_type, beta_mid) -> plt.Figure:
    """Single-trial true-vs-estimated graph preview (light background)."""
    if is_star:
        p      = p_side_or_p
        n      = max(30, int(np.ceil(beta_mid * 10 * d * np.log(p))))
        lam    = np.sqrt(np.log(p) / n)
        X      = exact_sample_star(theta, n, seed=42)
    else:
        p_side = p_side_or_p
        p      = p_side * p_side
        burnin = max(300, p * 5)
        n      = max(30, int(np.ceil(beta_mid * 10 * d * np.log(p))))
        lam    = np.sqrt(np.log(p) / n)
        X      = gibbs_sample_ising(theta, n, n_burnin=burnin, seed=42)

    true_set  = {(min(s, t), max(s, t)) for s, t in edges}
    est_set   = estimate_graph(X, lam)
    tp, fp, fn = _edge_partition(true_set, est_set)
    ok_str    = "✓ SUCCESS" if est_set == true_set else f"✗  FP={len(fp)}  FN={len(fn)}"

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor("white")

    if is_star:
        draw_star_graph(true_set, est_set, ax, p,
                        title=f"n={n}   β={beta_mid:.2f}   {ok_str}")
        ax.legend(
            handles=[mpatches.Patch(color=_HUB_COLOR, label="Hub")] + _LEGEND_HANDLES,
            loc="upper right", fontsize=6.5,
            facecolor="white", edgecolor="#dee2e6", labelcolor="#1a1a2e",
        )
    else:
        draw_grid_graph(p_side_or_p, true_set, est_set, ax,
                        title=f"n={n}   β={beta_mid:.2f}   {ok_str}")
        ax.legend(handles=_LEGEND_HANDLES, loc="upper right", fontsize=6.5,
                  facecolor="white", edgecolor="#dee2e6", labelcolor="#1a1a2e")

    fig.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════════════════════════════════════
#  Results summary table
# ══════════════════════════════════════════════════════════════

def build_summary_table(all_results: dict, n_trials: int) -> pd.DataFrame:
    rows = []
    for label, vals in all_results.items():
        betas, succs = vals[0], vals[1]
        max_succ = max(succs)
        cross    = None
        for i in range(len(succs) - 1):
            if succs[i] < 0.5 <= succs[i + 1]:
                frac  = (0.5 - succs[i]) / max(succs[i + 1] - succs[i], 1e-9)
                cross = betas[i] + frac * (betas[i + 1] - betas[i])
                break
        rows.append({
            "Graph":              label,
            "β range":            f"[{betas[0]:.2f}, {betas[-1]:.2f}]",
            "Trials / β":         n_trials,
            "Max P[success]":     f"{max_succ:.2f}",
            "β at 50% threshold": f"{cross:.2f}" if cross else "—",
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  Streamlit app
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Ising Model Selection",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light mode override (in case system is dark)
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: white; }
  [data-testid="stSidebar"]          { background: #f8f9fa; }
  h1, h2, h3, p, li, label           { color: #1a1a2e !important; }
  [data-testid="stMetricLabel"]      { color: #6c757d !important; }
</style>
""", unsafe_allow_html=True)

st.title("🔗 Ising Model Selection via ℓ₁-Regularized Logistic Regression")
st.markdown(
    "Reproduces experiments of **Ravikumar, Wainwright & Lafferty (2010)**, "
    "*Ann. Statist.* 38(3): 1287–1319. "
    "The graph of a binary Ising MRF is recovered by running **node-wise "
    "ℓ₁-penalised logistic regression** at each node and symmetrising with the **AND rule**. "
    "Key prediction: success probability transitions 0→1 as β = n/(10 d log p) grows, "
    "with curves for all p collapsing onto one universal function."
)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Experiment Settings")

    exp_type = st.selectbox(
        "Experiment type",
        ["4-NN Grid (Fig. 2)", "8-NN Grid (Fig. 3)", "Star Graph (Fig. 4/5)"],
    )
    is_star = "Star" in exp_type
    conn    = 8 if "8-NN" in exp_type else 4

    st.subheader("Graph sizes")
    if is_star:
        p_choices   = st.multiselect(
            "Nodes p", [20, 40, 64, 100, 225], default=[20, 40, 64],
        )
        degree_type = st.selectbox(
            "Hub degree growth", ["logarithmic", "linear"],
            help="logarithmic: d=⌈log p⌉ · linear: d=⌈0.1 p⌉",
        )
    else:
        # Smaller defaults so transition is visible in a reasonable β range
        side_choices  = st.multiselect(
            "Grid side √p", [4, 8, 10, 15], default=[8, 10, 15],
            help="p = side². Tip: keep side ≤ 6 for visible transitions with c = 1.",
        )
        coupling_type = st.selectbox(
            "Coupling type", ["mixed", "attractive"],
            help="mixed: θ=±ω · attractive: θ=+ω",
        )

    omega = st.slider("|ω| edge weight", 0.10, 1.0,
                      0.25 if "8-NN" in exp_type else 0.50, 0.05)

    st.subheader("β sweep")
    beta_min = st.slider("β min",         0.05, 1.0,  0.20, 0.05)
    # Sensible β max: star saturates by ~3, grid needs up to ~5
    beta_max = st.slider("β max",         1.0,  8.0,
                         3.0 if is_star else 5.0, 0.25)
    n_betas  = st.slider("# grid points", 5, 20, 10)
    n_trials = st.slider("Trials per β",  5, 100, 20)

    st.divider()
    st.markdown(
        r"""
**Regularisation** (c = 1, fixed)

$$\lambda_n = \sqrt{\frac{\log p}{n}}, \quad C_{\text{sk}} = \frac{1}{\sqrt{n \log p}}$$

**AND rule**: edge *(s,t)* included iff  
*t* ∈ N̂(s)  **and**  *s* ∈ N̂(t).

**Burnin**: max(300, 5p) steps for grids.
"""
    )
    run_btn = st.button("▶ Run Experiment", type="primary", use_container_width=True)

beta_values = np.linspace(beta_min, beta_max, n_betas)

# ── Theory expander ──────────────────────────────────────────
with st.expander("📐 Theory — Theorem 1 (click to expand)"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(r"""
**Ising model**
$$\mathbb{P}_{\theta^*}(x) \propto \exp\!\Bigl(\sum_{(s,t)\in E}\theta^*_{st}x_s x_t\Bigr),\quad x\in\{-1,+1\}^p$$

**Gibbs conditional**
$$\mathbb{P}(x_r=+1\mid x_{-r}) = \sigma\!\Bigl(2\!\sum_{t\in N(r)}\theta^*_{rt}x_t\Bigr)$$

**Estimator** (solved per node *r*)
$$\hat\theta = \underset{\theta}{\arg\min}\;
-\tfrac{1}{n}\!\sum_i \log \mathbb{P}_\theta(x^{(i)}_r\mid x^{(i)}_{-r}) + \lambda_n\|\theta\|_1$$
""")
    with c2:
        st.markdown(r"""
**Estimated graph**
$$\hat N(r) = \{t : \hat\theta_{rt}\neq 0\}, \quad \text{success}=\mathbf{1}[\hat N(r)=N(r)\;\forall r]$$

**Main result** — under mutual incoherence on $Q^*$:
$$n = \Omega(d^3\!\log p) \;\Longrightarrow\; \Pr[\text{success}]\to 1$$

**Control parameter**
$$\beta = \frac{n}{10\,d\,\log p}$$
All curves collapse when plotted vs. β.
""")

# ── Main content ─────────────────────────────────────────────
col_plot, col_graph = st.columns([3, 2])
with col_plot:
    chart_ph = st.empty()
with col_graph:
    graph_ph = st.empty()
    graph_ph.info("📊 Graph preview will appear after running.")

status_ph   = st.empty()
progress_ph = st.empty()

# ── Run ──────────────────────────────────────────────────────
if run_btn:
    if is_star and not p_choices:
        st.warning("Select at least one graph size p.")
        st.stop()
    if not is_star and not side_choices:
        st.warning("Select at least one grid side length.")
        st.stop()

    pb          = progress_ph.progress(0.0)
    t0          = time.time()
    all_results = {}

    # ── Grid ─────────────────────────────────────────────────
    if not is_star:
        for p_side in sorted(side_choices):
            p_val = p_side ** 2
            label = f"p = {p_val}  (d = {conn})"
            status_ph.info(f"Running {p_side}×{p_side} grid (p = {p_val}) …")
            betas, succs = run_grid_experiment(
                p_side, conn, coupling_type, omega,
                beta_values, n_trials, progress_bar=pb,
            )
            all_results[label] = (betas, succs)

        conn_str = "4-NN" if conn == 4 else "8-NN"
        title    = f"{conn_str} Grid — {coupling_type.capitalize()} couplings, ω = {omega}"
        chart_ph.plotly_chart(
            plotly_phase_transition(all_results, title),
            use_container_width=True,
        )

        p_side_ex  = sorted(side_choices)[0]
        p_ex, edges_ex = build_grid_graph(p_side_ex, conn)
        theta_ex   = make_theta_matrix(p_ex, edges_ex, omega, coupling_type)
        beta_mid   = float(np.median(beta_values))
        fig_prev   = make_graph_preview(
            False, p_side_ex, edges_ex, theta_ex, conn, coupling_type, beta_mid,
        )
        graph_ph.pyplot(fig_prev)
        plt.close(fig_prev)

    # ── Star ─────────────────────────────────────────────────
    else:
        for p_val in sorted(p_choices):
            _, _, d_val = build_star_graph(p_val, degree_type)
            label = f"p = {p_val}  (d = {d_val})"
            status_ph.info(f"Running star p = {p_val}, d = {d_val} ({degree_type}) …")
            betas, succs, disagrees = run_star_experiment(
                p_val, degree_type, omega,
                beta_values, n_trials, progress_bar=pb,
            )
            all_results[label] = (betas, succs, disagrees)

        title = f"Star Graph — {degree_type.capitalize()} degree, ω = {omega}"
        chart_ph.plotly_chart(
            plotly_star_dual(all_results, title),
            use_container_width=True,
        )

        p_ex              = sorted(p_choices)[0]
        _, edges_ex, d_ex = build_star_graph(p_ex, degree_type)
        theta_ex          = make_theta_matrix(p_ex, edges_ex, omega, "attractive")
        beta_mid          = float(np.median(beta_values))
        fig_prev          = make_graph_preview(
            True, p_ex, edges_ex, theta_ex, d_ex, "attractive", beta_mid,
        )
        graph_ph.pyplot(fig_prev)
        plt.close(fig_prev)

    # ── Done ─────────────────────────────────────────────────
    pb.empty()
    status_ph.success(f"✅  Done in {time.time() - t0:.1f} s")

    st.subheader("Results summary")
    st.dataframe(build_summary_table(all_results, n_trials), use_container_width=True)

    st.markdown("""
**Interpreting the results**
- **Curves aligning** across different p values confirms that n/(d log p) is the governing ratio (Theorem 1).
- The **β at 50% threshold** should be similar across graph sizes — divergence suggests the β range needs extending.
- **Star graphs** use exact ancestral sampling and show the cleanest transitions.
- **Grid models**: burnin scales as 5p to ensure adequate Gibbs mixing. Mixed couplings frustrate the lattice and mix faster than purely attractive ones.
- With c = 1, the grid transition appears in the β = 2–5 range for p = 16–36. Larger grids need wider β sweeps.
""")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Method: node-wise ℓ₁-logistic regression (liblinear, no intercept) + AND-rule symmetrisation · "
    "λₙ = √(log p / n), c = 1 · "
    "Gibbs sampling for grids (burnin = max(300, 5p)); exact ancestral sampling for stars · "
    "Ravikumar, Wainwright & Lafferty (2010), Ann. Statist. 38(3):1287–1319"
)