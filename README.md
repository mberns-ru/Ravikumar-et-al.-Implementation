# 🔗 Ising Model Selection via ℓ₁-Regularized Logistic Regression

This project reproduces key experiments from:

> **Ravikumar, Wainwright & Lafferty (2010)**  
> *High-Dimensional Ising Model Selection Using ℓ₁-Regularized Logistic Regression*  
> Annals of Statistics, 38(3), 1287–1319

It provides an **interactive Streamlit app** to explore how well ℓ₁-regularized logistic regression recovers the structure of an Ising model under different graph topologies and parameters.

---

## 🚀 Features

- 📊 **Reproduces phase transition results** from the paper  
- 🔗 Supports multiple graph types:
  - 4-NN grid
  - 8-NN grid
  - Star graphs
- ⚙️ Fully interactive controls:
  - Graph size
  - Edge weight (ω)
  - Coupling type (mixed vs attractive)
  - Number of samples (via β scaling)
- 📈 Visualizations:
  - Success probability vs β (Plotly)
  - Edge recovery diagnostics
  - Graph structure comparison (matplotlib + networkx)
- 🧠 Implements:
  - Gibbs sampling (for grids)
  - Exact sampling (for star graphs)
  - Node-wise ℓ₁ logistic regression (liblinear)
  - AND-rule graph reconstruction

---

## 📐 Key Idea

The probability of correctly recovering the graph structure depends on:

\[
\beta = \frac{n}{10 \, d \, \log p}
\]

- \( n \): number of samples  
- \( d \): maximum node degree  
- \( p \): number of nodes  

The paper shows that **success probability transitions sharply from 0 → 1 as β increases**, and curves align across different graph sizes.

---

## 🛠 Installation

Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
````

---

## ▶️ Running the App

```bash
streamlit run ising_model_selection.py
```

Then open the provided local URL in your browser.

---

## 🧪 Experiments Included

### Grid Graphs

* 4-nearest neighbor (Fig. 2)
* 8-nearest neighbor (Fig. 3)
* Uses **Gibbs sampling** (with burn-in scaling with p)

### Star Graphs

* Logarithmic or linear degree growth
* Uses **exact ancestral sampling**
* Produces cleaner phase transitions

---

## 📊 Output

* **Phase transition plots**: success probability vs β
* **Graph recovery visualization**:

  * True positives (green)
  * False positives (red)
  * False negatives (orange)
* **Summary table**:

  * Max success probability
  * β threshold (~50% success)

---

## 🧩 Implementation Details

* Logistic regression:

  * ℓ₁ penalty
  * `liblinear` solver
  * No intercept (zero-field Ising model)
* Regularization:

[
\lambda_n = \sqrt{\frac{\log p}{n}}, \quad C = \frac{1}{\sqrt{n \log p}}
]

* Graph reconstruction via **AND rule**:

  * Edge (s, t) exists iff:

    * ( t \in \hat{N}(s) ) AND ( s \in \hat{N}(t) )

---

## 📁 File Structure

```
ising_model_selection.py   # Main Streamlit app
README.md                  # Project documentation
requirements.txt           # Dependencies
```

---

## 📚 Reference

Ravikumar, P., Wainwright, M. J., & Lafferty, J. D. (2010).
High-dimensional Ising model selection using ℓ₁-regularized logistic regression.
*Annals of Statistics*, 38(3), 1287–1319.

---

## 💡 Notes

* Grid models may require larger β ranges due to slower mixing.
* Mixed couplings typically mix faster than purely attractive ones.
* Star graphs provide the cleanest demonstration of theoretical predictions.

---

## 🙌 Acknowledgments

This app is built as an interactive reproduction of a classic result in graphical model selection and high-dimensional statistics.