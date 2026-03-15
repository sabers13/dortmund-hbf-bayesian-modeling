# Bayesian Modeling of Deutsche Bahn Operations at Dortmund Hbf (December 2025)

Bayesian project on railway operational reliability at **Dortmund Hbf** using **December 2025** event data.

The project studies three related prediction problems:

* **Story 1:** cancellation incidence
* **Story 2:** arrival delay incidence + positive-delay magnitude
* **Story 3:** departure delay incidence + positive-delay magnitude

The final report version is based on **Phase 2**, where the primary comparison is between:

* **event-level Bernoulli incidence models**
* **grouped Binomial incidence benchmarks**

with **`train_cat`** treated as the **exchangeable multilevel grouping factor**.

Positive-delay **log-normal** models are retained as **support models** for non-zero delay magnitude.

---

## Project status

* **Presentation:** based on **Phase 1** (legacy version)
* **Report:** based on **Phase 2** (revised version)

### Phase 1

Phase 1 focused mainly on comparing a simpler baseline structure with a hierarchical / multilevel version within the same general modeling family.

### Phase 2

Phase 2 keeps the same dataset and story lines, but changes the main comparison axis to a clearer methodological contrast:

* **event-level Bernoulli**
* **grouped Binomial**

This revision was made to better align the project with the course requirement of comparing **qualitatively distinct Bayesian approaches** and to make **exchangeability / partial pooling** more explicit through `train_cat`.

---

## Main result

Across all three incidence stories, the **event-level Bernoulli** formulation performs better than the grouped Binomial benchmark on **common held-out event sets**.

Summary of the final comparison:

* **Story 1:** Bernoulli clearly better
* **Story 2:** Bernoulli moderately better
* **Story 3:** Bernoulli slightly better, but still preferred

Model comparison is based primarily on:

* **held-out Brier score**
* **held-out log loss**

Direct Bernoulli-vs-Binomial PSIS-LOO ranking is **not** used as the main decision rule, because the two formulations use **different observation units**.

---

## Research questions

1. How well can we model **cancellation incidence** at the event level?
2. How well can we model **arrival delay incidence** and the **magnitude of positive arrival delays**?
3. How well can we model **departure delay incidence** and the **magnitude of positive departure delays**?
4. For binary incidence outcomes, is **event-level Bernoulli** preferable to a **grouped Binomial** benchmark when both are evaluated on the same held-out events?
5. How does the exchangeable grouping factor **`train_cat`** influence inference and prediction?

---

## Data

The dataset consists of operational railway events at **Dortmund Hauptbahnhof (Dortmund Hbf)** during **December 2025**.

### Unit of analysis

One **train stop event**.

### Summary

* **Total events:** 27,201
* **Arrival rows:** 16,205
* **Departure rows:** 16,152
* **Cancellation rate:** 9.2%
* **Delay incidence rate (> 0 min):** 72.8%
* **Median positive delay:** 5.0 min
* **Mean positive delay:** 9.8 min

### Key predictors

* weekday
* hour / time-of-day features
* day of month
* days until Christmas
* destination bucket
* train category / train type
* **`train_cat`** (exchangeable grouping factor)

---

## Modeling strategy

### Incidence models

#### 1. Event-level Bernoulli

For binary outcomes at the original event level:

* cancellation incidence
* arrival delay incidence
* departure delay incidence

General form:

```math
 y_i \sim \mathrm{Bernoulli}(p_i), \qquad \mathrm{logit}(p_i) = \eta_i
```

#### 2. Grouped Binomial

For grouped binary counts as a coarser benchmark:

```math
 y_g \sim \mathrm{Binomial}(n_g, p_g), \qquad \mathrm{logit}(p_g) = \eta_g
```

These models target the same substantive incidence questions, but use **aggregated observation units**.

### Support models for positive delays

#### 3. Log-normal positive-delay magnitude models

For Stories 2 and 3, conditional on delay > 0:

* positive arrival-delay magnitude
* positive departure-delay magnitude

Modeled on the log scale.

---

## Exchangeability and multilevel structure

A key part of the revised project is making the multilevel structure explicit.

The factor **`train_cat`** is treated as an **exchangeable grouping factor**, which means:

* train categories are modeled as related rather than fully independent
* partial pooling shares information across categories
* posterior group effects and category-level probabilities become interpretable

This is one of the central methodological improvements of Phase 2 over the earlier presentation framing.

---

## Priors

All fitted models use **explicit proper priors**.

The project avoids:

* flat priors
* hidden software defaults

General prior design:

* regression effects: weakly informative, centered at 0
* intercepts: story-specific, scale-aware
* group-level standard deviations: positive regularizing priors
* residual scale for log-magnitude models: positive regularizing prior

Prior predictive checks are included for the main incidence models.

---

## Diagnostics and checks

The project uses a full Bayesian workflow:

* prior predictive checks
* posterior predictive checks
* convergence diagnostics
* calibration checks
* held-out predictive comparison
* PSIS-LOO summaries for per-model stability checks

### Convergence diagnostics

Checked with:

* trace plots
* (\hat{R})
* effective sample size (ESS)
* divergences

### Predictive checks

* incidence PPCs
* magnitude-model PPCs
* calibration curves
* story-specific `train_cat` effect plots
* posterior category probability plots

---

## Why PSIS-LOO is not the main Bernoulli-vs-Binomial comparison

The Bernoulli and grouped Binomial incidence models use **different observation units**:

* Bernoulli: individual events
* Binomial: grouped counts

Because of that, direct Bernoulli-vs-Binomial PSIS-LOO ranking is **not the main comparison method**.

Instead, the project compares them on:

* the **same held-out event sets**
* using **Brier score** and **log loss**

PSIS-LOO is still computed **within each fitted model** to assess the numerical stability of the LOO estimates.

---

## Repository structure

Example high-level structure:

```text
.
├── run_project.py
├── pipeline.py
├── models.py
├── config.py
├── data/
├── outputs/
│   ├── eda/
│   ├── loo/
│   ├── models/
│   ├── ppc/
│   ├── report_bundle/
│   └── tables/
├── report/
│   ├── report.tex
│   └── report.pdf
└── README.md
```

### Important outputs

* `outputs/eda/`
  Exploratory plots

* `outputs/ppc/`
  Prior predictive checks, posterior predictive checks, calibration, holdout comparison plots

* `outputs/loo/`
  PSIS-LOO summaries and Pareto-k diagnostics

* `outputs/models/`
  Saved fitted models and trace plots

* `outputs/report_bundle/`
  Report-facing summaries and selected story artifacts

* `outputs/tables/`
  CSV summaries, diagnostics, calibration tables, holdout summaries

---

## How to run

### 1. Install dependencies

Create a virtual environment and install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the pipeline

Typical full run:

```bash
python run_project.py \
  --data-path data/Dortmund_HBF_December.csv \
  --output-dir outputs \
  --draws 2000 \
  --tune 2000 \
  --chains 4 \
  --target-accept 0.95
```

If your CLI arguments differ, adapt them to your local setup.

### 3. Inspect outputs

Main report-facing artifacts are typically found under:

```text
outputs/ppc/
outputs/report_bundle/
outputs/tables/
outputs/loo/
```

---

## Reproducibility notes

The pipeline is designed to run the revised **Phase 2** analysis end to end:

1. load and clean the dataset
2. prepare event-level and grouped data
3. fit all incidence and support models
4. produce diagnostics and predictive checks
5. export report-facing figures and tables

Sampling settings for the final report run:

* **chains:** 4
* **posterior draws:** 2000 per chain
* **tuning iterations:** 2000 per chain
* **target acceptance:** 0.95

---

## Report contents

The accompanying report is organized around:

* Introduction
* Data
* Models
* Priors
* Code and implementation
* Convergence diagnostics
* Model comparison
* Model checks and predictive performance
* Limitations and potential improvements
* Conclusion
* Reflection on own learnings
* Appendix

The appendix contains lower-priority supporting material such as:

* individual trace plots
* additional PPCs
* additional calibration figures
* extra exchangeability diagnostics

---

## Limitations

Important limitations of the current Phase 2 analysis:

* grouped Binomial models are intentionally coarser than event-level Bernoulli models
* the positive-delay log-normal models are support models, not the main comparison axis
* predictive evaluation is based on a single held-out split rather than repeated blocked validation
* some categorical structure is simplified for tractability

These do not change the main conclusion, but they define the scope of the claims.

---

## Suggested future work

* repeated blocked holdout evaluation
* richer temporal structure
* more detailed destination modeling
* tail-focused checks for positive-delay magnitude models
* additional robustness / sensitivity analysis

---

## Citation / usage

If you use or adapt this project, cite the report or repository appropriately.

Suggested citation format:

```text
Author(s). Bayesian Modeling of Deutsche Bahn Operations at Dortmund Hbf (December 2025). GitHub repository / course project report.
```

---

## Notes

* **Presentation** corresponds to **Phase 1**
* **Final report** corresponds to **Phase 2**
* The report should be treated as the authoritative final version of the project

---
