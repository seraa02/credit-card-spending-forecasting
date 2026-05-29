💳 Credit Card Spending Forecasting & Budget Recommendations

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?logo=pytorch)
![Prophet](https://img.shields.io/badge/Meta-Prophet-blue)
![Course](https://img.shields.io/badge/STAT%20654-Statistical%20Computing%20with%20R%20%26%20Python-maroon)

📌 What This Project Does

Credit cards are useful — and easy to misuse. The average annual interest rate sits between
30–35%, and most people consistently overestimate their ability to pay off what they spend.

This project asks: *can we use someone's own transaction history to forecast their future
spending and help them course-correct before things go wrong?*

We built two time-series forecasting models — **LSTM** and **Meta's Prophet** — on over
**1 million credit card transactions** from 900+ unique customers, then layered a
**budget recommendation tool** on top of the Prophet output.

Key Results

| Model | R² Score | Notes |
|---|---|---|
| LSTM | 0.45 | Struggled with sparse holiday patterns; strong on routine spending |
| Prophet | **0.65** | Captured seasonality and holiday spikes cleanly; better for forecasting |

Prophet outperformed LSTM primarily because of its built-in handling of seasonality,
holidays, and irregular data — exactly the patterns that dominate credit card spending.

Methods

### Data & Preprocessing
- **Dataset:** 1M+ transactions, 900+ customers (Kaggle — Priyam Choksi)
- Customers filtered to 60+ transactions minimum to reduce noise
- Unique customer IDs created from First Name + Last Name + DOB
- Proxies engineered for missing income data: job title, age, city
- Time features added: weekends, US holidays, paydays (1st and 15th)
- One-hot encoding for categorical variables (job, city, day)
- Subsampling to manage compute while preserving customer uniqueness

### Model 1 — LSTM
A Recurrent Neural Network designed for long-term sequential dependencies. Three gates
(forget, input, output) control information flow through a cell state, letting the model
retain relevant spending patterns over time. Hyperparameters tuned:

| Parameter | Values Tested |
|---|---|
| Learning rate | 1e-3, 5e-4, 1e-4 |
| Hidden size | 64, 128 |
| Layers | 1, 2 |
| Dropout | 0.0, 0.2 |

### Model 2 — Prophet
Facebook/Meta's open-source forecasting tool using an additive decomposition:

`y(t) = g(t) + s(t) + h(t) + εt`

Where g(t) is trend, s(t) is seasonality (Fourier series), h(t) captures holiday effects,
and εt is noise. Prophet handles missing data and irregular intervals natively — no
additional preprocessing required.

### Budget Recommendation Tool
A prototype built on top of Prophet forecasts. Users input monthly take-home pay,
desired savings %, and desired investing %. The tool outputs a category-by-category
budget breakdown including rent, groceries, utilities, savings, investing, and
discretionary spending.

📊 Descriptive Insights

A few patterns that came out of the EDA and shaped the modeling:

- **December 2019** had significantly higher spending than any other month — holiday
  season + end-of-year bonuses
- **Monday and Sunday** were the highest-spending days — possibly linked to
  stress-driven spending at the Sunday/Monday transition (supported by behavioral
  economics literature)
- **Texas** had the highest total spending despite New York having more customers —
  three Texas cities landed in the top 10 spending cities
- **Summer 2019** showed a mild uptick, consistent with seasonal travel patterns

Tech Stack

- **Python** — core language
- **PyTorch** — LSTM implementation
- **Prophet (Meta)** — time-series forecasting
- **pandas / NumPy** — data processing and feature engineering
- **matplotlib / seaborn** — visualization (heatmaps, bar charts, forecast plots)
- **scikit-learn** — preprocessing, evaluation metrics
- **Jupyter Notebook** — development and analysis

Run It Yourself

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-spending-forecasting.git
cd credit-card-spending-forecasting
pip install torch prophet pandas numpy matplotlib scikit-learn
jupyter notebook "Group 13_Code&Outputs.ipynb"
```

Or run the standalone script:
```bash
python Project_Code.py
```

**Data:** Download the dataset from
[Kaggle](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)
and place `credit_card_transactions.csv` in the root directory.

🔮 Future Work

- Add **income data** as a direct feature — currently proxied by job, age, and city
- Expand to **3–5 years** of transaction history for better LSTM generalization
- Build a **web-based budget recommendation app** from the Prophet + budget tool prototype
- Integrate **TensorFlow.js** for real-time browser-based forecasting
- Add **more spending categories** and customizable budget templates
- Experiment with **hybrid LSTM-Prophet** architectures

👥 Team & Contributions

| Name | Contributions |
|---|---|
| **Thinh Nguyen** | Project ideation, descriptive analysis, EDA, trend interpretation |
| **Saher Thekedar** | LSTM model development, overall code structure and implementation |
| **Akib Mahmud** | LSTM + Prophet model building, tuning, and results interpretation |
| All three | Budget recommendation system, final report, presentation |

*Texas A&M University — STAT 654: Statistical Computing with R and Python*

📚 Dataset

[Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)
by Priyam Choksi on Kaggle — 1M+ transactions, 900+ customers, Jan 2019–Jun 2020.

📄 Report

Full write-up available in `STAT_654_Group13_Project_Report.pdf` — covers methodology,
LSTM architecture with gate equations, Prophet decomposition, EDA visualizations,
model comparison, and limitations.
