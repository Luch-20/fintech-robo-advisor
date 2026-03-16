# AI-Powered Fintech Robo-Advisor for Personal Portfolio Optimization

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg)

## 📌 Overview

This project implements an intelligent Robo-advisor system designed to optimize personal investment portfolios. By leveraging cutting-edge Artificial Intelligence algorithms, it bridges the gap between individual risk tolerance and optimal market performance. 

The system utilizes a hybrid approach combining two core algorithms:
1. **Inverse Portfolio Optimization (IPO)**: Infers the investor's implicit risk appetite based on their current asset allocation.
2. **Deep Reinforcement Learning (DDPG)**: Dynamically optimizes multi-period capital allocation to maximize risk-adjusted returns over time.

Designed specifically for the Vietnamese stock market (VN-Index), the current model is trained and evaluated on 30 highly liquid, large-cap stocks.

---

## 🏗 System Architecture

### 1. Data Acquisition & Preprocessing
* **Sources**: Yahoo Finance and `vnstock`.
* **Features**: Daily Closing Prices, Daily Log Returns, Trading Volumes, and Technical Indicators.
* **Processing**: Built-in handling for missing data, outliers, and a rolling window of 126 days (approx. 6 months).

### 2. Inverse Portfolio Optimization (IPO)
Rather than relying on subjective questionnaires, the system *learns* the investor's risk-aversion coefficient ($\lambda$) and expected returns ($\mu$) directly from their existing portfolio distribution using Mean-Variance Optimization principles.

*Objective Function:*
```text
max_w [ w^T * μ_t - λ * w^T * Σ_t * w ]
s.t. Σ_i w_i = 1, w_i ≥ 0
```

### 3. Deep Deterministic Policy Gradient (DDPG)
An Actor-Critic reinforcement learning architecture is employed to handle continuous action spaces (portfolio weights).
* **State**: Historical prices, technical indicators, market news sentiment, and IPO risk parameters.
* **Action**: New weight distribution across the selected assets.
* **Reward**: Risk-adjusted return metrics, specifically prioritizing the Sharpe ratio, Sortino ratio, while explicitly penalizing maximum drawdowns.

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.7 or higher

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

### 2. Project Structure
```text
.
├── app.py                  # Flask web application entry point
├── main.py                 # CLI entry point for portfolio analysis
├── Train_Model.py          # Script to train the DDPG Agent
├── robo_agent.py           # Core implementations of IPO & DDPG algorithms
├── Get_data.py             # Data ingestion and preprocessing utilities
├── data_source.py          # APIs for fetching financial data
├── news_scraper.py         # Financial news scraping & sentiment analysis
├── data/                   # Directory for historical datasets
├── models/                 # Directory for serialized, trained models
└── templates/              # HTML/CSS for the Flask portal
```

---

## 💻 Usage Instructions

### Step 1: Training the Agent (Optional)
To train the model from scratch using the top 30 VN-Index stocks over the past 2 years:
```bash
python Train_Model.py
```
*The trained weights will be saved automatically to `models/trained_model.pth`.*

### Step 2: Running the Application

**Option A: Web Application (Recommended)**
Provides an interactive GUI to input current portfolio holdings and view the AI's recommendations.
```bash
python app.py
```
*Navigate to `http://localhost:5000` (or `http://localhost:5001` depending on port availability).*

**Option B: Command Line Interface (CLI)**
For programmatic usage or terminal-based analysis:
```bash
python main.py
```

---

## 📊 Evaluation Metrics

The agent's performance is heavily monitored using industry-standard financial metrics:
1. **Annualized Mean Return** 
2. **Standard Deviation (Volatility)**
3. **Sharpe Ratio** (Assumes a 4.5% risk-free rate based on VN Government Bonds)
4. **Portfolio Turnover Rate**
5. **Cumulative Transaction Costs** (Modeled at 0.3% per trade)
6. **Maximum Drawdown (MDD)**

---

## 📚 References & Literature
* Wang & Yu (2021) - *"Robo-Advisor using Inverse Portfolio Optimization and Deep Reinforcement Learning"*
* Markowitz (1952) - *"Portfolio Selection"*
* Lillicrap et. al (2015) - *"Continuous control with deep reinforcement learning"*

---

## 📄 License
This project is developed for academic and research purposes.
