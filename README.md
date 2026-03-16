# Ứng dụng Trí tuệ Nhân tạo trong Nền tảng Robo-advisor Fintech nhằm Tối ưu hóa Danh mục Đầu tư Cá nhân

## Tóm tắt

Nghiên cứu này đề xuất một hệ thống Robo-advisor sử dụng kết hợp hai thuật toán:
1. **Inverse Portfolio Optimization (IPO)** - Học khẩu vị rủi ro từ phân bổ danh mục hiện tại
2. **Deep Reinforcement Learning (DRL - DDPG)** - Tối ưu hóa phân bổ vốn đa kỳ

Hệ thống được áp dụng trên thị trường chứng khoán Việt Nam (VN-Index) với dữ liệu từ 30 mã cổ phiếu có thanh khoản tốt và vốn hóa lớn.

## Kiến trúc Hệ thống

### 1. Thu thập và Tiền xử lý Dữ liệu
- Thu thập dữ liệu giá lịch sử (Daily Closing Price) từ Yahoo Finance và vnstock
- Tính toán lợi suất logarit hàng ngày (Daily Log Returns)
- Xử lý dữ liệu thiếu và ngoại lai
- Rolling window: 126 ngày (6 tháng)

### 2. Inverse Portfolio Optimization (IPO)
- Học khẩu vị rủi ro từ phân bổ danh mục hiện tại của nhà đầu tư
- Suy luận hệ số rủi ro (risk-aversion coefficient λ) và lợi suất kỳ vọng (expected returns μ)
- Sử dụng Mean-Variance Optimization để tính optimal weights từ historical data

**Công thức:**
```
max_w w^T * μ_t - λ * w^T * Σ_t * w
s.t. Σ_i w_i = 1, w_i ≥ 0
```

### 3. Deep Reinforcement Learning (DDPG)
- Actor-Critic architecture với DDPG algorithm
- State: Dữ liệu giá lịch sử, technical indicators, và tham số rủi ro từ IPO
- Action: Quyết định phân bổ tỷ trọng mới cho các tài sản
- Reward: Tối ưu hóa tỷ suất lợi nhuận đã điều chỉnh rủi ro (Sharpe ratio, Sortino ratio, drawdown penalty)

## Cài đặt

### Yêu cầu Hệ thống
- Python 3.7+
- Các thư viện Python (xem requirements.txt)

### Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### Cấu trúc Thư mục

```
Test_NCKH/
├── app.py                  # Flask web application
├── main.py                 # Command-line interface
├── Train_Model.py          # Script training model
├── robo_agent.py           # IPO Agent và DDPG Agent implementation
├── portfolio_advisor.py    # Portfolio advisor module
├── Get_data.py             # Data acquisition và preprocessing
├── data_source.py          # Data source handlers
├── data/
│   └── Data_test.csv       # Historical stock data
├── models/
│   └── trained_model.pth   # Trained DDPG model
├── templates/
│   └── index.html          # Web interface
├── FORMULAS.md             # Tài liệu công thức toán học
└── README.md               # File này
```

## Sử dụng

### Bước 1: Training Model

```bash
python Train_Model.py
```

Script này sẽ:
- Tải dữ liệu 30 mã cổ phiếu VN Index trong 2 năm gần nhất
- Train DDPG model với tất cả 30 mã
- Lưu model vào `models/trained_model.pth`

### Bước 2: Chạy Ứng dụng

**Option 1: Web Application (Khuyến nghị)**

```bash
python app.py
```

Mở trình duyệt tại: http://localhost:5000

**Option 2: Command Line Interface**

```bash
python main.py
```

## Các Chỉ Số Đánh Giá

Hệ thống tính toán và báo cáo các chỉ số sau:

1. **Lợi suất trung bình hàng năm (Annualized Mean Return)**
2. **Độ lệch chuẩn lợi suất (Standard Deviation)**
3. **Tỷ lệ Sharpe (Sharpe Ratio)** - với risk-free rate từ trái phiếu chính phủ VN (4.5%)
4. **Tỷ lệ Luân chuyển (Turnover Rate)**
5. **Chi phí Giao dịch Lũy kế (Cumulative Transaction Cost)** - 0.3%
6. **Mức sụt giảm tối đa (Maximum Drawdown - MDD)**

## Tham Số Hệ Thống

- **Risk-free rate**: 4.5% annual (trái phiếu chính phủ VN)
- **Transaction cost**: 0.3% (phí giao dịch thị trường VN)
- **Rolling window**: 126 ngày (6 tháng)
- **Trading days per year**: 252 ngày

## Tài liệu Tham khảo

- Wang & Yu (2021) - "Robo-Advisor using Inverse Portfolio Optimization and Deep Reinforcement Learning"
- Markowitz (1952) - "Portfolio Selection"
- Lillicrap et al. (2015) - "Continuous control with deep reinforcement learning"

## Tác giả

Nghiên cứu khoa học - Ứng dụng Trí tuệ Nhân tạo trong Nền tảng Robo-advisor Fintech

## License

Nghiên cứu khoa học - Sử dụng cho mục đích học thuật
