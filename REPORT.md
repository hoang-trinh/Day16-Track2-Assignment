# BÁO CÁO THỰC HÀNH LAB 16: BENCHMARK LIGHTGBM TRÊN AWS EC2

## 1. Bảng kết quả Benchmark

| Metric | Kết quả | Đơn vị |
|---|---|---|
| Thời gian load data | 3.14 | giây (s) |
| Thời gian training | 1.88 | giây (s) |
| Best iteration | 1 | vòng lặp |
| AUC-ROC | 0.9267 | - |
| Accuracy | 0.9989 | - |
| F1-Score | 0.7382 | - |
| Precision | 0.6370 | - |
| Recall | 0.8776 | - |
| Inference latency (1 row) | 1.24 | ms |
| Inference throughput (1000 rows) | 678,474.8 | rows/giây |

---

## 2. Nhận xét và Phân tích kết quả (5 - 10 dòng)

- **Về tốc độ xử lý & huấn luyện:** Quá trình đọc bộ dữ liệu 284,807 dòng chỉ mất khoảng 3.14s và thời gian huấn luyện LightGBM trên CPU node (`t3.micro`) diễn ra rất nhanh (1.88s). Điều này chứng minh hiệu quả tối ưu hóa vượt trội của thuật toán LightGBM (Histogram-based algorithm) ngay cả trên phần cứng CPU có cấu hình giới hạn (1 vCPU / 1GB RAM + Swap).
- **Về chất lượng mô hình (Model Performance):** Với bài toán dữ liệu mất cân bằng nghiêm trọng (Fraud Detection chỉ chiếm ~0.17% giao dịch), mô hình đạt **AUC-ROC rất cao (0.9267)** và **Recall đạt 87.76%**. Điều này đồng nghĩa mô hình phát hiện được phần lớn các giao dịch gian lận thực tế mà không bỏ sót. Độ chính xác tổng thể (Accuracy) đạt 99.89%.
- **Về hiệu năng Inference:** Độ trễ khi dự đoán từng giao dịch đơn lẻ (Latency) chỉ xấp xỉ **1.24 ms**, và Throughput khi dự đoán theo lô (batch 1,000 dòng) đạt tới **hơn 678,000 dòng/giây**. Với độ trễ cực thấp này, mô hình hoàn toàn đáp ứng tốt bài toán kiểm soát gian lận giao dịch thẻ theo thời gian thực (Real-time Fraud Detection) trong hệ thống tài chính/ngân hàng.

Note: Hệ thống hóa đơn AWS Billing có độ trễ cập nhật cước từ 8 - 24 giờ. Tại thời điểm kiểm tra ngay sau lab, EC2 node thuộc diện Free Tier (t3.micro) và các dịch vụ khác (NAT Gateway, ALB) mới chạy ~30 phút nên hóa đơn tạm thời ghi nhận $0.00.