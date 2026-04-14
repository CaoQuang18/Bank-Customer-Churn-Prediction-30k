# Luận văn: Hệ thống Dự báo Chấp dứt Sử dụng Dịch vụ (Churn Prediction) tích hợp XAI & Logic Kinh tế

Dự án này triển khai một hệ thống dự báo churn cho ngân hàng, tập trung vào tính toàn vẹn dữ liệu, khả năng giải thích của mô hình (Explainable AI) và phân tích hiệu quả kinh tế (ROI).

## 🚀 Tính năng nổi bật
- **Pipeline ML chuẩn nghiên cứu**: Quy trình Feature Engineering chặt chẽ, xử lý Imbalanced Data bằng SMOTE.
- **Explainable AI (XAI)**: Sử dụng SHAP (Linear & Tree Explainer) để giải nghĩa từng quyết định của mô hình.
- **ROI Analysis**: Tính toán thiệt hại tiềm năng và hiệu quả kinh tế của các chiến dịch giữ chân khách hàng.
- **Dynamic Dashboard**: Giao diện trực quan hóa toàn bộ quá trình từ EDA đến Dự báo thời gian thực.

## 🛠 Hướng dẫn cài đặt

1. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Chạy Pipeline huấn luyện**:
   ```bash
   python run_pipeline.py
   ```
   *Lưu ý: Bước này sẽ thực hiện EDA, Feature Engineering, Training Models và Phân tích SHAP.*

3. **Khởi động Dashboard**:
   ```bash
   python app.py
   ```
   *Truy cập: http://127.0.0.1:5000*

## 📊 Cấu trúc Pipeline
- `pipeline/eda.py`: Khám phá dữ liệu và trích xuất insights.
- `pipeline/feature_engineering.py`: Tiền xử lý, Scaling và SMOTE.
- `pipeline/model_training.py`: Huấn luyện LR, RF, XGB với Cross-Validation.
- `pipeline/shap_analysis.py`: Tính toán giá trị SHAP cho tính minh bạch của AI.

## 🎓 Hội đồng chấm luận văn
Dự án được thiết kế để đáp ứng các tiêu chí gắt gao về:
1. **Tính Reproducible**: Mọi bước đều được code hóa, không xử lý thủ công.
2. **Tính Stability**: Đánh giá mô hình dựa trên Mean/Std của CV AUC.
3. **Giá trị thực tiễn**: Logic phân cụm khách hàng (Clustering) phục vụ chiến dịch Marketing mục tiêu.
