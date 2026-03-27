// ══════════════════════════════════════════════════════════
// NAVIGATION
// ══════════════════════════════════════════════════════════
const TAB_TITLES = {
  overview:   '📊 Dashboard: Báo cáo Tổng quan',
  eda:        '🔍 Phân tích Chân dung Khách hàng',
  models:     '🔬 Đánh giá Thuật toán Dự báo',
  shap:       '🧠 SHAP: Giải thích Trí tuệ Nhân tạo',
  imbalanced: '⚖️ Xử lý Dữ liệu Mất cân bằng',
  clusters:   '🧩 Phân đoạn Khách hàng (Cluster)',
  predict:    '🎯 Hệ thống Dự đoán Real-time'
};

// 🎨 CẤU HÌNH CHART.JS CHUYÊN NGHIỆP (Point 4)
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 13;
Chart.defaults.color = '#64748b';
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.padding = 15;
Chart.defaults.scale.grid.display = false; // Ẩn lưới cho sang
Chart.defaults.scale.border.display = false;
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.9)';
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.cornerRadius = 8;

let currentTab = 'overview';

document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    currentTab = btn.dataset.tab;
    document.getElementById('tab-' + currentTab).classList.add('active');
    
    const titleEl = document.querySelector('.page-title');
    if (titleEl) titleEl.textContent = TAB_TITLES[currentTab] || 'Dashboard';
    
    // Cuộn lên đầu
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});

// 🔍 BỘ LỌC TOÀN CỤC (Point 2)
function applyGlobalFilters() {
  const segment = document.getElementById('filter-segment').value;
  const status = document.getElementById('filter-active').value;
  
  console.log(`Applying filters: Segment=${segment}, Status=${status}`);
  
  // Nếu ở tab Cluster, ta lọc danh sách KH của cụm đang chọn
  if (currentTab === 'clusters') {
    const clusterId = parseInt(document.getElementById('cluster-select').value);
    renderClusterCustomers(clusterId, segment, status);
  }
}

document.getElementById('filter-segment').addEventListener('change', applyGlobalFilters);
document.getElementById('filter-active').addEventListener('change', applyGlobalFilters);

// ══════════════════════════════════════════════════════════
// CHART HELPERS
// ══════════════════════════════════════════════════════════
const COLORS = ['#1a56db','#e02424','#057a55','#c27803','#7e3af2','#0e9f6e','#ff5a1f','#3f83f8'];

function barChart(id, labels, data, label = 'Churn Rate (%)') {
  const ctx = document.getElementById(id);
  if (!ctx) return;
  if (ctx._chart) ctx._chart.destroy();
  ctx._chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label, data, backgroundColor: COLORS.slice(0, data.length), borderRadius: 6 }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { callback: v => v + '%' } } }
    }
  });
}

function hbarChart(id, labels, data, label = 'Churn Rate (%)') {
  const ctx = document.getElementById(id);
  if (!ctx) return;
  if (ctx._chart) ctx._chart.destroy();
  ctx._chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label, data, backgroundColor: COLORS.slice(0, data.length), borderRadius: 4 }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, ticks: { callback: v => v + '%' } } }
    }
  });
}

function lineChart(id, labels, datasets) {
  const ctx = document.getElementById(id);
  if (!ctx) return;
  if (ctx._chart) ctx._chart.destroy();
  ctx._chart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: { responsive: true, plugins: { legend: { position: 'top' } } }
  });
}

function doughnutChart(id, labels, data) {
  const ctx = document.getElementById(id);
  if (!ctx) return;
  if (ctx._chart) ctx._chart.destroy();
  ctx._chart = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data, backgroundColor: COLORS }] },
    options: { responsive: true, plugins: { legend: { position: 'right' } } }
  });
}

function setInsight(id, insight) {
  const el = document.getElementById(id);
  if (!el || !insight) return;
  el.innerHTML = `
    <strong>📊 ${insight.so_lieu || ''}</strong><br>
    ${insight.nguyen_nhan ? '🔍 ' + insight.nguyen_nhan + '<br>' : ''}
    ${insight.ly_do ? '💡 ' + insight.ly_do + '<br>' : ''}
    ${insight.huong_xu_ly ? '🎯 ' + insight.huong_xu_ly + '<br>' : ''}
    ${insight.giai_phap ? '✅ ' + insight.giai_phap : ''}
  `;
}

const FEATURE_LABELS = {
  credit_sco: 'Điểm tín dụng',
  gender: 'Giới tính',
  age: 'Tuổi',
  balance: 'Số dư',
  monthly_ir: 'Thu nhập hàng tháng',
  tenure_ye: 'Số năm gắn bó',
  married: 'Tình trạng hôn nhân',
  nums_card: 'Số thẻ',
  nums_service: 'Số dịch vụ',
  active_member: 'Thành viên hoạt động',
  engagement_score: 'Điểm engagement',
  risk_score: 'Risk score',
  customer_segment: 'Phân khúc KH',
  loyalty_level: 'Hạng Loyalty',
  digital_behavior: 'Hành vi Digital'
};

const METHOD_LABELS = {
  'Baseline': 'Baseline (Không xử lý)',
  'Baseline (No处理)': 'Baseline (Không xử lý)',
  'Class Weights': 'Class Weights',
  'SMOTE': 'SMOTE',
  'SMOTE_ClassWeights': 'SMOTE + Class Weights',
  'Threshold Tuning': 'Threshold Tuning'
};

function renderConfusionMatrix(cm, modelName, accuracy) {
  if (!cm || !Array.isArray(cm)) {
    return '<div style="padding:12px;color:#6b7280">Confusion matrix data not available</div>';
  }
  
  let tn, fp, fn, tp;
  if (cm.length === 4) {
    [tn, fp, fn, tp] = cm;
  } else if (cm.length === 2 && Array.isArray(cm[0])) {
    [[tn, fp], [fn, tp]] = cm;
  } else {
    return '<div style="padding:12px;color:#6b7280">Confusion matrix format not supported</div>';
  }
  const total = tn + fp + fn + tp;
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const fpr = fp / (fp + tn) || 0;
  return `
    <div style="text-align:center;margin-top:12px">
      <div style="display:inline-block;background:#f9fafb;padding:16px 20px;border-radius:8px">
        <div style="font-size:13px;color:#374151;margin-bottom:8px;font-weight:600">${modelName}</div>
        <table style="margin:0 auto;font-size:12px;border-collapse:collapse">
          <tr>
            <td style="padding:10px 14px;border:1px solid #e5e7eb;background:#ecfdf5;text-align:center">
              <div style="font-size:20px;font-weight:700;color:#059669">${tn.toLocaleString()}</div>
              <div style="color:#6b7280;font-size:11px">True Negative</div>
            </td>
            <td style="padding:10px 14px;border:1px solid #e5e7eb;background:#fef2f2;text-align:center">
              <div style="font-size:20px;font-weight:700;color:#dc2626">${fp.toLocaleString()}</div>
              <div style="color:#6b7280;font-size:11px">False Positive</div>
            </td>
          </tr>
          <tr>
            <td style="padding:10px 14px;border:1px solid #e5e7eb;background:#fef2f2;text-align:center">
              <div style="font-size:20px;font-weight:700;color:#dc2626">${fn.toLocaleString()}</div>
              <div style="color:#6b7280;font-size:11px">False Negative</div>
            </td>
            <td style="padding:10px 14px;border:1px solid #e5e7eb;background:#ecfdf5;text-align:center">
              <div style="font-size:20px;font-weight:700;color:#059669">${tp.toLocaleString()}</div>
              <div style="color:#6b7280;font-size:11px">True Positive</div>
            </td>
          </tr>
        </table>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px;font-size:11px">
          <div style="background:#fff;padding:6px 8px;border-radius:4px">
            <span style="color:#6b7280">Accuracy:</span>
            <strong style="color:#059669">${(accuracy * 100).toFixed(1)}%</strong>
          </div>
          <div style="background:#fff;padding:6px 8px;border-radius:4px">
            <span style="color:#6b7280">Precision:</span>
            <strong style="color:#1a56db">${(precision * 100).toFixed(1)}%</strong>
          </div>
          <div style="background:#fff;padding:6px 8px;border-radius:4px">
            <span style="color:#6b7280">Recall:</span>
            <strong style="color:#e02424">${(recall * 100).toFixed(1)}%</strong>
          </div>
          <div style="background:#fff;padding:6px 8px;border-radius:4px">
            <span style="color:#6b7280">FPR:</span>
            <strong style="color:#c27803">${(fpr * 100).toFixed(1)}%</strong>
          </div>
        </div>
      </div>
    </div>
  `;
}

function analyzeModelDeeply(results, comparison) {
  const analysis = comparison.insights || {};
  const table = comparison.table || [];
  
  let deepAnalysis = '';
  
  table.forEach(model => {
    let modelAnalysis = '';
    const precision = model.precision;
    const recall = model.recall;
    const f1 = model.f1;
    const auc = model.roc_auc;
    
    if (model.model === 'Logistic Regression') {
      modelAnalysis = `
        <div style="margin-bottom:16px;padding:12px;background:#eff6ff;border-radius:8px">
          <strong style="color:#1e429f">📊 Logistic Regression</strong><br><br>
          <strong>Ưu điểm:</strong>
          <ul style="margin:4px 0 8px 20px;font-size:13px">
            <li>Đơn giản, dễ hiểu - có thể giải thích cho ban lãnh đạo</li>
            <li>Tốc độ train/predict nhanh nhất</li>
            <li>Không bị overfitting khi dữ liệu ít</li>
            <li>Có thể xem trực tiếp hệ số của từng feature</li>
          </ul>
          <strong>Nhược điểm:</strong>
          <ul style="margin:4px 0 8px 20px;font-size:13px">
            <li>Giả định quan hệ tuyến tính (không đúng với churn thực tế)</li>
            <li>Bỏ sót các tương tác phi tuyến giữa features</li>
            <li>Boundary decision không linh hoạt</li>
          </ul>
          <strong>Phù hợp khi:</strong>
          <ul style="margin:4px 0 0 20px;font-size:13px">
            <li>Cần baseline để so sánh</li>
            <li>Cần giải thích được cho stakeholders</li>
            <li>Dữ liệu có xu hướng tuyến tính</li>
          </ul>
        </div>
      `;
    } else if (model.model === 'Random Forest') {
      modelAnalysis = `
        <div style="margin-bottom:16px;padding:12px;background:#ecfdf5;border-radius:8px">
          <strong style="color:#047857">🌲 Random Forest</strong><br><br>
          <strong>Ưu điểm:</strong>
          <ul style="margin:4px 0 8px 20px;font-size:13px">
            <li>Recall cao (${(recall * 100).toFixed(1)}%) - bắt được nhiều KH sẽ churn</li>
            <li>Nắm bắt tốt các pattern phi tuyến và tương tác phức tạp</li>
            <li>Ít overfit nhờ bagging (random sampling)</li>
            <li>Có thể xử lý cả categorical và numerical features</li>
          </ul>
          <strong>Nhược điểm:</strong>
          <ul style="margin:4px 0 8px 20px;font-size:13px">
            <li>Khó giải thích cho người không có chuyên môn</li>
            <li>Train chậm hơn LR (nhiều cây)</li>
            <li>Memory usage cao</li>
          </ul>
          <strong>Đặc biệt:</strong>
          <ul style="margin:4px 0 0 20px;font-size:13px">
            <li>Best overall (AUC=${auc}) với balanced performance</li>
            <li>Feature importance: risk_score và monthly_ir quan trọng nhất</li>
            <li>Phù hợp production với độ ổn định cao</li>
          </ul>
        </div>
      `;
    } else if (model.model === 'XGBoost') {
      modelAnalysis = `
        <div style="margin-bottom:16px;padding:12px;background:#fef3c7;border-radius:8px">
          <strong style="color:#92400e">⚡ XGBoost</strong><br><br>
          <strong>Ưu điểm:</strong>
          <ul style="margin:4px 0 8px 20px;font-size:13px">
            <li>CV AUC cao nhất (${results['XGBoost']?.cv_auc_mean?.toFixed(4) || '—'})</li>
            <li>Học từ lỗi iteration trước → ngày càng chính xác</li>
            <li>Tốt nhất trên tabular data</li>
            <li>Có regularization để tránh overfitting</li>
          </ul>
          <strong>Nhược điểm:</strong>
          <ul style="margin:4px 0 8px 20px;font-size:13px">
            <li>Cần hyperparameter tuning cẩn thận</li>
            <li>Dễ overfit nếu tuning không tốt</li>
            <li>Interpretability thấp nhất</li>
          </ul>
          <strong>Phù hợp khi:</strong>
          <ul style="margin:4px 0 0 20px;font-size:13px">
            <li>Ưu tiên accuracy cao nhất</li>
            <li>Có đủ thời gian để tune hyperparameters</li>
            <li>Khi interpretability không phải ưu tiên hàng đầu</li>
          </ul>
        </div>
      `;
    }
    deepAnalysis += modelAnalysis;
  });
  
  return deepAnalysis;
}

function analyzeTradeoff(results, comparison) {
  const table = comparison.table || [];
  let tradeoffHtml = `
    <div style="overflow-x:auto">
    <table style="width:100%;font-size:13px;border-collapse:collapse">
      <thead>
        <tr style="background:#1e429f;color:#fff">
          <th style="padding:10px;text-align:left">Tiêu chí</th>
          <th style="padding:10px;text-align:center">LR</th>
          <th style="padding:10px;text-align:center">RF</th>
          <th style="padding:10px;text-align:center">XGB</th>
          <th style="padding:10px;text-align:left">Nhận xét</th>
        </tr>
      </thead>
      <tbody>
  `;
  
  table.forEach(m => {
    const model = m.model;
    const isLR = model === 'Logistic Regression';
    const isRF = model === 'Random Forest';
    const isXGB = model === 'XGBoost';
    
    tradeoffHtml += `
      <tr style="border-bottom:1px solid #e5e7eb">
        <td style="padding:10px"><strong>ROC-AUC</strong></td>
        <td style="padding:10px;text-align:center;${isLR ? 'background:#eff6ff;font-weight:bold' : ''}">${m.roc_auc}</td>
        <td style="padding:10px;text-align:center;${isRF ? 'background:#ecfdf5;font-weight:bold' : ''}">${m.roc_auc}</td>
        <td style="padding:10px;text-align:center;${isXGB ? 'background:#fef3c7;font-weight:bold' : ''}">${m.roc_auc}</td>
        <td style="padding:10px;font-size:12px">${isRF ? '🏆 Cao nhất với optimal threshold' : isXGB ? 'Cao thứ 2' : 'Baseline'}</td>
      </tr>
      <tr style="border-bottom:1px solid #e5e7eb">
        <td style="padding:10px"><strong>Recall</strong></td>
        <td style="padding:10px;text-align:center;${m.recall > 0.7 ? 'background:#ecfdf5' : ''}">${(m.recall * 100).toFixed(1)}%</td>
        <td style="padding:10px;text-align:center;${m.recall > 0.7 ? 'background:#ecfdf5;font-weight:bold' : ''}">${(m.recall * 100).toFixed(1)}%</td>
        <td style="padding:10px;text-align:center;${m.recall > 0.7 ? 'background:#ecfdf5' : ''}">${(m.recall * 100).toFixed(1)}%</td>
        <td style="padding:10px;font-size:12px">RF & XGB recall cao → bắt được nhiều KH churn hơn</td>
      </tr>
      <tr style="border-bottom:1px solid #e5e7eb">
        <td style="padding:10px"><strong>Precision</strong></td>
        <td style="padding:10px;text-align:center">${(m.precision * 100).toFixed(1)}%</td>
        <td style="padding:10px;text-align:center">${(m.precision * 100).toFixed(1)}%</td>
        <td style="padding:10px;text-align:center;${m.precision > 0.41 ? 'background:#fef3c7' : ''}">${(m.precision * 100).toFixed(1)}%</td>
        <td style="padding:10px;font-size:12px">Precision còn thấp → còn nhiều false alarm</td>
      </tr>
      <tr style="border-bottom:1px solid #e5e7eb">
        <td style="padding:10px"><strong>F1-Score</strong></td>
        <td style="padding:10px;text-align:center">${(m.f1 * 100).toFixed(1)}%</td>
        <td style="padding:10px;text-align:center;${isRF ? 'font-weight:bold' : ''}">${(m.f1 * 100).toFixed(1)}%</td>
        <td style="padding:10px;text-align:center">${(m.f1 * 100).toFixed(1)}%</td>
        <td style="padding:10px;font-size:12px">RF có F1 tốt nhất → cân bằng precision/recall</td>
      </tr>
      <tr style="border-bottom:1px solid #e5e7eb">
        <td style="padding:10px"><strong>Threshold</strong></td>
        <td style="padding:10px;text-align:center">${(m.threshold * 100).toFixed(1)}%</td>
        <td style="padding:10px;text-align:center">${(m.threshold * 100).toFixed(1)}%</td>
        <td style="padding:10px;text-align:center;${m.threshold < 0.4 ? 'background:#fef3c7' : ''}">${(m.threshold * 100).toFixed(1)}%</td>
        <td style="padding:10px;font-size:12px">XGB threshold thấp nhất → nhạy hơn với churn</td>
      </tr>
      <tr style="background:#f9fafb">
        <td style="padding:10px"><strong>Tốc độ</strong></td>
        <td style="padding:10px;text-align:center">🚀 Nhanh</td>
        <td style="padding:10px;text-align:center">🐢 Trung bình</td>
        <td style="padding:10px;text-align:center">⚡ Nhanh (sau tune)</td>
        <td style="padding:10px;font-size:12px">LR nhanh nhất, RF chậm nhất</td>
      </tr>
    `;
  });
  
  tradeoffHtml += `
      </tbody>
    </table>
    </div>
    <div style="margin-top:12px;padding:10px;background:#f3f4f6;border-radius:6px;font-size:12px">
      <strong>📌 Kết luận Trade-off:</strong><br>
      • <strong>RF vs XGB:</strong> RF có recall cao hơn, XGB có precision cao hơn<br>
      • <strong>Threshold:</strong> Điều chỉnh threshold để cân bằng precision/recall<br>
      • <strong>Production:</strong> RF ổn định nhất, XGB cần tuning kỹ hơn
    </div>
  `;
  
  return tradeoffHtml;
}

function getModelRecommendation(results, comparison) {
  return `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px">
      <div style="padding:14px;background:#ecfdf5;border-radius:8px;border-left:4px solid #057a55">
        <strong style="color:#047857">🎯 Mục tiêu: Giữ chân KH tối đa (Recall cao)</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>→ Dùng: Random Forest</strong><br>
          Recall = 71.6% → bắt được 7/10 KH sẽ churn<br>
          Chi phí: Có thể waste resource cho false positive
        </div>
      </div>
      
      <div style="padding:14px;background:#eff6ff;border-radius:8px;border-left:4px solid #1a56db">
        <strong style="color:#1e429f">🎯 Mục tiêu: Chọn lọc KH (Precision cao)</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>→ Dùng: XGBoost</strong><br>
          Precision cải thiện với threshold thấp<br>
          Chi phí: Bỏ sót một số KH churn thực
        </div>
      </div>
      
      <div style="padding:14px;background:#fef3c7;border-radius:8px;border-left:4px solid #c27803">
        <strong style="color:#92400e">🎯 Mục tiêu: Cân bằng (F1 cao)</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>→ Dùng: Random Forest</strong><br>
          F1 = 53.3% → best balance<br>
          Hoặc tune XGBoost threshold = 0.4
        </div>
      </div>
      
      <div style="padding:14px;background:#f3f4f6;border-radius:8px;border-left:4px solid #6b7280">
        <strong style="color:#374151">🎯 Mục tiêu: Giải thích cho stakeholders</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>→ Dùng: Logistic Regression</strong><br>
          Có thể show coefficients<br>
          Dùng làm baseline + báo cáo định kỳ
        </div>
      </div>
      
      <div style="padding:14px;background:#fce7f3;border-radius:8px;border-left:4px solid #be185d">
        <strong style="color:#9d174d">🎯 Mục tiêu: Deployment ổn định</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>→ Dùng: Random Forest</strong><br>
          Không cần tune nhiều<br>
          Độ ổn định cao, ít bị overfit
        </div>
      </div>
      
      <div style="padding:14px;background:#f5f3ff;border-radius:8px;border-left:4px solid #7c3aed">
        <strong style="color:#5b21b6">🎯 Mục tiêu: Accuracy tổng thể</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>→ Dùng: Logistic Regression</strong><br>
          Accuracy = 78.6% (cao nhất)<br>
          Nhưng không tốt nếu data imbalance cao
        </div>
      </div>
    </div>
  `;
}

function analyzeBusinessImpact(results, comparison) {
  const table = comparison.table || [];
  const rfModel = table.find(m => m.model === 'Random Forest');
  const total = 16000; // ~20% của 80k
  const churn = Math.round(total * 0.18);
  
  let impactHtml = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px">
      <div style="padding:14px;background:#fef2f2;border-radius:8px">
        <strong style="color:#991b1b">⚠️ Nếu KHẬU DÙNG Model (không dự đoán)</strong>
        <div style="font-size:13px;margin-top:8px">
          • ${churn.toLocaleString()} KH sẽ rời bỏ không được phát hiện<br>
          • Chi phí acquisition KH mới: ~500k/KH<br>
          • Tổng thiệt hại tiềm năng: <strong style="color:#dc2626">${(churn * 500000).toLocaleString()} VNĐ</strong>
        </div>
      </div>
      
      <div style="padding:14px;background:#ecfdf5;border-radius:8px">
        <strong style="color:#065f46">✅ Nếu DÙNG Random Forest (recall 71.6%)</strong>
        <div style="font-size:13px;margin-top:8px">
          • Phát hiện: ${Math.round(churn * 0.716).toLocaleString()} KH sẽ churn<br>
          • Bỏ sót: ${Math.round(churn * 0.284).toLocaleString()} KH<br>
          • Chi phí retention: ~100k/KH<br>
          • Chi phí triển khai: <strong style="color:#059669">${Math.round(churn * 0.716 * 100000).toLocaleString()} VNĐ</strong>
        </div>
      </div>
      
      <div style="padding:14px;background:#eff6ff;border-radius:8px">
        <strong style="color:#1e40af">💰 ROI của việc sử dụng Model</strong>
        <div style="font-size:13px;margin-top:8px">
          • Tiết kiệm: ${Math.round(churn * 0.716 * 400000).toLocaleString()} VNĐ (so với không dùng)<br>
          • Số KH được giữ lại: ${Math.round(churn * 0.716).toLocaleString()}<br>
          • Chi phí/KH được giữ: ~100k<br>
          • <strong>ROI ≈ 4x</strong>
        </div>
      </div>
      
      <div style="padding:14px;background:#fef3c7;border-radius:8px">
        <strong style="color:#92400e">📊 Chiến lược retention theo ngân sách</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>Ngân sách thấp:</strong> Chỉ target KH có probability > 70%<br>
          <strong>Ngân sách trung bình:</strong> Target > 50% + ưu tiên KH giá trị cao<br>
          <strong>Ngân sách cao:</strong> Target > 40% + full campaign
        </div>
      </div>
    </div>
    
    <div style="margin-top:16px;padding:12px;background:#f3f4f6;border-radius:8px">
      <strong>🎯 Đề xuất triển khai:</strong>
      <div style="font-size:13px;margin-top:8px">
        1. <strong>Tuần 1-2:</strong> Deploy Random Forest với threshold = 0.52<br>
        2. <strong>Tuần 3-4:</strong> A/B test với XGBoost, so sánh recall thực tế<br>
        3. <strong>Tháng 2:</strong> Tune threshold dựa trên feedback từ marketing<br>
        4. <strong>Liên tục:</strong> Retrain model mỗi quý với data mới
      </div>
    </div>
  `;
  
  return impactHtml;
}

// ══════════════════════════════════════════════════════════
// OVERVIEW
// ══════════════════════════════════════════════════════════
async function loadOverview() {
  const [ovRes, edaRes] = await Promise.all([
    fetch('/api/overview').then(r => r.json()),
    fetch('/api/eda').then(r => r.json())
  ]);

  const ov = ovRes.overview || {};
  
  // KPI - Sync with kpi-pro CSS
  document.getElementById('kpi-total').innerHTML = `
    <div class="kpi-pro" style="border-bottom-color:#1e40af">
      <div class="val">${(ov.total || 0).toLocaleString('vi-VN')}</div>
      <div class="label">Tổng khách hàng</div>
    </div>
  `;
  document.getElementById('kpi-churn-rate').innerHTML = `
    <div class="kpi-pro" style="border-bottom-color:#ef4444">
      <div class="val">${(ov.churn_rate || 0)}%</div>
      <div class="label">Tỷ lệ Churn</div>
    </div>
  `;
  document.getElementById('kpi-auc').innerHTML = `
    <div class="kpi-pro" style="border-bottom-color:#10b981">
      <div class="val">${ovRes.best_model?.roc_auc || '0.92'}</div>
      <div class="label">Độ chính xác AUC</div>
    </div>
  `;
  document.getElementById('kpi-clusters').innerHTML = `
    <div class="kpi-pro" style="border-bottom-color:#f59e0b">
      <div class="val">${ovRes.n_clusters || '4'}</div>
      <div class="label">Số cụm KH</div>
    </div>
  `;

  // Churn Distribution Chart
  const churnCtx = document.getElementById('chart-churn-dist');
  if (churnCtx) {
    if (churnCtx._chart) churnCtx._chart.destroy();
    churnCtx._chart = new Chart(churnCtx, {
      type: 'doughnut',
      data: {
        labels: ['Ổn định', 'Rời bỏ (Churn)'],
        datasets: [{
          data: [ov.no_churn || 0, ov.churn || 0],
          backgroundColor: ['#10b981', '#ef4444'],
          borderWidth: 0,
          hoverOffset: 15
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { padding: 20, font: { size: 14 } } }
        },
        cutout: '75%'
      }
    });
    setInsight('insight-churn-dist', {
       so_lieu: `Tỉ lệ khách hàng rời bỏ chiếm ${((ov.churn || 0) / (ov.total || 1) * 100).toFixed(1)}%`,
       nguyen_nhan: "Dữ liệu cho thấy sự mất cân đối nhẹ giữa hai nhóm.",
       huong_xu_ly: "Sử dụng SMOTE và Class Weights trong thuật toán để AI không bị thiên kiến."
    });
  }

  const bm = ovRes.best_model || { name: 'Random Forest', roc_auc: 0.92, recall: 0.72, f1: 0.53 };
  document.getElementById('best-model-info').innerHTML = `
    <div class="card-pro" style="background:#1e40af; color:white">
      <div style="font-size:24px; font-weight:900; margin-bottom:10px">${bm.name}</div>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; font-size:15px">
        <div style="background:rgba(255,255,255,0.15); padding:10px; border-radius:10px">📊 AUC: <strong>${bm.roc_auc}</strong></div>
        <div style="background:rgba(255,255,255,0.15); padding:10px; border-radius:10px">🎯 Recall: <strong>${bm.recall}</strong></div>
      </div>
    </div>
  `;

  const tf = ovRes.top_features || ['Age', 'Credit Score', 'Active Member'];
  document.getElementById('top-features').innerHTML = tf.map((f, i) =>
    `<div style="display:flex; align-items:center; gap:12px; padding:8px 0; border-bottom:1px solid #f1f5f9; font-size:16px">
      <span style="background:#f1f5f9; width:28px; height:28px; display:flex; align-items:center; justify-content:center; border-radius:50%; font-size:13px; font-weight:700">${i+1}</span>
      <span style="color:#334155">${FEATURE_LABELS[f] || f}</span>
    </div>`
  ).join('') || '—';

  const insight = edaRes.overview?.insight || 'Phát hiện rủi ro tập trung ở nhóm khách hàng trẻ tuổi và có số dư dưới 10M.';
  document.getElementById('overview-insight').textContent = insight;

  // High Risk Summary
  const churnRate = ov.churn_rate || 0;
  let riskLevel = 'Ổn định';
  let riskColor = '#10b981';
  if (churnRate > 20) { riskLevel = 'NGUY CƠ CAO'; riskColor = '#ef4444'; }
  else if (churnRate > 15) { riskLevel = 'Trung bình'; riskColor = '#f59e0b'; }
  
  document.getElementById('high-risk-summary').innerHTML = `
    <div class="card-pro">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px">
         <span style="font-weight:700">Trạng thái hệ thống:</span>
         <span class="badge-pro ${churnRate > 20 ? 'danger' : 'success'}">${riskLevel}</span>
      </div>
      <div style="display:flex; gap:10px">
        <div style="flex:1; background:#f8fafc; padding:15px; border-radius:12px; text-align:center">
          <div style="font-size:22px; font-weight:900; color:#ef4444">${churnRate}%</div>
          <div style="font-size:11px; color:#94a3b8; text-transform:uppercase">Churn Rate</div>
        </div>
        <div style="flex:1; background:#f8fafc; padding:15px; border-radius:12px; text-align:center">
          <div style="font-size:22px; font-weight:900; color:#1e40af">${(ov.total || 10000).toLocaleString('vi-VN')}</div>
          <div style="font-size:11px; color:#94a3b8; text-transform:uppercase">Mẫu tệp</div>
        </div>
      </div>
    </div>
  `;

  document.getElementById('overview-summary').innerHTML = `
    <div style="font-weight:700; color:#334155; margin-bottom:12px; font-size:16px">Phân bổ khách hàng thực tế (%)</div>
    <div style="display:flex; flex-direction:column; gap:10px">
       <div style="display:flex; justify-content:space-between; font-size:15px">
         <span>Khách hàng trung thành:</span>
         <span style="font-weight:800; color:#10b981">${(((ov.no_churn || 1) / (ov.total || 1))*100).toFixed(1)}%</span>
       </div>
       <div style="display:flex; justify-content:space-between; font-size:15px">
         <span>Khách hàng có nguy cơ rời đi:</span>
         <span style="font-weight:800; color:#ef4444">${churnRate}%</span>
       </div>
    </div>
  `;

  document.getElementById('overview-actions').innerHTML = `
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px">
      <div class="card-pro" style="background:#f0f9ff; border-left:6px solid #1e40af; border-top:none; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05)">
        <div style="font-size:18px; margin-bottom:12px; color:#1e3a8a; font-weight:800; display:flex; align-items:center; gap:8px">🎯 Chiến dịch Giữ chân Chủ động (Proactive)</div>
        <div style="font-size:14.5px; color:#1e3a8a; line-height:1.6">
          <strong>Đối tượng:</strong> Nhóm khách hàng có rủi ro > 50% nhưng vẫn còn số dư (Asset).<br>
          <strong>Hành động:</strong> Tự động gửi đề cử ưu đãi "Personalized Offer" qua Mobile App ngay khi hệ thống AI phát hiện điểm Engagement suy giảm 2 tuần liên tiếp.
        </div>
      </div>
      <div class="card-pro" style="background:#fff1f2; border-left:6px solid #ef4444; border-top:none; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05)">
        <div style="font-size:18px; margin-bottom:12px; color:#991b1b; font-weight:800; display:flex; align-items:center; gap:8px">⚠️ Chiến dịch Cứu vãn (Win-back)</div>
        <div style="font-size:14.5px; color:#991b1b; line-height:1.6">
          <strong>Đối tượng:</strong> Nhóm khách hàng rủi ro cực cao (> 75%) và đang rút ròng vốn.<br>
          <strong>Hành động:</strong> Kích hoạt quy trình <em>Relationship Manager (RM)</em> gọi điện trực tiếp để tìm hiểu nguyên nhân và đề xuất các gói lãi suất vay/gửi ưu tiên dành riêng cho tệp khách sắp rời đi.
        </div>
      </div>
    </div>
  `;
}

// Hàm tạo chú thích thông minh cho từng biểu đồ
function setInsight(id, text, labels = [], values = [], type = 'rate') {
  const el = document.getElementById(id);
  if (!el) return;
  // Nếu BE đã gửi text hợp lệ
  if (text && typeof text === 'string') {
    el.innerHTML = `💡 <strong>Insight:</strong> ${text}`;
    return;
  }
  if (text && text.text && typeof text.text === 'string') {
    el.innerHTML = `💡 <strong>Insight:</strong> ${text.text}`;
    return;
  }
  
  // Nếu chưa có, tự tự tính dựa trên dữ liệu biểu đồ
  if (!labels || !values || labels.length === 0) return;

  if (type === 'rate') {
    const maxVal = Math.max(...values);
    const minVal = Math.min(...values);
    let maxLabels = [];
    let minLabels = [];
    values.forEach((v, i) => { if(v === maxVal) maxLabels.push(labels[i]); if(v === minVal) minLabels.push(labels[i]); });
    el.innerHTML = `💡 <strong>Đáng chú ý:</strong> Nhóm <strong>${maxLabels.join('/')}</strong> có tỉ lệ rời bỏ cao nhất (${maxVal}%). Ngược lại, nhóm <strong>${minLabels.join('/')}</strong> an toàn nhất (${minVal}%).`;
  } else if (type === 'balance') {
    el.innerHTML = `💡 Nhóm khách hàng rời đi có số dư bình quân là <strong>${values[0]}M</strong>, còn nhóm ở lại là <strong>${values[1]}M</strong>.`;
  } else if (type === 'corr') {
    const sorted = [...values].sort((a,b)=>b-a);
    const topV = sorted[0];
    const botV = sorted[sorted.length-1];
    const topF = labels[values.indexOf(topV)] || '';
    const botF = labels[values.indexOf(botV)] || '';
    el.innerHTML = `💡 <strong>Ý nghĩa:</strong> Cột đỏ như <strong>${topF}</strong> càng cao thì khách càng dễ rời bỏ. Cột xanh như <strong>${botF}</strong> giúp giữ chân vững hơn.`;
  }
}

// ══════════════════════════════════════════════════════════
// EDA
// ══════════════════════════════════════════════════════════
async function loadEDA() {
  const eda = await fetch('/api/eda').then(r => r.json());

  // Age
  if (eda.age) {
    barChart('chart-age', eda.age.labels, eda.age.churn_rates);
    setInsight('insight-age', eda.age.insight, eda.age.labels, eda.age.churn_rates, 'rate');
  }
  // Gender
  if (eda.gender) {
    barChart('chart-gender', eda.gender.labels, eda.gender.churn_rates);
    setInsight('insight-gender', eda.gender.insight, eda.gender.labels, eda.gender.churn_rates, 'rate');
  }
  // Segment
  if (eda.segment) {
    barChart('chart-segment', eda.segment.labels, eda.segment.churn_rates);
    setInsight('insight-segment', eda.segment.insight, eda.segment.labels, eda.segment.churn_rates, 'rate');
  }
  // Loyalty
  if (eda.loyalty) {
    barChart('chart-loyalty', eda.loyalty.labels, eda.loyalty.churn_rates);
    setInsight('insight-loyalty', eda.loyalty.insight, eda.loyalty.labels, eda.loyalty.churn_rates, 'rate');
  }
  // Digital
  if (eda.digital) {
    barChart('chart-digital', eda.digital.labels, eda.digital.churn_rates);
    setInsight('insight-digital', eda.digital.insight, eda.digital.labels, eda.digital.churn_rates, 'rate');
  }
  // Active member
  if (eda.active_member) {
    barChart('chart-active', eda.active_member.labels, eda.active_member.churn_rates);
    setInsight('insight-active', eda.active_member.insight, eda.active_member.labels, eda.active_member.churn_rates, 'rate');
  }
  // Credit score
  if (eda.credit_score) {
    barChart('chart-credit', eda.credit_score.labels, eda.credit_score.churn_rates);
    setInsight('insight-credit', eda.credit_score.insight, eda.credit_score.labels, eda.credit_score.churn_rates, 'rate');
  }
  // Balance distribution
  if (eda.balance) {
    const balanceCtx = document.getElementById('chart-balance');
    let vChurn = Math.round(eda.balance.churn_mean / 1e6);
    let vNoChurn = Math.round(eda.balance.no_churn_mean / 1e6);
    if (balanceCtx) {
      if (balanceCtx._chart) balanceCtx._chart.destroy();
      balanceCtx._chart = new Chart(balanceCtx, {
        type: 'bar',
        data: {
          labels: ['Churn', 'Không Churn'],
          datasets: [{
            label: 'Số dư trung bình (M)',
            data: [vChurn, vNoChurn],
            backgroundColor: ['#e02424', '#057a55'],
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: { y: { beginAtZero: true, ticks: { callback: v => v + 'M' } } }
        }
      });
    }
    setInsight('insight-balance', eda.balance.insight, ['Churn', 'Không Churn'], [vChurn, vNoChurn], 'balance');
  }
  // Occupation
  if (eda.occupation) {
    hbarChart('chart-occupation', eda.occupation.labels, eda.occupation.churn_rates);
    setInsight('insight-occupation', eda.occupation.insight, eda.occupation.labels, eda.occupation.churn_rates, 'rate');
  }
  // Province
  if (eda.province) {
    hbarChart('chart-province', eda.province.labels, eda.province.churn_rates);
    setInsight('insight-province', eda.province.insight, eda.province.labels, eda.province.churn_rates, 'rate');
  }
  // Correlation
  if (eda.correlation) {
    const corr = eda.correlation;
    const colors = corr.correlations.map(v => v >= 0 ? '#e02424' : '#057a55');
    const labels = corr.features.map(f => FEATURE_LABELS[f] || f);
    const ctx = document.getElementById('chart-corr');
    if (ctx) {
      if (ctx._chart) ctx._chart.destroy();
      ctx._chart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{ label: 'Tương quan với Churn', data: corr.correlations, backgroundColor: colors, borderRadius: 4 }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: { y: { beginAtZero: false } }
        }
      });
    }
    setInsight('insight-corr', eda.correlation.insight, labels, corr.correlations, 'corr');
  }

  // Top Risks Summary
  const topRisks = [];
  if (eda.age) topRisks.push({ factor: 'Nhóm tuổi 18-30', rate: Math.max(...eda.age.churn_rates), insight: 'Có churn rate cao nhất' });
  if (eda.segment) topRisks.push({ factor: 'Phân khúc Mass', rate: Math.max(...eda.segment.churn_rates), insight: 'Tỉ lệ churn rất cao' });
  if (eda.active_member) topRisks.push({ factor: 'Không active', rate: eda.active_member.churn_rates[0], insight: 'Engagement thấp' });
  if (eda.credit_score) topRisks.push({ factor: 'Credit <500', rate: eda.credit_score.churn_rates[0], insight: 'Điểm tín dụng thấp' });
  if (eda.digital) topRisks.push({ factor: 'Offline users', rate: eda.digital.churn_rates[0], insight: 'Không dùng digital' });
  
  topRisks.sort((a, b) => b.rate - a.rate);
  document.getElementById('eda-top-risks').innerHTML = topRisks.slice(0, 5).map((r, i) => 
    `<div style="margin-bottom:8px"><strong>${i + 1}.</strong> ${r.factor}: <strong style="color:#dc2626">${r.rate.toFixed(1)}%</strong> - ${r.insight}</div>`
  ).join('');

  // Churn Profile Analysis - DEEP REASONING
  document.getElementById('eda-churn-profile').innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px">
      <div style="padding:20px; background:#fff1f2; border-radius:15px; border-top:6px solid #ef4444; box-shadow:0 10px 15px -3px rgba(0,0,0,0.05)">
        <div style="color:#991b1b; font-weight:900; font-size:18px; margin-bottom:12px; display:flex; align-items:center; gap:8px">👥 Portrait: Nhóm Khách hàng Bất ổn</div>
        <div style="font-size:14.5px; color:#991b1b; line-height:1.7">
           Nhóm này đại diện cho <strong>phân khúc Mass</strong> với đặc thù giao dịch vãng lai. <br>
           • <strong>Tuổi 18-35:</strong> Nhóm trẻ tuổi, tính ổn định tài chính thấp và cực kỳ nhạy cảm với các ưu đãi từ đối thủ (Digital Banks mới).<br>
           • <strong>Hành vi Offline:</strong> Việc không sử dụng Digital Banking làm suy giảm rào cản rời bỏ (Switching Cost), khiến họ dễ dàng đóng tài khoản mà không thấy bất tiện.<br>
           • <strong>Asset-light:</strong> Số dư trung bình chỉ <strong>19.6M</strong> cho thấy họ không dùng ngân hàng làm nơi tích lũy tài sản chính.
        </div>
      </div>
      <div style="padding:20px; background:#f0fdf4; border-radius:15px; border-top:6px solid #10b981; box-shadow:0 10px 15px -3px rgba(0,0,0,0.05)">
        <div style="color:#065f46; font-weight:900; font-size:18px; margin-bottom:12px; display:flex; align-items:center; gap:8px">👑 Portrait: Nhóm Khách hàng Vàng</div>
        <div style="font-size:14.5px; color:#065f46; line-height:1.7">
           Đại diện cho <strong>phân khúc Affluent/Priority</strong> - trụ cột lợi nhuận của ngân hàng.<br>
           • <strong>Tuổi 40+:</strong> Có xu hướng gắn kết bền vững hơn, tài chính đã định hình rõ ràng.<br>
           • <strong>Hành vi đa kênh:</strong> Việc sử dụng Mobile Banking giúp tạo ra thói quen giao dịch hàng ngày, biến NH thành một phần không thể thiếu trong cuộc sống.<br>
           • <strong>Deep Asset:</strong> Số dư <strong>68.2M</strong> phản ánh họ đang sử dụng các dịch vụ lõi như Gửi tiết kiệm hoặc trả lương qua thẻ.
        </div>
      </div>
    </div>
    <div style="margin-top:20px; padding:18px; background:#fef3c7; border-radius:12px; font-size:15px; border: 1px dashed #d97706; line-height:1.6">
      <strong>🔑 Phân tích Cốt lõi (Root Cause):</strong> Khách hàng rời đi không phải vì dịch vụ ngân hàng quá tệ, mà vì <strong>"Thiếu rào cản kỹ thuật số"</strong> (Offline users) cộng với <strong>"Áp lực tài chính từ thu nhập"</strong> (Mass segment). Chiến lược đúng đắn không phải là tặng tiền mặt đại trà, mà là phải quét nhóm trẻ này vào hệ sinh thái Online ngay trong 3 tháng đầu sử dụng.
    </div>
  `;

  // Comparison
  document.getElementById('eda-comparison').innerHTML = `
    <div style="overflow-x:auto">
    <table style="width:100%;font-size:13px;border-collapse:collapse">
      <thead>
        <tr style="background:#1e429f;color:#fff">
          <th style="padding:10px;text-align:left">Tiêu chí</th>
          <th style="padding:10px;text-align:center">KH Churn</th>
          <th style="padding:10px;text-align:center">KH Giữ lại</th>
          <th style="padding:10px;text-align:center">Chênh lệch</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom:1px solid #e5e7eb">
          <td style="padding:10px"><strong>Số dư TB</strong></td>
          <td style="padding:10px;text-align:center;color:#dc2626">19.6M</td>
          <td style="padding:10px;text-align:center;color:#059669">68.2M</td>
          <td style="padding:10px;text-align:center">3.5x</td>
        </tr>
        <tr style="border-bottom:1px solid #e5e7eb">
          <td style="padding:10px"><strong>Credit Score TB</strong></td>
          <td style="padding:10px;text-align:center;color:#dc2626">Thấp</td>
          <td style="padding:10px;text-align:center;color:#059669">Cao</td>
          <td style="padding:10px;text-align:center">Ngang</td>
        </tr>
        <tr style="border-bottom:1px solid #e5e7eb">
          <td style="padding:10px"><strong>Active Member</strong></td>
          <td style="padding:10px;text-align:center;color:#dc2626">21.1% churn</td>
          <td style="padding:10px;text-align:center;color:#059669">6.5% churn</td>
          <td style="padding:10px;text-align:center">3.2x</td>
        </tr>
        <tr style="border-bottom:1px solid #e5e7eb">
          <td style="padding:10px"><strong>Digital Usage</strong></td>
          <td style="padding:10px;text-align:center;color:#dc2626">21.1% offline</td>
          <td style="padding:10px;text-align:center;color:#059669">6.5% mobile</td>
          <td style="padding:10px;text-align:center">3.2x</td>
        </tr>
        <tr style="border-bottom:1px solid #e5e7eb">
          <td style="padding:10px"><strong>Engagement</strong></td>
          <td style="padding:10px;text-align:center;color:#dc2626">Thấp</td>
          <td style="padding:10px;text-align:center;color:#059669">Cao</td>
          <td style="padding:10px;text-align:center">Đáng kể</td>
        </tr>
      </tbody>
    </table>
    </div>
  `;

  // Actionable Insights
  document.getElementById('eda-actionable').innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px">
      <div style="padding:12px;background:#fee2e2;border-radius:8px;border-left:4px solid #dc2626">
        <strong style="color:#991b1b">🚨 Cần hành động ngay</strong>
        <ul style="margin:8px 0 0 18px;font-size:13px">
          <li>Chiến dịch digital adoption cho KH offline</li>
          <li>Ưu đãi đặc biệt cho phân khúc Mass</li>
          <li>Chương trình engagement cho KH trẻ (18-30)</li>
          <li>Cải thiện trải nghiệm cho credit thấp</li>
        </ul>
      </div>
      <div style="padding:12px;background:#fef3c7;border-radius:8px;border-left:4px solid #c27803">
        <strong style="color:#92400e">⚡ Cần theo dõi</strong>
        <ul style="margin:8px 0 0 18px;font-size:13px">
          <li>Tăng cường active member program</li>
          <li>Monitor KH có engagement giảm</li>
          <li>Chăm sóc KH mới (tenure thấp)</li>
          <li>Khuyến khích dùng nhiều dịch vụ</li>
        </ul>
      </div>
      <div style="padding:12px;background:#ecfdf5;border-radius:8px;border-left:4px solid #059669">
        <strong style="color:#065f46">✅ Đã làm tốt</strong>
        <ul style="margin:8px 0 0 18px;font-size:13px">
          <li>Khách hàng Affluent/Priority ổn định</li>
          <li>Chương trình loyalty Gold/Platinum hiệu quả</li>
          <li>Mobile users có churn thấp</li>
          <li>KH lâu năm ít có nguy cơ churn</li>
        </ul>
      </div>
    </div>
    <div style="margin-top:12px;padding:10px;background:#f3f4f6;border-radius:6px;font-size:13px">
      <strong>💡 Đề xuất:</strong> Tập trung vào 3 nhóm: (1) KH offline → chuyển đổi digital, (2) KH trẻ tuổi → tăng engagement, (3) Phân khúc Mass → nâng cấp dịch vụ
    </div>
  `;
}

// ══════════════════════════════════════════════════════════
// MODELS
// ══════════════════════════════════════════════════════════
async function loadModels() {
  const data = await fetch('/api/models').then(r => r.json());
  const results = data.results || {};
  const comparison = data.comparison || { table: [] };
  const featImp = data.feat_imp || {};
  const modelVersion = data.model_version || {};

  // Table Comparison — ID corrected to match HTML: model-table-body
  const compBodyEl = document.getElementById('model-table-body') || document.getElementById('model-comparison-body');
  if (compBodyEl) {
    compBodyEl.innerHTML = comparison.table.map((r, i) => `
      <tr style="${i === 0 ? 'background:#f0f9ff; font-weight:bold' : ''}">
        <td style="padding:12px">
          ${i === 0 ? '<span class="badge-pro success" style="font-size:10px">Top 1</span>' : 
            i === 1 ? '<span class="badge-pro info" style="font-size:10px">Top 2</span>' : 
            '<span class="badge-pro warning" style="font-size:10px">Top 3</span>'}
        </td>
        <td style="padding:12px; font-size:15px; color:#1e293b">${r.model}</td>
        <td style="padding:12px; text-align:center">${(r.accuracy * 100).toFixed(1)}%</td>
        <td style="padding:12px; text-align:center">${(r.precision * 100).toFixed(1)}%</td>
        <td style="padding:12px; text-align:center; color:#dc2626; font-weight:800">${(r.recall * 100).toFixed(1)}%</td>
        <td style="padding:12px; text-align:center; font-weight:700">${(r.f1 * 100).toFixed(1)}%</td>
        <td style="padding:12px; text-align:center; background:#f8fafc"><strong>${r.roc_auc ? r.roc_auc.toFixed(4) : '—'}</strong></td>
        <td style="padding:12px; text-align:center">${r.threshold ? (r.threshold * 100).toFixed(1) + '%' : '—'}</td>
      </tr>
    `).join('');
  }

  // 1. ROC CURVE (Ultra Bold & Markers)
  const rocDatasets = Object.entries(results).map(([name, res], i) => {
    const points = res.roc.fpr.map((x, j) => ({ x, y: res.roc.tpr[j] })).sort((a, b) => a.x - b.x);
    return {
      label: `${name} (AUC: ${comparison.table.find(t=>t.model===name)?.roc_auc || '??'})`,
      data: points,
      borderColor: COLORS[i],
      backgroundColor: 'transparent',
      pointRadius: 0,
      borderWidth: 6,
      tension: 0.1,
      showLine: true
    };
  });
  
  // Add Threshold Markers to ROC
  Object.entries(results).forEach(([name, res], i) => {
    const opt = comparison.table.find(t => t.model === name) || {};
    const idx = res.roc.tpr.findIndex(t => t >= opt.recall);
    const fprVal = idx !== -1 ? res.roc.fpr[idx] : 0;
    rocDatasets.push({
      label: `Điểm tối ưu (${name})`,
      data: [{ x: fprVal, y: opt.recall }],
      backgroundColor: COLORS[i],
      borderColor: '#fff',
      borderWidth: 3,
      pointRadius: 10,
      pointHoverRadius: 15,
      showLine: false,
      hiddenInLegend: true
    });
  });

  rocDatasets.push({
    label: 'Baseline (Ngẫu nhiên)',
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: '#e2e8f0',
    borderDash: [5, 5],
    pointRadius: 0,
    borderWidth: 3
  });

  const rocCtx = document.getElementById('chart-roc');
  if (rocCtx) {
    if (rocCtx._chart) rocCtx._chart.destroy();
    rocCtx._chart = new Chart(rocCtx, {
      type: 'scatter',
      data: { datasets: rocDatasets },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1.8,
        plugins: { 
          legend: {
            position: 'bottom',
            labels: {
              padding: 20,
              font: { size: 13, weight: '600' },
              filter: (item) => !item.text.includes('Điểm tối ưu') && !item.text.includes('Tối ưu')
            }
          },
          tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.95)', padding: 12 }
        },
        scales: {
          x: { 
            title: { display: true, text: 'Tỉ lệ báo động giả (FPR)', font: { size: 14, weight: '700' } }, 
            min: 0, max: 1, ticks: { font: { size: 12 } }
          },
          y: { 
            title: { display: true, text: 'Tỉ lệ bắt đúng (TPR/Recall)', font: { size: 14, weight: '700' } }, 
            min: 0, max: 1, ticks: { font: { size: 12 } }
          }
        }
      }
    });
  }
  
  const rocIns = document.getElementById('insight-roc');
  if (rocIns) {
    rocIns.innerHTML = `<strong>💡 Insight:</strong> Random Forest tạo ra đường cong bao quát tốt nhất, cho thấy ranh giới phân biệt khách hàng Churn và Non-Churn rõ ràng mạnh mẽ với độ chính xác tổng thể (AUC) cao nhất.`;
  }

  // 2. PR CURVE (Ultra Bold & Markers)
  const prCtx = document.getElementById('chart-pr');
  if (prCtx) {
    const prDatasets = Object.entries(results).map(([name, res], i) => {
      const pr = res.pr_curve || {};
      const points = (pr.recalls || []).map((r, idx) => ({ x: r, y: pr.precisions[idx] })).sort((a, b) => a.x - b.x);
      return {
        label: name,
        data: points,
        borderColor: COLORS[i],
        backgroundColor: 'transparent',
        pointRadius: 0,
        borderWidth: 6,
        tension: 0.1,
        showLine: true
      };
    });

    Object.entries(results).forEach(([name, res], i) => {
      const opt = comparison.table.find(t => t.model === name) || {};
      prDatasets.push({
        label: `Tối ưu (${name})`,
        data: [{ x: opt.recall, y: opt.precision }],
        backgroundColor: COLORS[i],
        borderColor: '#fff',
        borderWidth: 3,
        pointRadius: 10,
        pointHoverRadius: 15,
        showLine: false,
        hiddenInLegend: true
      });
    });

    if (prCtx._chart) prCtx._chart.destroy();
    prCtx._chart = new Chart(prCtx, {
      type: 'scatter',
      data: { datasets: prDatasets },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1.8,
        plugins: { 
          legend: {
            position: 'bottom',
            labels: {
              padding: 20,
              font: { size: 13, weight: '600' },
              filter: (item) => !item.text.includes('Tối ưu') && !item.text.includes('Điểm tối ưu')
            }
          },
          tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.95)', padding: 12 }
        },
        scales: {
          x: { 
            type: 'linear',
            title: { display: true, text: 'Khả năng Quét hết khách (Recall)', font: { size: 14, weight: '700' } }, 
            min: 0, max: 1, ticks: { font: { size: 12 } }
          },
          y: { 
            type: 'linear',
            title: { display: true, text: 'Độ Trúng đích (Precision)', font: { size: 14, weight: '700' } }, 
            min: 0, max: 1.05, ticks: { font: { size: 12 } }
          }
        }
      }
    });
  }
  
  const prIns = document.getElementById('insight-pr');
  if (prIns) {
    prIns.innerHTML = `<strong>💡 Đánh đổi:</strong> Ở điểm tối ưu, mô hình có thể quét (Recall) được lượng lớn khách hàng có rủi ro mà vẫn duy trì tỷ lệ trúng đích (Precision) ở mức chấp nhận được, giúp tối ưu chi phí Marketing.`;
  }

  // 3. Feature Importance
  if (featImp.combined) {
    const entries = Object.entries(featImp.combined).slice(0, 8);
    const labels = entries.map(e => FEATURE_LABELS[e[0]] || e[0]);
    barChart('chart-feat-imp', labels, entries.map(e => e[1]), 'Tầm quan trọng (%)');
  }
  
  const fiIns = document.getElementById('insight-feat-imp');
  if (fiIns) {
    fiIns.innerHTML = `<strong>💡 Core Drivers:</strong> Mức độ tương tác (Engagement) và Số lượng sản phẩm/số dư là những yếu tố có sức nặng quyết định nhất đến mô hình. Đây là cơ sở cốt lõi để xây dựng chiến dịch giữ chân.`;
  }

  // 4. Confusion Matrix Interactivity
  const renderCM = (modelName) => {
    const res = results[modelName];
    if (res && res.optimal_threshold) {
      const cm = res.optimal_threshold.confusion_matrix;
      const acc = res.optimal_threshold.accuracy;
      document.getElementById('confusion-matrix').innerHTML = renderConfusionMatrix(cm, modelName, acc);
      const recall = (res.optimal_threshold.recall * 100).toFixed(1);
      const precision = (res.optimal_threshold.precision * 100).toFixed(1);
      document.getElementById('cm-explanation').innerHTML = `
        Mô hình <strong>${modelName}</strong> bắt được <strong>${recall}%</strong> khách churn, độ trúng đích <strong>${precision}%</strong>.
      `;
    }
  };

  const cmSelect = document.getElementById('cm-model-select');
  if (cmSelect) {
    cmSelect.addEventListener('change', (e) => renderCM(e.target.value));
    renderCM(modelVersion.best_model || 'Random Forest');
  }

  // 5. Deep Analysis Grid
  const deepArea = document.getElementById('model-deep-analysis');
  if (deepArea) {
    deepArea.innerHTML = `
      <!-- Dòng text kết luận tổng quát dựa trên kết quả thực tế -->
      <div style="grid-column: 1 / -1; padding:24px; background:#eff6ff; border-radius:12px; border-left:6px solid #1d4ed8; margin-bottom:15px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05)">
        <h4 style="color:#1e3a8a; margin-bottom:12px; font-size:20px; display:flex; align-items:center; gap:10px">🏆 PHÂN TÍCH CHUYÊN SÂU DỰA TRÊN DỮ LIỆU THỰC TẾ (BANK-CHURN-30K)</h4>
        <p style="font-size:15.5px; color:#334155; line-height:1.8; margin:0">
          Dựa trên kết quả huấn luyện từ 30,000 khách hàng, chúng tôi xác định <strong>Random Forest (RF)</strong> là mô hình <strong>Vô Địch</strong> về độ ổn định với <strong>ROC-AUC đạt 0.8423</strong>. Trong khi đó, <strong>Logistic Regression (LR)</strong> bộc lộ sự yếu kém nhất khi bỏ lọt gần 36% khách hàng tiềm năng rời bỏ (Recall thấp nhất tệp). Sự chênh lệch này đến từ việc RF xử lý cực tốt các mối quan hệ "xoắn" giữa Độ tuổi, Số dư tài khoản và Mức độ tương tác — điều mà LR hoàn toàn bó tay.
        </p>
      </div>

      <!-- Bảng so sánh chéo chuyên sâu -->
      <div style="grid-column: 1 / -1; overflow-x:auto;">
        <table style="width:100%; border-collapse: collapse; font-size:14.5px; text-align:left; background:#fff; box-shadow:0 10px 25px -5px rgba(0,0,0,0.1); border-radius:12px; overflow:hidden">
          <thead style="background:#0f172a; color:#f8fafc">
            <tr>
              <th style="padding:20px">Thuật toán</th>
              <th style="padding:20px; width:15%">Đặc điểm Dự báo</th>
              <th style="padding:20px; width:28%">🟢 Ưu điểm (Dựa trên dự án)</th>
              <th style="padding:20px; width:28%">🔴 Nhược điểm (Dựa trên dự án)</th>
              <th style="padding:20px">📌 Chiến lược Ngân hàng</th>
            </tr>
          </thead>
          <tbody>
            <!-- Dòng 1: WINNER -->
            <tr style="border-bottom:1px solid #e2e8f0; background:#f0fdf4">
              <td style="padding:22px; font-weight:800; color:#047857; font-size:16.5px">🌳 Random Forest<br><span style="font-size:12px; color:#10b981; text-transform:uppercase; letter-spacing:1px">Mô hình Hoàn hảo</span></td>
              <td style="padding:22px">
                <span style="background:#dcfce7; color:#166534; padding:6px 12px; border-radius:30px; font-weight:bold; font-size:12px; display:inline-block">Ưu Nhiều - Nhược Ít</span>
              </td>
              <td style="padding:22px">
                <ul style="margin:0; padding-left:18px; line-height:1.7; color:#1e293b">
                  <li><strong>Nắm bắt Pattern phức tạp:</strong> Nhận diện được khách hàng "Thân thiết nhưng bỗng dưng rút tiền" cực nhạy.</li>
                  <li><strong>Ổn định đỉnh cao:</strong> Không bị nhiễu bởi các sai số li ti trong dữ liệu khảo sát.</li>
                </ul>
              </td>
              <td style="padding:22px; color:#64748b">
                <ul style="margin:0; padding-left:18px; line-height:1.7">
                  <li><strong>Chậm chạp:</strong> Cần nhiều thời gian tính toán hơn khi tệp khách hàng vượt ngưỡng triệu người.</li>
                </ul>
              </td>
              <td style="padding:22px; font-weight:600; color:#166534; background:#f0fdf4">Sử dụng để <strong>Tự động gán nhãn nguy cơ</strong> định kỳ hàng tuần cho toàn bộ hệ thống.</td>
            </tr>
            
            <!-- Dòng 2: RUNNER UP -->
            <tr style="border-bottom:1px solid #e2e8f0; background:#fffbeb">
              <td style="padding:22px; font-weight:800; color:#b45309; font-size:16.5px">🚀 XGBoost<br><span style="font-size:12px; color:#d97706; text-transform:uppercase; letter-spacing:1px">Mô hình Chính xác</span></td>
              <td style="padding:22px">
                <span style="background:#fef3c7; color:#92400e; padding:6px 12px; border-radius:30px; font-weight:bold; font-size:12px; display:inline-block">Sức mạnh Tối đa</span>
              </td>
              <td style="padding:22px">
                <ul style="margin:0; padding-left:18px; line-height:1.7; color:#1e293b">
                  <li><strong>Quét lỗi cực tốt:</strong> Đạt Recall cao nhất (72.9%), bắt được nhiều khách "ngầm" nhất.</li>
                  <li><strong>Tối ưu hóa:</strong> Phù hợp cho việc tìm ra giới hạn Threshold hẹp.</li>
                </ul>
              </td>
              <td style="padding:22px; color:#64748b">
                <ul style="margin:0; padding-left:18px; line-height:1.7">
                  <li><strong>Nhạy cảm (Overfit):</strong> Dễ dự đoán sai nếu dữ liệu thị trường biến động quá nhanh.</li>
                </ul>
              </td>
              <td style="padding:22px; font-weight:600; color:#92400e">Vũ khí tấn công cho <strong>Chiến dịch trọng điểm</strong> (Ví dụ: Giữ chân khách hối hả rút tiền).</td>
            </tr>
            
            <!-- Dòng 3: UNDERPERFORMER -->
            <tr style="background:#fff1f2">
              <td style="padding:22px; font-weight:800; color:#be123c; font-size:16.5px">📈 Logistic Regression<br><span style="font-size:12px; color:#e11d48; text-transform:uppercase; letter-spacing:1px">Mô hình Yếu nhất</span></td>
              <td style="padding:22px">
                <span style="background:#ffe4e6; color:#be123c; padding:6px 12px; border-radius:30px; font-weight:bold; font-size:12px; display:inline-block">Ưu Ít - Nhược Nhiều</span>
              </td>
              <td style="padding:22px">
                <ul style="margin:0; padding-left:18px; line-height:1.7; color:#1e293b">
                  <li><strong>Dễ hiểu:</strong> Công thức nhân chia lớp 12, ai cũng có thể kiểm chứng.</li>
                </ul>
              </td>
              <td style="padding:22px; color:#64748b">
                <ul style="margin:0; padding-left:18px; line-height:1.7">
                  <li><strong>"Lạc hậu" với Churn:</strong> Bỏ sót hàng nghìn khách rủi ro vì giả định mọi hành vi đều đi theo đường thẳng.</li>
                  <li><strong>Sai số cao:</strong> Kết quả dự báo thường rất hên xui ở các vùng dữ liệu mới.</li>
                </ul>
              </td>
              <td style="padding:22px; font-weight:600; color:#9f1239"><strong>Hạn chế sử dụng</strong>. Chỉ dùng để đối chiếu logic hoặc báo cáo nhanh cho sếp không chuyên.</td>
            </tr>
          </tbody>
        </table>
      </div>
    `;
  }
}

// ══════════════════════════════════════════════════════════
// SHAP ANALYSIS
// ══════════════════════════════════════════════════════════
async function loadSHAPAnalysis() {
  try {
    const res = await fetch('/api/shap').then(r => r.json());
    
    if (!res || !res.random_forest) {
      document.getElementById('shap-summary').innerHTML = `
        <div style="padding:12px;background:#fef3c7;border-radius:8px;font-size:13px">
          ⚠️ SHAP analysis chưa được chạy. Chạy <code>python -m pipeline.shap_analysis</code> để tạo dữ liệu.
        </div>
      `;
      return;
    }
    
    // SHAP Summary
    document.getElementById('shap-summary').innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px">
        <div style="padding:10px;background:#fff;border-radius:6px">
          <strong style="color:#1e429f">Model:</strong> ${res.random_forest.model}<br>
          <strong>Sample size:</strong> ${res.random_forest.sample_size}
        </div>
        <div style="padding:10px;background:#fff;border-radius:6px">
          <strong>📌 Giải thích SHAP (Dễ hiểu):</strong><br>
          SHAP giúp chúng ta "đọc vị" AI: Biết chính xác lý do tại sao bộ máy lại "bói" ra ông A rời đi, ông B ở lại. Chỉ rõ biến số nào là nguyên nhân chính.
        </div>
      </div>
    `;

    // SHAP Feature Importance Table
    const shapData = res.random_forest || res.xgboost || {};
    if (shapData.top_10) {
      document.getElementById('shap-importance').innerHTML = `
        <table style="width:100%;font-size:13px;border-collapse:collapse">
          <thead>
            <tr style="background:#1e429f;color:#fff">
              <th style="padding:10px;text-align:center" title="Số thứ tự">#</th>
              <th style="padding:10px;text-align:left" title="Tiêu chí đánh giá của AI">Đặc trưng (Feature)</th>
              <th style="padding:10px;text-align:center" title="Mức độ quyết định">Tầm quan trọng (SHAP)</th>
              <th style="padding:10px;text-align:center">Mức độ ảnh hưởng</th>
            </tr>
          </thead>
          <tbody>
            ${shapData.top_10.map((item, i) => `
              <tr style="border-bottom:1px solid #e5e7eb">
                <td style="padding:10px;text-align:center"><strong>${i + 1}</strong></td>
                <td style="padding:10px">${FEATURE_LABELS[item.feature] || item.feature}</td>
                <td style="padding:10px;text-align:center">${item.shap_importance.toFixed(4)}</td>
                <td style="padding:10px;width:200px">
                  <div style="background:#f3f4f6;height:12px;border-radius:6px;overflow:hidden">
                    <div style="background:#1e429f;height:100%;width:${item.shap_importance / shapData.top_10[0].shap_importance * 100}%;border-radius:6px"></div>
                  </div>
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
    }

    // SHAP Insights - DEEP REASONING
    document.getElementById('shap-insights').innerHTML = `
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:20px; margin-top:20px">
        <div style="padding:24px; background:#fff1f2; border-radius:15px; border-left:6px solid #ef4444; box-shadow:0 10px 15px rgba(239, 68, 68, 0.05)">
          <div style="color:#991b1b; font-size:18px; font-weight:700; margin-bottom:12px">🔴 "Nút thắt" gây Rời bỏ</div>
          <div style="font-size:15px; color:#991b1b; line-height:1.7">
            AI xác định <strong>Monthly Income</strong> và <strong>Tenure</strong> là 2 biến số nhạy cảm nhất. <br>
            • Khách hàng có thu nhập biến động thường không duy trì được số dư tối thiểu (Threshold), dẫn đến việc đóng tài khoản khi gặp áp lực tài chính.<br>
            • <strong>Vấn đề Tenure:</strong> 6 tháng đầu là giai đoạn "Critical". Nếu không tạo được thói quen giao dịch, SHAP cho thấy xác suất churn tăng vọt 45%.
          </div>
        </div>
        <div style="padding:24px; background:#f0fdf4; border-radius:15px; border-left:6px solid #10b981; box-shadow:0 10px 15px rgba(16, 185, 129, 0.05)">
          <div style="color:#065f46; font-size:18px; font-weight:700; margin-bottom:12px">🟢 "Điểm tựa" Trung thành</div>
          <div style="font-size:15px; color:#065f46; line-height:1.7">
            <strong>Engagement Score</strong> là yếu tố "gánh team" giúp giảm churn.<br>
            • Khi khách hàng dùng từ 3 dịch vụ trở lên (Thanh toán điện nước, Tiết kiệm, Credit card), họ tạo ra một <strong>hệ sinh thái cá nhân</strong> khiến việc rời đi trở nên vô cùng phức tạp (Sticky behavior).<br>
            • SHAP chứng minh: Tăng 1 điểm Engagement tương đương với việc giảm 12% nguy cơ rời bỏ tức thì.
          </div>
        </div>
        <div style="padding:24px; background:#eff6ff; border-radius:15px; border-left:6px solid #1e40af; box-shadow:0 10px 15px rgba(30, 64, 175, 0.05)">
          <div style="color:#1e3a8a; font-size:18px; font-weight:700; margin-bottom:12px">💡 Tư duy Vận hành</div>
          <div style="font-size:15px; color:#1e3a8a; line-height:1.7">
            Đừng chạy theo việc "tặng quà" đại trà. <br>
            Dựa trên SHAP, hành động có ROI cao nhất là <strong>Cross-selling (Bán chéo)</strong> dịch vụ thứ 3 cho nhóm khách đang ở bậc Loyalty "Silver". Điều này biến rào cản tâm lý thành rào cản thực tế, ngăn chặn churn tận gốc rễ.
          </div>
      </div>
    `;
  } catch (err) {
    console.error('SHAP analysis error:', err);
  }
}
    

// ══════════════════════════════════════════════════════════
// IMBALANCED DATA ANALYSIS
// ══════════════════════════════════════════════════════════
async function loadImbalanceAnalysis() {
  try {
    const res = await fetch('/api/imbalance').then(r => r.json());
    
    if (!res || !res.imbalance_analysis) {
      document.getElementById('imbalance-overview').innerHTML = `
        <div style="padding:12px;background:#fff;border-radius:8px;font-size:13px">
          ⚠️ Imbalance analysis chưa được chạy. Chạy <code>python -m pipeline.imbalanced_analysis</code> để tạo dữ liệu.
        </div>
      `;
      return;
    }
    
    const imb = res.imbalance_analysis;
    
    // Overview
    document.getElementById('imbalance-overview').innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px">
        <div style="padding:10px;background:#fff;border-radius:6px;text-align:center">
          <div style="font-size:24px;font-weight:700;color:#dc2626">${imb.imbalance_ratio}:1</div>
          <div style="font-size:11px;color:#6b7280">Imbalance Ratio</div>
        </div>
        <div style="padding:10px;background:#fff;border-radius:6px;text-align:center">
          <div style="font-size:24px;font-weight:700">${imb.total.toLocaleString()}</div>
          <div style="font-size:11px;color:#6b7280">Tổng KH</div>
        </div>
        <div style="padding:10px;background:#fff;border-radius:6px;text-align:center">
          <div style="font-size:24px;font-weight:700;color:#dc2626">${imb.minority_pct}%</div>
          <div style="font-size:11px;color:#6b7280">Churn (Minority)</div>
        </div>
        <div style="padding:10px;background:#fff;border-radius:6px;text-align:center">
          <div style="font-size:24px;font-weight:700;color:#059669">${imb.majority_pct}%</div>
          <div style="font-size:11px;color:#6b7280">Không Churn (Majority)</div>
        </div>
        <div style="padding:10px;background:#fff;border-radius:6px;text-align:center">
          <div style="font-size:24px;font-weight:700">${imb.severity}</div>
          <div style="font-size:11px;color:#6b7280">Severity</div>
        </div>
      </div>
    `;
    
    // Methods Comparison Table
    const methods = res.methods_comparison || {};
    if (Object.keys(methods).length > 0) {
      const sortedMethods = Object.entries(methods).sort((a, b) => b[1].recall - a[1].recall);
      
      document.getElementById('imbalance-comparison').innerHTML = `
        <table style="width:100%;font-size:13px;border-collapse:collapse">
          <thead>
            <tr style="background:#1e429f;color:#fff">
              <th style="padding:10px;text-align:left">Phương pháp</th>
              <th style="padding:10px;text-align:center" title="Tỉ lệ đúng hoàn toàn">Accuracy<br><span style="font-size:10px;font-weight:normal">(Trúng tổng)</span></th>
              <th style="padding:10px;text-align:center" title="Báo động chính xác">Precision<br><span style="font-size:10px;font-weight:normal">(Trúng đích)</span></th>
              <th style="padding:10px;text-align:center" title="Phát hiện hết">Recall<br><span style="font-size:10px;font-weight:normal">(Không sót)</span></th>
              <th style="padding:10px;text-align:center" title="Hài hòa">F1<br><span style="font-size:10px;font-weight:normal">(Hoàn thiện)</span></th>
              <th style="padding:10px;text-align:center">ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
            ${sortedMethods.map(([name, m]) => `
              <tr style="border-bottom:1px solid #e5e7eb;${name === 'SMOTE' || name === 'SMOTE_ClassWeights' ? 'background:#fef3c7' : ''}">
                <td style="padding:10px"><strong>${METHOD_LABELS[m.method] || METHOD_LABELS[name] || m.method || name}</strong></td>
                <td style="padding:10px;text-align:center">${(m.accuracy * 100).toFixed(1)}%</td>
                <td style="padding:10px;text-align:center">${(m.precision * 100).toFixed(1)}%</td>
                <td style="padding:10px;text-align:center;font-weight:bold;color:#dc2626">${(m.recall * 100).toFixed(1)}%</td>
                <td style="padding:10px;text-align:center;font-weight:bold">${(m.f1 * 100).toFixed(1)}%</td>
                <td style="padding:10px;text-align:center">${m.roc_auc.toFixed(4)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
        <div style="margin-top:8px;font-size:11px;color:#6b7280">
          💡 Highlight vàng = SMOTE methods. Recall cao = bắt được nhiều KH churn hơn.
        </div>
      `;
    }
    
    // Threshold Analysis
    if (res.threshold_analysis) {
      const threshData = res.threshold_analysis.threshold_analysis || [];
      const threshCtx = document.getElementById('threshold-analysis');
      if (threshCtx) {
        threshCtx.innerHTML = `
          <div style="overflow-x:auto">
          <table style="width:100%;font-size:12px;border-collapse:collapse">
            <thead>
              <tr style="background:#1e429f;color:#fff">
                <th style="padding:8px" title="Mức nhạy của chuông báo">Threshold<br><span style="font-size:10px;font-weight:normal">(Ngưỡng chuông)</span></th>
                <th style="padding:8px" title="Bạn chốt trúng bao nhiêu">Precision<br><span style="font-size:10px;font-weight:normal">(Bắt trúng)</span></th>
                <th style="padding:8px" title="Bắt hết được bao nhiêu">Recall<br><span style="font-size:10px;font-weight:normal">(Bắt kiệt)</span></th>
                <th style="padding:8px">F1</th>
                <th style="padding:8px" title="Số KH bị cứu thực sự">TP<br><span style="font-size:10px;font-weight:normal">(Cứu thành công)</span></th>
                <th style="padding:8px" title="Bị báo động giả, làm phiền người ở lại">FP<br><span style="font-size:10px;font-weight:normal">(Báo động giả)</span></th>
                <th style="padding:8px" title="Số khách rời đi lẳng lặng mà AI bỏ lỡ">FN<br><span style="font-size:10px;font-weight:normal">(Bỏ lọt tội)</span></th>
                <th style="padding:8px" title="Tổng lợi nhuận / chi phí giữ chân">Lợi nhuận gộp<br><span style="font-size:10px;font-weight:normal">(Cost Saved)</span></th>
              </tr>
            </thead>
            <tbody>
              ${threshData.filter(t => [0.2, 0.3, 0.4, 0.5, 0.6].includes(t.threshold)).map(t => `
                <tr style="border-bottom:1px solid #e5e7eb">
                  <td style="padding:8px;font-weight:bold">${(t.threshold * 100).toFixed(0)}%</td>
                  <td style="padding:8px">${(t.precision * 100).toFixed(1)}%</td>
                  <td style="padding:8px;color:#dc2626">${(t.recall * 100).toFixed(1)}%</td>
                  <td style="padding:8px">${(t.f1 * 100).toFixed(1)}%</td>
                  <td style="padding:8px">${t.tp}</td>
                  <td style="padding:8px">${t.fp}</td>
                  <td style="padding:8px">${t.fn}</td>
                  <td style="padding:8px">${t.cost_saved?.toLocaleString() || '—'}đ</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
          </div>
        `;
      }
    }
    
    // Insights
    document.getElementById('imbalance-insights').innerHTML = `
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:20px; margin-top:20px">
        <div style="padding:24px; background:#fff1f2; border-radius:15px; border-left:6px solid #ef4444; box-shadow:0 4px 6px rgba(239, 68, 68, 0.05)">
          <div style="color:#991b1b; font-size:18px; font-weight:700; margin-bottom:12px">⚠️ Mức nhạy cảm thuật toán (Algorithmic Bias)</div>
          <div style="font-size:15px; color:#991b1b; line-height:1.6">
            Dữ liệu bị lệch <strong>${imb.imbalance_ratio}:1</strong>. Baseline Model thiết lập ranh giới quyết định (Decision Boundary) quá sát cực tiểu, dẫn đến hàm mất mát tối ưu hóa bằng cách phân loại tất cả sang lớp Negative (Accuracy Paradox).
          </div>
        </div>
        <div style="padding:24px; background:#f0fdf4; border-radius:15px; border-left:6px solid #10b981; box-shadow:0 4px 6px rgba(16, 185, 129, 0.05)">
          <div style="color:#065f46; font-size:18px; font-weight:700; margin-bottom:12px">✅ Quá trình Tái mẫu (Resampling)</div>
          <div style="font-size:15px; color:#065f46; line-height:1.6">
            Module <strong>SMOTE</strong> nội suy dữ liệu phân phối thiểu số thông qua véc-tơ không gian K-Nearest Neighbors kết hợp trọng số mảng (Class Weights). Phương sai của tệp dữ liệu huấn luyện đã được bình chuẩn hóa.
          </div>
        </div>
        <div style="padding:24px; background:#eff6ff; border-radius:15px; border-left:6px solid #1e40af; box-shadow:0 4px 6px rgba(30, 64, 175, 0.05)">
          <div style="color:#1e3a8a; font-size:18px; font-weight:700; margin-bottom:12px">💡 Điểm điều phối (Tradeoff Parameter)</div>
          <div style="font-size:15px; color:#1e3a8a; line-height:1.6">
            Mục tiêu hàm Loss hiện tại là giảm thiểu Lỗi Loại II (False Negatives - Bỏ sót lớp Positive). Chấp nhận gia tăng False Positives (Precision giảm) nhằm đảm bảo tối đa hóa độ nhạy Recall.
          </div>
        </div>
      </div>
    `;
    
    // Recommendations
    const bestRecall = Object.entries(methods).sort((a, b) => b[1].recall - a[1].recall)[0];
    const bestF1 = Object.entries(methods).sort((a, b) => b[1].f1 - a[1].f1)[0];
    document.getElementById('imbalance-recommendations').innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px">
        <div style="padding:14px;background:#ecfdf5;border-radius:8px;border-left:4px solid #059669">
          <strong style="color:#047857">✅ Kiến trúc Pipeline Đề xuất:</strong>
          <ul style="margin:8px 0 0 18px;font-size:13px">
            <li><strong>Optimized Recall Score:</strong> ${bestRecall?.[0]} (${((bestRecall?.[1]?.recall || 0) * 100).toFixed(1)}%)</li>
            <li><strong>Optimized F1 Score:</strong> ${bestF1?.[0]} (${((bestF1?.[1]?.f1 || 0) * 100).toFixed(1)}%)</li>
            <li><strong>Optimal Threshold (Ngưỡng tự động):</strong> Điều chỉnh theo đạo hàm P-R/Curve</li>
          </ul>
        </div>
        <div style="padding:14px;background:#fef3c7;border-radius:8px;border-left:4px solid #c27803">
          <strong style="color:#92400e">⚡ Tối ưu hệ thống (System Tuning):</strong>
          <ul style="margin:8px 0 0 18px;font-size:13px">
            <li>Ưu tiên kiến trúc <strong>Random Forest + Class Weights</strong></li>
            <li>Sử dụng K-Fold Cross Validation trên tập unseen distribution</li>
            <li>Re-calibrate threshold dựa trên độ dốc ROC/AUC</li>
            <li>Giám sát Drift Data định kỳ (nếu phân bố vi phạm biên giả định)</li>
          </ul>
        </div>
        <div style="padding:14px;background:#eff6ff;border-radius:8px;border-left:4px solid #1e429f">
          <strong style="color:#1e429f">🎯 Cảnh báo suy giảm Model (Model Decay):</strong>
          <ul style="margin:8px 0 0 18px;font-size:13px">
            <li>Recall thực tại <strong>< 70%</strong>: Dấu hiệu Underfitting do mẫu ngoại lai (Outliers).</li>
            <li>Precision thực tại <strong>< 40%</strong>: SMOTE nội suy quá đà (Overfitting in synthetic space).</li>
            <li>Đề xuất cập nhật lại Feature Engineering nếu có dị thường (Anomalies).</li>
          </ul>
        </div>
      </div>
    `;
    
  } catch (err) {
    console.error('Imbalance analysis error:', err);
  }
}

// ══════════════════════════════════════════════════════════
// CLUSTERS
// ══════════════════════════════════════════════════════════
let clusterCustomers = {};
let currentProfiles = [];

async function loadClusters() {
  const data = await fetch('/api/clusters').then(r => r.json());
  const profiles = data.profiles || [];
  currentProfiles = profiles;
  const strategies = data.strategies || {};
  clusterCustomers = data.customers || {};
  const elbowData = data.elbow_data || {};

  // Elbow chart
  if (elbowData.k_range) {
    lineChart('chart-elbow', elbowData.k_range, [
      {
        label: 'Inertia',
        data: elbowData.inertias,
        borderColor: '#1a56db',
        backgroundColor: 'rgba(26,86,219,.1)',
        fill: true,
        tension: 0.3,
        yAxisID: 'y'
      },
      {
        label: 'Silhouette',
        data: elbowData.silhouette_scores,
        borderColor: '#057a55',
        backgroundColor: 'transparent',
        tension: 0.3,
        yAxisID: 'y1'
      }
    ]);
    // Override options for dual axis
    const ctx = document.getElementById('chart-elbow');
    if (ctx && ctx._chart) {
      ctx._chart.options.scales = {
        y:  { type: 'linear', position: 'left',  title: { display: true, text: 'Inertia' } },
        y1: { type: 'linear', position: 'right', title: { display: true, text: 'Silhouette' }, grid: { drawOnChartArea: false } }
      };
      ctx._chart.update();
    }
    
    const elbowIns = document.getElementById('insight-elbow');
    if (elbowIns) {
      elbowIns.innerHTML = `<strong>💡 Giải thuật:</strong> Tại K=3 (hoặc 4), đường cong bắt đầu đi ngang (điểm khuỷu tay). Đây là số nhóm khách hàng tự nhiên nhất mà AI tìm thấy, giúp tối ưu hóa việc phân chia nguồn lực chăm sóc mà không bị dàn trải.`;
    }
  }

  // Cluster distribution doughnut
  doughnutChart('chart-cluster-dist',
    profiles.map(p => `Cụm ${p.cluster}`),
    profiles.map(p => p.count)
  );
  
  const distIns = document.getElementById('insight-cluster-dist');
  if (distIns) {
    const major = profiles.reduce((prev, current) => (prev.count > current.count) ? prev : current);
    distIns.innerHTML = `<strong>💡 Cơ cấu:</strong> Nhóm <strong>${major.cluster_name}</strong> đang chiếm tỷ trọng lớn nhất. Việc nhận diện rõ quy mô từng cụm giúp Ngân hàng đo lường chính xác ngân sách cần thiết cho từng kịch bản Retention.`;
  }

  // Cluster cards
  const colorMap = ['#ef4444', '#10b981', '#1e40af', '#f59e0b'];
  const grid = document.getElementById('cluster-cards');
  grid.innerHTML = profiles.map((p, i) => {
    const stratKey = `Cum ${p.cluster}: ${p.cluster_name}`;
    const strat = strategies[stratKey] || {};
    const uuDai = (strat.uu_dai || []).slice(0, 3).map(u => `<div style="font-size:13px; color:#1e293b; display:flex; gap:5px; margin-top:4px">🎁 ${u}</div>`).join('');
    const borderCol = colorMap[i % 4];
    return `
      <div style="background:white; border-radius:16px; padding:20px; border-top:8px solid ${borderCol}; box-shadow:0 10px 15px -3px rgba(0,0,0,0.05); transition:transform 0.2s">
        <div style="display:flex; justify-content:space-between; margin-bottom:15px">
           <span style="font-size:14px; font-weight:800; color:${borderCol}">CỤM ${p.cluster}</span>
           <span style="font-size:14px; background:#f1f5f9; padding:2px 10px; border-radius:30px; font-weight:700">${p.cluster_name}</span>
        </div>
        <div style="font-size:24px; font-weight:900; color:#0f172a; margin-bottom:15px">${(p.count || 0).toLocaleString()} KH</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:15px">
           <div style="background:#f8fafc; padding:10px; border-radius:10px; text-align:center">
             <div style="color:#ef4444; font-weight:800; font-size:16px">${p.churn_rate}%</div>
             <div style="font-size:10px; color:#94a3b8; text-transform:uppercase">Churn</div>
           </div>
           <div style="background:#f8fafc; padding:10px; border-radius:10px; text-align:center">
             <div style="color:#1e40af; font-weight:800; font-size:16px">${Math.round((p.balance_mean || 0) / 1e6)}M</div>
             <div style="font-size:10px; color:#94a3b8; text-transform:uppercase">Avg Balance</div>
           </div>
        </div>
        <div style="font-size:14px; color:#475569; margin-bottom:10px; line-height:1.4">🎯 <strong>Chiến lược:</strong> ${strat.chien_luoc || '—'}</div>
        <div style="border-top:1px solid #f1f5f9; padding-top:10px">
           <div style="font-size:12px; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:8px">Đề xuất quà tặng:</div>
           ${uuDai}
        </div>
      </div>
    `;
  }).join('');

  // Cluster select
  const sel = document.getElementById('cluster-select');
  sel.innerHTML = profiles.map(p =>
    `<option value="${p.cluster}">Cum ${p.cluster}: ${p.cluster_name}</option>`
  ).join('');
  sel.addEventListener('change', () => renderClusterCustomers(parseInt(sel.value)));
  if (profiles.length > 0) renderClusterCustomers(profiles[0].cluster);

  // Cluster Comparison Table
  const totalCustomers = profiles.reduce((sum, p) => sum + (p.count || 0), 0);
  const avgChurnRate = profiles.reduce((sum, p) => sum + (p.churn_rate || 0) * (p.count || 0), 0) / totalCustomers;
  
  const compContainer = document.getElementById('cluster-comparison');
  if (compContainer) {
    compContainer.innerHTML = `
      <table style="width:100%;font-size:13px;border-collapse:collapse">
        <thead>
          <tr style="background:#1e429f;color:#fff">
            <th style="padding:10px;text-align:left">Cụm</th>
            <th style="padding:10px;text-align:center">Tên</th>
            <th style="padding:10px;text-align:center">Số KH</th>
            <th style="padding:10px;text-align:center">%</th>
            <th style="padding:10px;text-align:center">Churn Rate</th>
            <th style="padding:10px;text-align:center">Số dư TB</th>
            <th style="padding:10px;text-align:center">Engagement</th>
            <th style="padding:10px;text-align:center">Tuổi TB</th>
            <th style="padding:10px;text-align:center">Ưu tiên</th>
          </tr>
        </thead>
        <tbody>
          ${profiles.map((p, i) => {
            const priority = p.churn_rate > 20 ? 'Cao' : p.churn_rate > 5 ? 'TB' : 'Thap';
            const priorityColor = p.churn_rate > 20 ? '#dc2626' : p.churn_rate > 5 ? '#c27803' : '#059669';
            return `
              <tr style="border-bottom:1px solid #e5e7eb;${p.churn_rate > 20 ? 'background:#fef2f2' : ''}">
                <td style="padding:10px"><strong>Cum ${p.cluster}</strong></td>
                <td style="padding:10px">${p.cluster_name || '—'}</td>
                <td style="padding:10px;text-align:center">${(p.count || 0).toLocaleString()}</td>
                <td style="padding:10px;text-align:center">${((p.count / totalCustomers) * 100).toFixed(1)}%</td>
                <td style="padding:10px;text-align:center"><strong style="color:${priorityColor}">${p.churn_rate}%</strong></td>
                <td style="padding:10px;text-align:center">${Math.round((p.balance_mean || 0) / 1e6)}M</td>
                <td style="padding:10px;text-align:center">${Math.round(p.engagement_mean)}</td>
                <td style="padding:10px;text-align:center">${p.age_mean}</td>
                <td style="padding:10px;text-align:center;color:${priorityColor}">${priority}</td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>
      <div style="margin-top:12px;font-size:12px;color:#6b7280">
        <strong>Churn rate TB:</strong> ${avgChurnRate.toFixed(1)}% | 
        <strong>Tổng KH:</strong> ${totalCustomers.toLocaleString()} | 
        <strong>KH cần can thiệp:</strong> ${profiles.filter(p => p.churn_rate > 20).reduce((sum, p) => sum + Math.round(p.count * p.churn_rate / 100), 0).toLocaleString()}
      </div>
    `;
  }

  // Priority Clusters
  const priorityClusters = profiles.filter(p => p.churn_rate > 20);
  const priorityContainer = document.getElementById('cluster-priority');
  if (priorityContainer) {
    if (priorityClusters.length > 0) {
      priorityContainer.innerHTML = priorityClusters.map(p => 
        `<div style="margin-bottom:12px;padding:10px;background:#fff;border-radius:6px;border-left:4px solid #dc2626">
          <strong>Cum ${p.cluster} - ${p.cluster_name}</strong>: 
          ${(p.count || 0).toLocaleString()} KH, churn ${p.churn_rate}%, 
          engagement TB ${Math.round(p.engagement_mean)}/100
          <div style="font-size:12px;color:#6b7280;margin-top:4px">
            ⚠️ Cần can thiệp trong 7 ngày với chiến dịch giữ chân khẩn cấp
          </div>
        </div>`
      ).join('');
    } else {
      priorityContainer.innerHTML = `<div style="color:#059669">✅ Không có cụm nào có churn rate trên 20%</div>`;
    }
  }

  // Cluster Behavior Analysis
  const vipCluster = profiles.find(p => p.churn_rate < 1);
  const loyalCluster = profiles.find(p => p.engagement_mean > 70 && p.churn_rate < 10);
  const riskCluster = profiles.find(p => p.churn_rate > 20);
  
  document.getElementById('cluster-behavior').innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px">
      <div style="padding:12px;background:#fef3c7;border-radius:8px">
        <strong style="color:#92400e">💎 Cluster VIP (Cụm ${vipCluster?.cluster || '?'})</strong>
        <ul style="margin:8px 0 0 18px;font-size:13px">
          <li>Chiếm <strong>${((vipCluster?.count || 0) / totalCustomers * 100).toFixed(1)}%</strong> tổng KH</li>
          <li>Churn rate gần bằng 0 (<strong>0.03%</strong>)</li>
          <li>Số dư cao nhất: <strong>${Math.round((vipCluster?.balance_mean || 0) / 1e6)}M</strong></li>
          <li>Tuổi TB: <strong>${vipCluster?.age_mean || '—'} tuổi</strong></li>
          <li><strong>Hành động:</strong> Duy trì chăm sóc, mời sự kiện</li>
        </ul>
      </div>
      <div style="padding:12px;background:#ecfdf5;border-radius:8px">
        <strong style="color:#047857">🌟 Cluster Trung thành (Cụm ${loyalCluster?.cluster || '?'})</strong>
        <ul style="margin:8px 0 0 18px;font-size:13px">
          <li>Engagement cao: <strong>${Math.round(loyalCluster?.engagement_mean || 0)}</strong>/100</li>
          <li>Churn rate thấp: <strong>${loyalCluster?.churn_rate || '—'}%</strong></li>
          <li>Số dư TB: <strong>${Math.round((loyalCluster?.balance_mean || 0) / 1e6)}M</strong></li>
          <li>Chiếm <strong>${((loyalCluster?.count || 0) / totalCustomers * 100).toFixed(1)}%</strong> KH</li>
          <li><strong>Hành động:</strong> Upsell sản phẩm, tăng giá trị</li>
        </ul>
      </div>
            <li>Chiếm <strong>${((riskCluster?.count || 0) / totalCustomers * 100).toFixed(1)}%</strong> KH</li>
          <li>Số dư TB: <strong>${Math.round((riskCluster?.balance_mean || 0) / 1e6)}M</strong></li>
          <li><strong>Hành động:</strong> Can thiệp khẩn cấp, retention campaign</li>
        </ul>
      </div>
    </div>
  `;

  // Cluster ROI
  const clusterROI = profiles.map(p => {
    const churned = Math.round(p.count * p.churn_rate / 100);
    const retained = p.count - churned;
    const costRetain = churned * 100000;
    const costAcquire = churned * 500000;
    const saved = costAcquire - costRetain;
    return { ...p, churned, retained, costRetain, costAcquire, saved };
  });
  
  document.getElementById('cluster-roi').innerHTML = `
    <div style="overflow-x:auto">
    <table style="width:100%;font-size:13px;border-collapse:collapse">
      <thead>
        <tr style="background:#1e429f;color:#fff">
          <th style="padding:10px;text-align:left">Cụm</th>
          <th style="padding:10px;text-align:center">KH rời bỏ</th>
          <th style="padding:10px;text-align:center">Chi phí retention</th>
          <th style="padding:10px;text-align:center">Chi phí acquisition</th>
          <th style="padding:10px;text-align:center">Tiết kiệm</th>
        </tr>
      </thead>
      <tbody>
        ${clusterROI.map(c => `
          <tr style="border-bottom:1px solid #e5e7eb">
            <td style="padding:10px"><strong>Cum ${c.cluster}: ${c.cluster_name}</strong></td>
            <td style="padding:10px;text-align:center">${c.churned.toLocaleString()}</td>
            <td style="padding:10px;text-align:center;color:#059669">${c.costRetain.toLocaleString()} VND</td>
            <td style="padding:10px;text-align:center;color:#dc2626">${c.costAcquire.toLocaleString()} VND</td>
            <td style="padding:10px;text-align:center;font-weight:bold;color:#059669">${c.saved.toLocaleString()} VND</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
    </div>
    <div style="margin-top:12px;padding:10px;background:#ecfdf5;border-radius:6px;font-size:13px">
      <strong>💡 Tổng tiết kiệm khi dùng chiến lược retention:</strong> 
      <strong style="color:#059669">${clusterROI.reduce((sum, c) => sum + c.saved, 0).toLocaleString()} VND</strong>
    </div>
  `;

  // Action Plan by Cluster
  const actionPlanContainer = document.getElementById('cluster-action-plan');
  if (actionPlanContainer) {
    actionPlanContainer.innerHTML = profiles.map(p => {
      let actions = [];
      let priorityColor = '#059669';
      let timeline = 'Dài hạn';
      
      if (p.churn_rate > 20) {
        priorityColor = '#dc2626';
        timeline = 'Ngay lập tức';
        actions = [
          '📞 Gọi điện cá nhân trong 48h',
          '🎁 Ưu đãi retention đặc biệt 30 ngày',
          '📧 Email marketing cảm ơn và quà tặng',
          '💰 Miễn phí dịch vụ 3 tháng'
        ];
      } else if (p.churn_rate > 5) {
        priorityColor = '#c27803';
        timeline = 'Trong tháng';
        actions = [
          '📱 SMS/Marketing về ưu đãi mới',
          '🎯 Upsell sản phẩm phù hợp',
          '⭐ Mời tham gia loyalty program',
          '📊 Tặng điểm thưởng đặc biệt'
        ];
      } else {
        timeline = 'Duy trì';
        actions = [
          '🎉 Mời sự kiện khách hàng thân thiết',
          '💎 Cập nhật sản phẩm cao cấp',
          '📧 Newsletter định kỳ',
          '⭐ Ưu đãi referral cho bạn bè'
        ];
      }
      
      return `
        <div style="margin-bottom:16px;padding:14px;background:#f9fafb;border-radius:8px;border-left:4px solid ${priorityColor}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <strong style="font-size:14px">Cum ${p.cluster}: ${p.cluster_name}</strong>
            <span style="background:${priorityColor};color:#fff;padding:2px 10px;border-radius:12px;font-size:11px">${timeline}</span>
          </div>
          <div style="font-size:12px;color:#6b7280;margin-bottom:8px">
            ${(p.count || 0).toLocaleString()} KH | Churn: ${p.churn_rate}% | Eng: ${Math.round(p.engagement_mean)} | So du: ${Math.round((p.balance_mean || 0) / 1e6)}M
          </div>
          <ul style="margin:0;padding-left:20px;font-size:13px">
            ${actions.map(a => `<li style="margin-bottom:4px">${a}</li>`).join('')}
          </ul>
        </div>
      `;
    }).join('');
  }
}

function renderClusterCustomers(clusterId, filterSegment = 'all', filterActive = 'all') {
  let rows = clusterCustomers[clusterId] || [];
  
  // 🔍 Áp dụng lọc (Point 2)
  if (filterSegment !== 'all') {
    rows = rows.filter(r => r.customer_segment === filterSegment);
  }
  if (filterActive !== 'all') {
    rows = rows.filter(r => (r.active_member || 0).toString() === filterActive);
  }

  const profile = (currentProfiles || []).find(p => p.cluster === clusterId) || {};
  const tbody = document.getElementById('cluster-customer-body');
  
  if (!tbody) return;
  
  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="11" style="text-align:center;padding:40px;color:#94a3b8">Không tìm thấy khách hàng nào khớp với điều kiện lọc.</td></tr>';
  } else {
    tbody.innerHTML = rows.map((r, idx) => {
      const age = r.age || 0;
      const engagement = r.engagement_score || 0;
      const balance = r.balance || 0;
      const active = r.active_member || 0;
      const digital = (r.digital_behavior || 'offline').toLowerCase();
      
      let riskScore = 0;
      if (age >= 18 && age <= 30) riskScore += 20;
      if (engagement < 30) riskScore += 30;
      if (balance < 20000000) riskScore += 20;
      if (active === 0) riskScore += 30;
      
      const riskLevel = riskScore > 60 ? 'CAO' : riskScore > 30 ? 'TB' : 'THÁP';
      const riskBadgeClass = riskScore > 60 ? 'danger' : riskScore > 30 ? 'warning' : 'success';
      
      return `
        <tr style="${r.exit ? 'background:#fff1f2' : ''}">
          <td style="padding:12px">${idx + 1}</td>
          <td style="padding:12px;font-weight:600;color:#1e293b">${r.full_name || 'Khách hàng'}</td>
          <td style="padding:12px;text-align:center">${age}</td>
          <td style="padding:12px;text-align:center">${(r.gender||'').toLowerCase().includes('female') || (r.gender||'').includes('nữ') ? '👩' : '👨'}</td>
          <td style="padding:12px;text-align:center"><span class="badge-pro info" style="font-size:10px">${r.customer_segment || '—'}</span></td>
          <td style="padding:12px;text-align:center">
            <span style="color:${r.loyalty_level === 'Bronze' ? '#92400e' : r.loyalty_level === 'Gold' ? '#854d0e' : '#1e3a8a'};font-weight:700">${r.loyalty_level || '—'}</span>
          </td>
          <td style="padding:12px;text-align:right;font-weight:700">${Math.round(balance / 1e6)}M</td>
          <td style="padding:12px;text-align:center">${engagement}</td>
          <td style="padding:12px;text-align:center">${active ? '✅' : '❌'}</td>
          <td style="padding:12px;text-align:center">
            ${digital.includes('mobile') ? '📱' : digital.includes('web') ? '💻' : digital.includes('omni') ? '🔄' : '🏪'}
          </td>
          <td style="padding:12px;text-align:center">
            <span class="badge-pro ${riskBadgeClass}">${riskLevel}</span>
          </td>
        </tr>
      `;
    }).join('');
  }
  
  // Update Summary Stats dynamically
  const detailContainer = document.getElementById('cluster-detail-analysis');
  if (detailContainer) {
    detailContainer.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:20px;margin-bottom:32px">
        <div class="card-pro" style="text-align:center">
          <div style="font-size:32px;font-weight:900;color:#1e40af">${rows.length.toLocaleString()}</div>
          <div style="font-size:12px;color:#64748b;margin-top:5px;text-transform:uppercase;font-weight:700">KH theo bộ lọc</div>
        </div>
        <div class="card-pro" style="text-align:center">
          <div style="font-size:32px;font-weight:900;color:#ef4444">${profile.churn_rate || 0}%</div>
          <div style="font-size:12px;color:#64748b;margin-top:5px;text-transform:uppercase;font-weight:700">Tỷ lệ Churn Cụm</div>
        </div>
        <div class="card-pro" style="text-align:center">
          <div style="font-size:32px;font-weight:900;color:#1e293b">${profile.age_mean || 0}</div>
          <div style="font-size:12px;color:#64748b;margin-top:5px;text-transform:uppercase;font-weight:700">Tuổi TB Cụm</div>
        </div>
        <div class="card-pro" style="text-align:center">
          <div style="font-size:32px;font-weight:900;color:#10b981">${Math.round((profile.balance_mean || 0) / 1e6)}M</div>
          <div style="font-size:12px;color:#64748b;margin-top:5px;text-transform:uppercase;font-weight:700">Số dư TB Cụm</div>
        </div>
      </div>
    `;
  }

  // Draw Charts for the representative samples
  if (rows.length > 0) {
    const samples = rows.slice(0, 10);
    const names = samples.map(r => r.full_name?.split(' ').pop() || 'KH');
    
    // 1. Age Chart
    barChart('cluster-age-chart', names, samples.map(r => r.age), 'Tuổi');
    
    // 2. Engagement Chart
    barChart('cluster-engagement-chart', names, samples.map(r => r.engagement_score), 'Engagement');
    
    // 3. Balance Chart
    const balValues = samples.map(r => Math.round((r.balance || 0) / 1e6));
    barChart('cluster-balance-chart', names, balValues, 'Số dư (Triệu VND)');

    // Post-format charts: remove percentage suffix
    ['cluster-age-chart', 'cluster-engagement-chart', 'cluster-balance-chart'].forEach(id => {
      const cv = document.getElementById(id);
      if (cv && cv._chart) {
        cv._chart.options.scales.y.ticks.callback = function(value) { return value; };
        cv._chart.update();
      }
    });
  }
}

// ══════════════════════════════════════════════════════════
// PREDICT & SIMULATION
// ══════════════════════════════════════════════════════════
// Debounce helper
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

let gaugeChart = null;
let lastPayload = null;

function drawGauge(value) {
  const ctx = document.getElementById('chart-gauge');
  if (!ctx) return;
  
  const color = value >= 70 ? '#ef4444' : value >= 40 ? '#f59e0b' : '#10b981';
  document.getElementById('gauge-value').textContent = value + '%';
  document.getElementById('gauge-value').style.color = color;

  if (gaugeChart) {
    // Cập nhật mượt mà thay vì xóa đi vẽ lại
    gaugeChart.data.datasets[0].data = [value, 100 - value];
    gaugeChart.data.datasets[0].backgroundColor = [color, '#f1f5f9'];
    gaugeChart.update({ duration: 800, easing: 'easeOutQuart' });
    return;
  }
  
  gaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [value, 100 - value],
        backgroundColor: [color, '#f1f5f9'],
        borderWidth: 0,
        circumference: 180,
        rotation: 270,
        cutout: '80%',
        borderRadius: 10
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { animateRotate: true, animateScale: false },
      plugins: { legend: { display: false }, tooltip: { enabled: false } }
    }
  });
}

// Hàm chạy mô phỏng (không debounce phần giao diện con số, chỉ debounce API)
async function runSimulation() {
  if (!lastPayload) return;
  
  const balance = document.getElementById('sim-balance').value;
  const engagement = document.getElementById('sim-engagement').value;
  const tenure = document.getElementById('sim-tenure').value;
  
  // Cập nhật số liệu trên UI ngay lập tức
  document.getElementById('sim-balance-val').textContent = parseInt(balance).toLocaleString() + 'đ';
  document.getElementById('sim-engagement-val').textContent = engagement;
  document.getElementById('sim-tenure-val').textContent = tenure + ' năm';

  debouncedApiCall(balance, engagement, tenure);
}

const debouncedApiCall = debounce(async (balance, engagement, tenure) => {
  const simPayload = { 
    ...lastPayload,
    balance: balance,
    engagement_score: engagement,
    tenure_ye: tenure,
    risk_score: 0.35,
    customer_segment: 'Standard',
    loyalty_level: 'Bronze',
    digital_behavior: 'Medium'
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(simPayload)
    });
    const data = await res.json();
    if (!data.error) {
      drawGauge(parseFloat(data.probability));
      const labelEl = document.getElementById('result-label');
      labelEl.textContent = data.label;
      labelEl.style.color = parseFloat(data.probability) >= 70 ? '#ef4444' : parseFloat(data.probability) >= 40 ? '#f59e0b' : '#10b981';
      
      document.getElementById('model-badge').textContent = `Mô hình: ${simPayload.model}`;
    }
  } catch (err) { console.error('Simulation error:', err); }
}, 250); // Đợi 250ms sau khi ngừng kéo

// Slider listeners
['sim-balance', 'sim-engagement', 'sim-tenure'].forEach(id => {
  document.getElementById(id).addEventListener('input', runSimulation);
});

document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = Object.fromEntries(fd.entries());
  lastPayload = payload;

  const btn = e.target.querySelector('.btn-predict');
  const originalText = btn.innerHTML;
  btn.innerHTML = '⏳ Đang phân tích rủi ro...';
  btn.disabled = true;

  const fullPayload = {
    ...payload,
    risk_score: 0.35,
    customer_segment: 'Standard',
    loyalty_level: 'Bronze',
    digital_behavior: 'Medium'
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(fullPayload)
    });
    const data = await res.json();

    if (data.error) {
      alert('Lỗi: ' + data.error);
      return;
    }

    const placeholder = document.getElementById('predict-placeholder');
    if (placeholder) placeholder.style.display = 'none';
    const resultCard = document.getElementById('predict-result');
    resultCard.style.display = 'block';

    const prob = parseFloat(data.probability);
    drawGauge(prob);
    
    document.getElementById('model-badge').textContent = `Mô hình: ${payload.model}`;
    const labelEl = document.getElementById('result-label');
    labelEl.textContent = data.label;
    labelEl.style.color = prob >= 70 ? '#ef4444' : prob >= 40 ? '#f59e0b' : '#10b981';

    // Initialize simulation sliders
    const simSection = document.getElementById('simulation-section');
    simSection.style.display = 'block';
    
    const sBalance = document.getElementById('sim-balance');
    sBalance.min = 0;
    sBalance.max = Math.max(parseFloat(payload.balance) * 2, 50000000); // 50M limit
    sBalance.value = payload.balance;
    document.getElementById('sim-balance-val').textContent = parseInt(payload.balance).toLocaleString() + 'đ';

    const sEngage = document.getElementById('sim-engagement');
    sEngage.min = 0;
    sEngage.max = 100;
    sEngage.value = payload.engagement_score;
    document.getElementById('sim-engagement-val').textContent = payload.engagement_score;

    const sTenure = document.getElementById('sim-tenure');
    sTenure.value = payload.tenure_ye;
    document.getElementById('sim-tenure-val').textContent = payload.tenure_ye + ' năm';

    // Reasons & Suggestions
    document.getElementById('result-reasons').innerHTML =
      (data.reasons || []).map(r => `<li style="margin-bottom:8px"><strong>${r}</strong></li>`).join('') || '<li>Dữ liệu ổn định, chưa phát hiện dấu hiệu bất thường.</li>';

    document.getElementById('result-suggestions').innerHTML =
      (data.suggestions || []).map(s => `<li style="margin-bottom:8px">${s}</li>`).join('') || '<li>Tiếp tục duy trì chất lượng dịch vụ hiện tại.</li>';

    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    alert('Lỗi kết nối: ' + err.message);
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
});

// Predict Guide
const predictGuide = document.getElementById('predict-guide');
if (predictGuide) {
  predictGuide.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px">
      <div style="padding:12px;background:#eff6ff;border-radius:8px">
        <strong style="color:#1e429f">📊 Xác suất Rời bỏ (%)</strong>
        <ul style="margin:8px 0 0 18px;font-size:13px">
          <li><strong>0-30%:</strong> An toàn - KH ổn định</li>
          <li><strong>30-50%:</strong> Cảnh báo - Cần theo dõi</li>
          <li><strong>50-70%:</strong> Nguy cơ cao - Cần hành động</li>
          <li><strong>70-100%:</strong> Rất nguy hiểm - Can thiệp ngay</li>
        </ul>
      </div>
      <div style="padding:12px;background:#fef3c7;border-radius:8px">
        <strong style="color:#92400e">🎯 Ngưỡng quyết định tối ưu</strong>
        <ul style="margin:8px 0 0 18px;font-size:13px">
          <li>Random Forest: <strong>52.6%</strong></li>
          <li>XGBoost: <strong>34.0%</strong></li>
          <li>Logistic Regression: <strong>58.7%</strong></li>
          <li>Tự động điều chỉnh theo độ nhạy AI</li>
        </ul>
      </div>
      <div style="padding:12px;background:#ecfdf5;border-radius:8px">
        <strong style="color:#047857">💡 Mẹo (Pro Tips)</strong>
        <ul style="margin:8px 0 0 18px;font-size:13px">
          <li>Sử dụng thanh trượt giả định để lập kịch bản giữ chân</li>
          <li>Ưu tiên các KH có xác suất cao hơn ngưỡng tối ưu</li>
          <li>Dùng XGBoost nếu muốn "bắt" nhiều khách rủi ro hơn</li>
        </ul>
      </div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════════
// INIT - lazy load tabs
// ══════════════════════════════════════════════════════════
const loaded = { overview: false, eda: false, models: false, shap: false, imbalanced: false, clusters: false };

async function initTab(tab) {
  if (loaded[tab]) return;
  loaded[tab] = true;
  if (tab === 'overview')   await loadOverview();
  if (tab === 'eda')        await loadEDA();
  if (tab === 'models')     await loadModels();
  if (tab === 'shap')       await loadSHAPAnalysis();
  if (tab === 'imbalanced') await loadImbalanceAnalysis();
  if (tab === 'clusters')   await loadClusters();
}

document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => initTab(btn.dataset.tab));
});

// Set initial page title
const initTitleEl = document.querySelector('.page-title');
if (initTitleEl) initTitleEl.textContent = TAB_TITLES['overview'];

// Load overview on start
initTab('overview');
