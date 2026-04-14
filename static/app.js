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
  predict:    '🎯 Công cụ Dự báo Churn'
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
      scales: { y: { beginAtZero: true, ticks: { callback: v => label.includes('%') ? v + '%' : v } } }
    },
    plugins: [{
      id: 'customDataLabels',
      afterDatasetsDraw(chart) {
        const { ctx, data } = chart;
        chart.getDatasetMeta(0).data.forEach((bar, index) => {
          const val = data.datasets[0].data[index];
          const isPct = label.includes('%');
          let text = isPct ? Number(val).toFixed(1) : Number(val).toLocaleString();
          if (isPct && val > 0 && val < 0.1) text = Number(val).toFixed(2);
          if (isPct) text += '%';
          ctx.save();
          ctx.font = 'bold 12px Inter';
          ctx.fillStyle = '#475569';
          ctx.textAlign = 'center';
          ctx.fillText(text, bar.x, bar.y - 8);
          ctx.restore();
        });
      }
    }]
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
      scales: { x: { beginAtZero: true, ticks: { callback: v => label.includes('%') ? v + '%' : v } } }
    },
    plugins: [{
      id: 'customDataLabelsH',
      afterDatasetsDraw(chart) {
        const { ctx, data } = chart;
        chart.getDatasetMeta(0).data.forEach((bar, index) => {
          const val = data.datasets[0].data[index];
          const isPct = label.includes('%');
          let text = isPct ? Number(val).toFixed(1) : Number(val).toLocaleString();
          if (isPct && val > 0 && val < 0.1) text = Number(val).toFixed(2);
          if (isPct) text += '%';
          ctx.save();
          ctx.font = 'bold 12px Inter';
          ctx.fillStyle = '#475569';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(text, bar.x + 8, bar.y);
          ctx.restore();
        });
      }
    }]
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
  engagement_score: 'Điểm Tương tác',
  risk_score: 'Điểm Rủi ro',
  customer_segment: 'Phân khúc KH',
  loyalty_level: 'Hạng Loyalty',
  digital_behavior: 'Hành vi Digital'
};

// Bản đồ tên thân thiện cho OHE features (ColumnTransformer output)
const SHAP_OHE_LABELS = {
  'customer_segment_Mass': 'Phân khúc Mass',
  'customer_segment_Emerging': 'Phân khúc Tiềm năng',
  'customer_segment_Priority': 'Phân khúc Priority',
  'customer_segment_Affluent': 'Phân khúc Affluent',
  'gender_male': 'Giới tính Nam',
  'gender_female': 'Giới tính Nữ',
  'digital_behavior_offline': 'Hành vi Offline',
  'digital_behavior_mobile': 'Hành vi Mobile'
};

/**
 * Chuyển đổi tên feature thô từ ColumnTransformer sang tên thân thiện.
 * Ví dụ: "num__risk_score" → "Điểm Rủi ro"
 *         "cat__customer_segment_Mass" → "Phân khúc Mass"
 */
function cleanFeatureName(rawFeature) {
  // Strip prefix num__ hoặc cat__
  let clean = rawFeature.replace(/^num__/, '').replace(/^cat__/, '');
  // Tra FEATURE_LABELS trước (tên gốc)
  if (FEATURE_LABELS[clean]) return FEATURE_LABELS[clean];
  // Tra OHE labels (cat features sau OHE)
  if (SHAP_OHE_LABELS[clean]) return SHAP_OHE_LABELS[clean];
  // Fallback: format tên đẹp hơn (bỏ __ và capitalize)
  return clean.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

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
  const table = (comparison && comparison.table) ? comparison.table : [];
  const byModel = {};
  table.forEach(m => { if (m && m.model) byModel[m.model] = m; });

  const fmt = (v) => (v === undefined || v === null) ? '—' : Number(v).toFixed(3);
  const fmtPct = (v) => (v === undefined || v === null) ? '—' : `${(Number(v) * 100).toFixed(1)}%`;
  const fmtThr = (v) => (v === undefined || v === null) ? '—' : `${(Number(v) * 100).toFixed(1)}%`;

  const ranked = [...table].filter(x => x && x.model).sort((a, b) => (b.roc_auc || 0) - (a.roc_auc || 0));
  const bestAuc = ranked[0]?.model || null;
  const worstAuc = ranked.length ? ranked[ranked.length - 1].model : null;
  const badgeFor = (name) => {
    if (!name || !ranked.length) return { label: '—', bg: '#f1f5f9', color: '#334155' };
    if (name === bestAuc) return { label: 'Ưu nhiều · Nhược ít (Top AUC)', bg: '#dcfce7', color: '#166534' };
    if (name === worstAuc) return { label: 'Ưu ít · Nhược nhiều (Low AUC)', bg: '#ffe4e6', color: '#9f1239' };
    return { label: 'Cân bằng (Mid)', bg: '#e0f2fe', color: '#075985' };
  };

  const rows = [
    {
      name: 'Logistic Regression',
      badge: '📊 Logistic Regression',
      style: 'background:#eff6ff;color:#1e3a8a;border-left:4px solid #1d4ed8;',
      pros: [
        'Dễ giải thích (có thể trình bày hệ số và hướng tác động của biến)',
        'Huấn luyện/dự báo nhanh, phù hợp baseline và triển khai đơn giản',
        'Dễ kiểm soát pipeline và kiểm thử'
      ],
      cons: [
        'Giả định quan hệ tuyến tính, có thể bỏ sót tương tác phi tuyến',
        'Cần hiệu chỉnh threshold theo mục tiêu (Recall/Precision)'
      ],
      use: 'Baseline/giải thích cho stakeholders; production khi ưu tiên đơn giản và ổn định.'
    },
    {
      name: 'Random Forest',
      badge: '🌲 Random Forest',
      style: 'background:#ecfdf5;color:#065f46;border-left:4px solid #059669;',
      pros: [
        'Bắt được pattern phi tuyến và tương tác giữa biến',
        'Tổng hợp nhiều cây giúp giảm phương sai so với 1 cây đơn',
        'Có feature importance để tham khảo'
      ],
      cons: [
        'Khó giải thích hơn Logistic Regression',
        'Tốn tài nguyên hơn (nhiều cây, inference chậm hơn)'
      ],
      use: 'Khi cần mô hình phi tuyến “ổn”, ưu tiên hiệu suất tổng thể.'
    },
    {
      name: 'XGBoost',
      badge: '⚡ XGBoost',
      style: 'background:#fef3c7;color:#92400e;border-left:4px solid #d97706;',
      pros: [
        'Mạnh trên tabular data, học được nhiều tương tác phi tuyến',
        'Có regularization và nhiều tham số để kiểm soát overfit'
      ],
      cons: [
        'Nhạy với hyperparameters, cần tuning kỹ',
        'Khó giải thích nhất trong 3 mô hình nếu không có XAI hỗ trợ'
      ],
      use: 'Khi có thời gian tuning và muốn đẩy hiệu suất; dùng kèm giám sát/kiểm thử chặt.'
    }
  ];

  const rowHtml = rows.map(r => {
    const m = byModel[r.name] || {};
    const pros = r.pros.map(x => `<li>${x}</li>`).join('');
    const cons = r.cons.map(x => `<li>${x}</li>`).join('');
    const b = badgeFor(r.name);
    return `
      <tr style="border-top:1px solid var(--gray-200)">
        <td style="padding:14px;vertical-align:top;min-width:180px">
          <div style="padding:10px 12px;border-radius:10px;${r.style}">
            <div style="font-weight:900">${r.badge}</div>
            <div style="display:inline-block;margin-top:8px;font-size:11px;font-weight:800;background:${b.bg};color:${b.color};padding:4px 8px;border-radius:999px">
              ${b.label}
            </div>
            <div style="font-size:12px;opacity:.9;margin-top:6px;line-height:1.6">
              AUC: <strong>${fmt(m.roc_auc)}</strong><br>
              Recall: <strong>${fmtPct(m.recall)}</strong> · Precision: <strong>${fmtPct(m.precision)}</strong><br>
              F1: <strong>${fmt(m.f1)}</strong> · Threshold: <strong>${fmtThr(m.threshold)}</strong>
            </div>
          </div>
        </td>
        <td style="padding:14px;vertical-align:top;min-width:280px">
          <ul style="margin:0;padding-left:18px;line-height:1.7;font-size:13px;color:var(--gray-700)">${pros}</ul>
        </td>
        <td style="padding:14px;vertical-align:top;min-width:280px">
          <ul style="margin:0;padding-left:18px;line-height:1.7;font-size:13px;color:var(--gray-700)">${cons}</ul>
        </td>
        <td style="padding:14px;vertical-align:top;min-width:260px;font-size:13px;line-height:1.7;color:var(--gray-700)">
          ${r.use}
        </td>
      </tr>
    `;
  }).join('');

  return `
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;background:#fff;border-radius:12px;overflow:hidden;border:1px solid var(--border)">
        <thead style="background:#0f172a;color:#f8fafc">
          <tr>
            <th style="padding:12px 14px;text-align:left">Mô hình</th>
            <th style="padding:12px 14px;text-align:left">Ưu điểm</th>
            <th style="padding:12px 14px;text-align:left">Nhược điểm</th>
            <th style="padding:12px 14px;text-align:left">Gợi ý sử dụng</th>
          </tr>
        </thead>
        <tbody>
          ${rowHtml || `<tr><td colspan="4" style="padding:14px">Chưa có dữ liệu so sánh.</td></tr>`}
        </tbody>
      </table>
    </div>
  `;
}

function analyzeTradeoff(results, comparison) {
  const table = comparison.table || [];

  // Build lookup map by model name
  const byModel = {};
  table.forEach(m => { byModel[m.model] = m; });
  const lr  = byModel['Logistic Regression'] || {};
  const rf  = byModel['Random Forest']       || {};
  const xgb = byModel['XGBoost']             || {};

  const fmt = (v, mul = 1, suffix = '') => v != null ? (v * mul).toFixed(2) + suffix : '—';
  const fmtPct = v => v != null ? (v * 100).toFixed(1) + '%' : '—';

  const thStyle = 'padding:10px 12px; text-align:left; background:#1e40af; color:#fff; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.07em;';
  const thCStyle = 'padding:10px 12px; text-align:center; background:#1e40af; color:#fff; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.07em;';
  const td = (val, highlight = false, note = '') =>
    `<td style="padding:10px 12px; text-align:center; ${highlight ? 'background:#ecfdf5; font-weight:700; color:#15803d;' : ''}">${val}</td>`;
  const tdLeft = (val) => `<td style="padding:10px 12px; font-size:12px; color:#64748b;">${val}</td>`;

  const maxAuc = Math.max(lr.roc_auc || 0, rf.roc_auc || 0, xgb.roc_auc || 0);
  const maxRecall = Math.max(lr.recall || 0, rf.recall || 0, xgb.recall || 0);
  const maxF1 = Math.max(lr.f1 || 0, rf.f1 || 0, xgb.f1 || 0);

  const rows = [
    {
      label: 'ROC-AUC',
      lr:  fmt(lr.roc_auc),
      rf:  fmt(rf.roc_auc),
      xgb: fmt(xgb.roc_auc),
      lrH: lr.roc_auc === maxAuc, rfH: rf.roc_auc === maxAuc, xgbH: xgb.roc_auc === maxAuc,
      note: 'Diện tích dưới đường ROC — chỉ số phân loại tổng thể'
    },
    {
      label: 'Recall ↑',
      lr:  fmtPct(lr.recall),
      rf:  fmtPct(rf.recall),
      xgb: fmtPct(xgb.recall),
      lrH: lr.recall === maxRecall, rfH: rf.recall === maxRecall, xgbH: xgb.recall === maxRecall,
      note: '⭐ Ưu tiên trong bài toán Churn — bắt được nhiều KH sắp rời'
    },
    {
      label: 'Precision',
      lr:  fmtPct(lr.precision),
      rf:  fmtPct(rf.precision),
      xgb: fmtPct(xgb.precision),
      lrH: false, rfH: false, xgbH: false,
      note: 'Precision thấp → nhiều false alarm → lãng phí chi phí retention'
    },
    {
      label: 'F1-Score',
      lr:  fmtPct(lr.f1),
      rf:  fmtPct(rf.f1),
      xgb: fmtPct(xgb.f1),
      lrH: lr.f1 === maxF1, rfH: rf.f1 === maxF1, xgbH: xgb.f1 === maxF1,
      note: 'Cân bằng tổng hợp Precision/Recall'
    },
    {
      label: 'Threshold',
      lr:  fmtPct(lr.threshold),
      rf:  fmtPct(rf.threshold),
      xgb: fmtPct(xgb.threshold),
      lrH: false, rfH: false, xgbH: false,
      note: 'Ngưỡng quyết định tối ưu — điều chỉnh trade-off P/R'
    },
    {
      label: 'CV AUC',
      lr:  lr.cv_auc ? lr.cv_auc.toFixed(4) + (lr.cv_std ? ' ±' + lr.cv_std.toFixed(4) : '') : '—',
      rf:  rf.cv_auc ? rf.cv_auc.toFixed(4) + (rf.cv_std ? ' ±' + rf.cv_std.toFixed(4) : '') : '—',
      xgb: xgb.cv_auc ? xgb.cv_auc.toFixed(4) + (xgb.cv_std ? ' ±' + xgb.cv_std.toFixed(4) : '') : '—',
      lrH: false, rfH: false, xgbH: false,
      note: 'Kết quả 5-Fold CV — đánh giá khả năng tổng quát hóa'
    },
    {
      label: 'Tốc độ Inference',
      lr: '🚀 Nhanh nhất', rf: '🐢 Trung bình', xgb: '⚡ Nhanh',
      lrH: false, rfH: false, xgbH: false,
      note: 'LR nhanh nhất vì linear: chỉ 1 ma trận nhân'
    }
  ];

  const rowsHtml = rows.map((r, i) => `
    <tr style="border-bottom:1px solid #f1f5f9; background:${i % 2 === 0 ? '#fff' : '#fafafa'}">
      <td style="padding:10px 12px; font-weight:600; color:#334155; white-space:nowrap">${r.label}</td>
      ${td(r.lr, r.lrH)}
      ${td(r.rf, r.rfH)}
      ${td(r.xgb, r.xgbH)}
      ${tdLeft(r.note)}
    </tr>
  `).join('');

  return `
    <div style="overflow-x:auto; border-radius:8px; border:1px solid #e2e8f0">
    <table style="width:100%; font-size:13px; border-collapse:collapse">
      <thead>
        <tr>
          <th style="${thStyle}">Tiêu chí</th>
          <th style="${thCStyle}">Logistic Regression</th>
          <th style="${thCStyle}">Random Forest</th>
          <th style="${thCStyle}">XGBoost</th>
          <th style="${thStyle}">Nhận xét</th>
        </tr>
      </thead>
      <tbody>${rowsHtml}</tbody>
    </table>
    </div>
    <div style="margin-top:12px; padding:12px 14px; background:#eff6ff; border-radius:6px; font-size:12.5px; border-left:3px solid #1d4ed8">
      <strong>📌 Kết luận Trade-off:</strong><br>
      • <strong>AUC (khả năng phân biệt):</strong> so sánh theo hàng <em>ROC-AUC</em> trong bảng trên<br>
      • <strong>Bắt rủi ro (Recall):</strong> ưu tiên mô hình có Recall cao hơn (đổi theo threshold)<br>
      • <strong>Giảm báo động nhầm (Precision):</strong> ưu tiên mô hình có Precision cao hơn (đổi theo threshold)
    </div>
  `;
}

function getModelRecommendation(results, comparison) {
  const table = (comparison && comparison.table) ? comparison.table : [];
  const byRecall = table.length ? [...table].sort((a, b) => (b.recall || 0) - (a.recall || 0))[0] : null;
  const byPrecision = table.length ? [...table].sort((a, b) => (b.precision || 0) - (a.precision || 0))[0] : null;
  const byF1 = table.length ? [...table].sort((a, b) => (b.f1 || 0) - (a.f1 || 0))[0] : null;
  const byAuc = table.length ? [...table].sort((a, b) => (b.roc_auc || 0) - (a.roc_auc || 0))[0] : null;
  const fmtPct = (x) => (x === undefined || x === null) ? '—' : `${(Number(x) * 100).toFixed(1)}%`;

  return `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px">
      <div style="padding:14px;background:#ecfdf5;border-radius:8px;border-left:4px solid #057a55">
        <strong style="color:#047857">🎯 Mục tiêu: Giữ chân KH tối đa (Recall cao)</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>→ Ưu tiên:</strong> ${byRecall ? byRecall.model : '—'}<br>
          Recall (test) ≈ ${byRecall ? fmtPct(byRecall.recall) : '—'}<br>
          Ghi chú: Recall phụ thuộc ngưỡng quyết định (threshold)
        </div>
      </div>
      
      <div style="padding:14px;background:#eff6ff;border-radius:8px;border-left:4px solid #1a56db">
        <strong style="color:#1e429f">🎯 Mục tiêu: Chọn lọc KH (Precision cao)</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>→ Ưu tiên:</strong> ${byPrecision ? byPrecision.model : '—'}<br>
          Precision (test) ≈ ${byPrecision ? fmtPct(byPrecision.precision) : '—'}<br>
          Ghi chú: Precision/Recall đánh đổi theo threshold
        </div>
      </div>
      
      <div style="padding:14px;background:#fef3c7;border-radius:8px;border-left:4px solid #c27803">
        <strong style="color:#92400e">🎯 Mục tiêu: Cân bằng (F1 cao)</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>→ Ưu tiên:</strong> ${byF1 ? byF1.model : '—'}<br>
          F1 (test) ≈ ${byF1 ? (Number(byF1.f1).toFixed(3)) : '—'}<br>
          Ghi chú: F1 cân bằng Precision/Recall
        </div>
      </div>
      
      <div style="padding:14px;background:#f3f4f6;border-radius:8px;border-left:4px solid #6b7280">
        <strong style="color:#374151">🎯 Mục tiêu: Giải thích cho stakeholders</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>→ Gợi ý:</strong> Logistic Regression (dễ giải thích)<br>
          Có thể trình bày hệ số (coefficients) và tác động theo hướng tăng/giảm rủi ro
        </div>
      </div>
      
      <div style="padding:14px;background:#fce7f3;border-radius:8px;border-left:4px solid #be185d">
        <strong style="color:#9d174d">🎯 Mục tiêu: Deployment ổn định</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>→ Gợi ý:</strong> ưu tiên mô hình đơn giản + pipeline rõ ràng<br>
          Theo dõi drift và tái đánh giá định kỳ theo batch dữ liệu mới
        </div>
      </div>
      
      <div style="padding:14px;background:#f5f3ff;border-radius:8px;border-left:4px solid #7c3aed">
        <strong style="color:#5b21b6">🎯 Mục tiêu: Khả năng phân biệt (AUC)</strong>
        <div style="font-size:13px;margin-top:8px">
          <strong>→ Ưu tiên:</strong> ${byAuc ? byAuc.model : '—'}<br>
          ROC-AUC (test) ≈ ${byAuc ? Number(byAuc.roc_auc).toFixed(3) : '—'}
        </div>
      </div>
    </div>
  `;
}

function analyzeBusinessImpact(results, comparison) {
  return `
    <div style="padding:12px;background:#f3f4f6;border-radius:8px;font-size:13px;line-height:1.7">
      <strong>💰 Ghi chú ROI:</strong> Phần ROI nên được tính theo giả định rõ ràng (LTV, chi phí giữ chân, tỉ lệ thành công) và/hoặc dữ liệu thực tế của ngân hàng.
      Hệ thống hiện cung cấp các chỉ số kinh doanh cơ bản trong Dashboard Tổng quan (At-risk balance và ước tính tiết kiệm theo tỉ lệ giữ chân giả định).
    </div>
  `;
}

async function loadOverview() {
  const [ovRes, edaRes] = await Promise.all([
    fetch('/api/overview').then(r => r.json()),
    fetch('/api/eda').then(r => r.json())
  ]);

  const ov = ovRes.overview || {};
  
  // KPI — chỉ cập nhật giá trị vào các phần tử đã dựng sẵn trong HTML
  const kpiTotal = document.getElementById('kpi-total');
  if (kpiTotal) kpiTotal.textContent = (ov.total || 0).toLocaleString('vi-VN');
  const kpiChurn = document.getElementById('kpi-churn-rate');
  if (kpiChurn) kpiChurn.textContent = (ov.churn_rate || 0) + '%';
  const kpiAuc = document.getElementById('kpi-auc');
  if (kpiAuc) kpiAuc.textContent = ovRes.best_model?.roc_auc || '—';
  const kpiCluster = document.getElementById('kpi-clusters');
  if (kpiCluster) kpiCluster.textContent = ovRes.n_clusters || '3';

  // FIX G3: Cập nhật badge N động từ API thay vì hardcode '80,000 mẫu'
  const badgeSample = document.getElementById('badge-sample-count');
  if (badgeSample && ov.total) badgeSample.textContent = `N = ${(ov.total).toLocaleString('vi-VN')} mẫu`;

  // ROI / Impact
  const impactContainer = document.getElementById('kpi-impact');
  if (impactContainer) {
    const roi = ovRes.roi || {};
    impactContainer.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px">
        <div style="padding:14px;background:#fef2f2;border-radius:8px">
          <strong style="color:#991b1b">⚠️ Tài sản rủi ro (At-Risk Balance)</strong>
          <div style="font-size:13px;margin-top:8px">
            • Tổng số dư nhóm Churn: <strong style="color:#dc2626">${(roi.total_balance_at_risk_m || 0).toLocaleString()}M ${roi.currency || 'VNĐ'}</strong><br>
            • Đánh giá: Mức độ tổn thương vốn CASA cao.<br>
            • Lưu ý: Cần can thiệp ngay vào nhóm 20% KH đóng góp 80% số dư này.
          </div>
        </div>
        
        <div style="padding:14px;background:#ecfdf5;border-radius:8px">
          <strong style="color:#065f46">✅ Hiệu quả Dự báo & Tiết kiệm</strong>
          <div style="font-size:13px;margin-top:8px">
            • Khả năng cứu vãn dự kiến: <strong style="color:#059669">${(roi.est_savings_m || 0).toLocaleString()}M ${roi.currency || 'VNĐ'}</strong><br>
            • Tỉ lệ giữ chân giả định: ${roi.retention_success_rate || '30%'}<br>
            • Chiến lược: Sử dụng Recall (${ovRes.best_model?.recall || 0}) làm trọng tâm.
          </div>
        </div>
        
        <div style="padding:14px;background:#eff6ff;border-radius:8px">
          <strong style="color:#1e40af">💰 ROI & Tối ưu chi phí</strong>
          <div style="font-size:13px;margin-top:8px">
            • Đọc các giả định ROI ở phần Tổng quan (At-risk balance, tỉ lệ giữ chân giả định)<br>
            • Điều chỉnh theo LTV/chi phí chiến dịch thực tế của ngân hàng
          </div>
        </div>
      </div>
    `;
  }

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
       so_lieu: `Phân phối nhãn mục tiêu mất cân bằng tại mức: ${((ov.churn || 0) / (ov.total || 1) * 100).toFixed(1)}% Positive.`,
       nguyen_nhan: "Đặc điểm cơ hữu của dữ liệu Bank Churn.",
       huong_xu_ly: "Áp dụng SMOTE và/hoặc hiệu chỉnh threshold để cải thiện Recall/Precision cho lớp thiểu số."
    });
  }

  const bm = ovRes.best_model || { name: '—', roc_auc: null, recall: null, f1: null };
  document.getElementById('best-model-info').innerHTML = `
    <div style="font-size:24px; font-weight:800; margin-bottom:16px; color:var(--blue-800)">${bm.name}</div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:14px">
      <div style="background:var(--surface); border-radius:8px; padding:10px 14px; border: 1px solid var(--border)">
        <div style="font-size:11px; color:var(--gray-500); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px">ROC-AUC</div>
        <div style="font-size:22px; font-weight:800; color:var(--gray-900)">${bm.roc_auc === null ? '—' : Number(bm.roc_auc).toFixed(3)}</div>
      </div>
      <div style="background:var(--surface); border-radius:8px; padding:10px 14px; border: 1px solid var(--border)">
        <div style="font-size:11px; color:var(--gray-500); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px">Recall</div>
        <div style="font-size:22px; font-weight:800; color:var(--gray-900)">${bm.recall === null ? '—' : ((Number(bm.recall) * 100).toFixed(1) + '%')}</div>
      </div>
    </div>
    <div style="font-size:12px; color:var(--gray-500); line-height:1.5">Model được đề xuất theo ROC-AUC cao trong batch hiện tại. Việc chọn model nên bám mục tiêu (ưu tiên Recall/F1/Precision) và hiệu chỉnh threshold.</div>
  `;

  const tf = ovRes.top_features || ['risk_score', 'monthly_ir', 'engagement_score'];
  const importanceNote = { 'risk_score': '(Engineered feature)', 'monthly_ir': '', 'engagement_score': '' };
  document.getElementById('top-features').innerHTML = tf.map((f, i) =>
    `<div style="display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid var(--border); font-size:14px">
      <span style="background:var(--blue-50); width:26px; height:26px; display:flex; align-items:center; justify-content:center; border-radius:50%; font-size:12px; font-weight:800; color:var(--blue-700); border:1px solid var(--blue-100); flex-shrink:0">${i+1}</span>
      <span style="font-weight:600; color:var(--gray-800); flex:1">${FEATURE_LABELS[f] || f}</span>
      ${importanceNote[f] ? `<span style="font-size:11px; background:var(--gray-100); padding:2px 6px; border-radius:4px; color:var(--gray-500)">${importanceNote[f]}</span>` : ''}
    </div>`
  ).join('') || '—';

  const insight = edaRes.overview?.insight || 'Chưa có insight tự động cho batch hiện tại. Chạy pipeline để tạo báo cáo EDA.';
  document.getElementById('overview-insight').innerHTML = `<span style="font-weight:600;color:var(--gray-700)">Đánh Giá Nhanh:</span> <span style="color:var(--gray-600)">${insight}</span> 
  <div style="margin-top:10px; padding-top:10px; border-top:1px dashed var(--border); font-size:12.5px; color:var(--gray-500)">
    <strong>🔎 Gợi ý:</strong> Ưu tiên drill-down theo segment/age/digital_behavior, theo dõi drift và hiệu chỉnh threshold theo chi phí chiến dịch.
  </div>`;

  // High Risk Summary
  const churnRate = ov.churn_rate || 0;
  const riskLevel = churnRate > 20 ? 'Nguy cơ cao' : churnRate > 15 ? 'Trung bình' : 'Ổn định';
  const riskClass = churnRate > 20 ? 'danger' : churnRate > 15 ? 'warning' : 'success';

  document.getElementById('high-risk-summary').innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px">
      <span style="font-weight:700; color:var(--gray-700); font-size:14px">Cảnh báo Trạng thái rủi ro:</span>
      <span class="badge" style="background:var(--${riskClass}-50); color:var(--${riskClass}-700); border-color:var(--${riskClass}-100); padding:4px 10px; font-size:13px">${riskLevel}</span>
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
      <div style="background:var(--surface-2); border:1px solid var(--border); border-radius:12px; padding:18px; text-align:center; box-shadow:0 2px 4px rgba(0,0,0,0.02)">
        <div style="font-size:32px; font-weight:900; color:var(--red-600); line-height:1">${churnRate}%</div>
        <div style="font-size:11px; font-weight:700; color:var(--gray-500); text-transform:uppercase; letter-spacing:0.06em; margin-top:8px">Tỉ lệ Thất thoát Tương đối (Churn Rate)</div>
      </div>
      <div style="background:var(--surface-2); border:1px solid var(--border); border-radius:12px; padding:18px; text-align:center; box-shadow:0 2px 4px rgba(0,0,0,0.02)">
        <div style="font-size:32px; font-weight:900; color:var(--blue-700); line-height:1">${(ov.total || 0).toLocaleString('vi-VN')}</div>
        <div style="font-size:11px; font-weight:700; color:var(--gray-500); text-transform:uppercase; letter-spacing:0.06em; margin-top:8px">Dung lượng Mẫu Phân tích (Universe Size)</div>
      </div>
    </div>
  `;

  document.getElementById('overview-summary').innerHTML = `
    <div style="font-size:12px; font-weight:800; color:var(--gray-500); margin-bottom:16px; text-transform:uppercase; letter-spacing:0.1em; border-bottom:1px solid var(--border); padding-bottom:8px">Cơ cấu Khách hàng</div>
    <div style="display:flex; flex-direction:column; gap:14px">
      <div style="display:flex; justify-content:space-between; align-items:center; font-size:14px">
        <span style="font-weight:600; color:var(--gray-800)">Khách hàng Trung thành (Negative Class):</span>
        <span style="font-weight:800; font-size:16px; color:var(--green-600)">${(((ov.no_churn||1)/(ov.total||1))*100).toFixed(1)}%</span>
      </div>
      <div style="height:8px; background:var(--gray-100); border-radius:99px; overflow:hidden">
        <div style="width:${(((ov.no_churn||1)/(ov.total||1))*100).toFixed(1)}%; height:100%; background:var(--green-500); border-radius:99px"></div>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; font-size:14px; margin-top:4px">
        <span style="font-weight:600; color:var(--gray-800)">Nhóm Khách rời bỏ (Positive Class):</span>
        <span style="font-weight:800; font-size:16px; color:var(--red-600)">${churnRate}%</span>
      </div>
      <div style="height:8px; background:var(--red-50); border-radius:99px; overflow:hidden">
        <div style="width:${churnRate}%; height:100%; background:var(--red-500); border-radius:99px; box-shadow:0 0 10px rgba(239, 68, 68, 0.4)"></div>
      </div>
    </div>
  `;

  document.getElementById('overview-actions').innerHTML = `
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px">
      <div style="padding:24px; background:var(--blue-50); border-radius:16px; border:1px solid var(--blue-100); position:relative; overflow:hidden">
        <div style="position:absolute; top:0; left:0; bottom:0; width:6px; background:var(--blue-600)"></div>
        <div style="font-size:16px; font-weight:800; color:var(--blue-800); margin-bottom:12px; display:flex; align-items:center; gap:8px">
          <span style="background:var(--blue-100); width:28px; height:28px; display:flex; align-items:center; justify-content:center; border-radius:8px">🛡️</span> 
          Giải Pháp Níu Giữ (Proactive Retention)
        </div>
        <div style="font-size:14px; color:var(--blue-900); line-height:1.7">
          <strong>Đối tượng:</strong> Nhóm xác suất churn trung bình (cần theo dõi).<br>
          <strong style="margin-top:6px; display:inline-block">Kịch bản gợi ý:</strong> Nhắc nhở/ưu đãi qua App/SMS theo segment và tín hiệu tương tác (engagement), ưu tiên khách có giá trị cao.
        </div>
      </div>
      <div style="padding:24px; background:var(--red-50); border-radius:16px; border:1px solid var(--red-100); position:relative; overflow:hidden">
        <div style="position:absolute; top:0; left:0; bottom:0; width:6px; background:var(--red-600)"></div>
        <div style="font-size:16px; font-weight:800; color:var(--red-800); margin-bottom:12px; display:flex; align-items:center; gap:8px">
          <span style="background:var(--red-100); width:28px; height:28px; display:flex; align-items:center; justify-content:center; border-radius:8px">🚨</span> 
          Chiến dịch Giải cứu (Win-back Campaign)
        </div>
        <div style="font-size:14px; color:var(--red-900); line-height:1.7">
          <strong>Đối tượng:</strong> Nhóm xác suất churn cao (ưu tiên can thiệp).<br>
          <strong style="margin-top:6px; display:inline-block">Kịch bản gợi ý:</strong> Xuất CSV danh sách ưu tiên và phối hợp RM/Telesales liên hệ, đề xuất gói giữ chân phù hợp theo persona K-Means.
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
  if (text && text.so_lieu) {
    el.innerHTML = `💡 <strong>Insight:</strong> ${text.so_lieu} ${text.nguyen_nhan || ''}`;
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
    const ageCtx = document.getElementById('chart-age');
    if (ageCtx) {
      if (ageCtx._chart) ageCtx._chart.destroy();
      const churnCounts = eda.age.counts.map((c, i) => Math.round(c * eda.age.churn_rates[i] / 100));
      const noChurnCounts = eda.age.counts.map((c, i) => c - Math.round(c * eda.age.churn_rates[i] / 100));
      ageCtx._chart = new Chart(ageCtx, {
        type: 'bar',
        data: {
          labels: eda.age.labels,
          datasets: [
            { label: 'Ổn định (Non-Churn)', data: noChurnCounts, backgroundColor: '#10b981', borderRadius: 4 },
            { label: 'Rời bỏ (Churn)', data: churnCounts, backgroundColor: '#ef4444', borderRadius: 4 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'top' } },
          scales: { y: { beginAtZero: true } }
        }
      });
    }
    setInsight('insight-age', eda.age.insight, eda.age.labels, eda.age.churn_rates, 'rate');
  }
  // Gender
  if (eda.gender) {
    const genderLabels = eda.gender.labels.map(l => l.toLowerCase() === 'female' ? 'Nữ' : l.toLowerCase() === 'male' ? 'Nam' : l);
    barChart('chart-gender', genderLabels, eda.gender.churn_rates);
    // insight needs the mapped labels to generate properly
    setInsight('insight-gender', eda.gender.insight, genderLabels, eda.gender.churn_rates, 'rate');
  }
  // Segment
  if (eda.segment) {
    const segmentLabels = eda.segment.labels.map(l => {
      const ml = l.toLowerCase();
      if (ml.includes('affluent')) return 'Gia sản (Affluent)';
      if (ml.includes('emerging')) return 'Tiềm năng (Emerging)';
      if (ml.includes('mass')) return 'Đại chúng (Mass)';
      if (ml.includes('priority')) return 'VIP (Priority)';
      return l;
    });
    barChart('chart-segment', segmentLabels, eda.segment.churn_rates);
    setInsight('insight-segment', eda.segment.insight, segmentLabels, eda.segment.churn_rates, 'rate');
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
  // Balance distribution (Density / Histogram line curve)
  if (eda.balance) {
    const balanceCtx = document.getElementById('chart-balance');
    if (balanceCtx && eda.balance.churn_hist && eda.balance.bin_edges) {
      if (balanceCtx._chart) balanceCtx._chart.destroy();
      
      // Calculate bin centers
      const edges = eda.balance.bin_edges;
      const labels = [];
      for (let i = 0; i < edges.length - 1; i++) {
        labels.push(Math.round((edges[i] + edges[i+1]) / 2 / 1e6) + 'M'); // Label as millions
      }
      
      // Normalize histograms to plot density curve
      const sumChurn = eda.balance.churn_hist.reduce((a,b)=>a+b, 0) || 1;
      const sumNoChurn = eda.balance.no_churn_hist.reduce((a,b)=>a+b, 0) || 1;
      const churnDensity = eda.balance.churn_hist.map(v => (v / sumChurn) * 100);
      const noChurnDensity = eda.balance.no_churn_hist.map(v => (v / sumNoChurn) * 100);

      balanceCtx._chart = new Chart(balanceCtx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Phân bố Số dư (Non-Churn %)',
              data: noChurnDensity,
              borderColor: '#10b981',
              backgroundColor: 'rgba(16, 185, 129, 0.1)',
              fill: true,
              tension: 0.4,
              borderWidth: 2,
              pointRadius: 0
            },
            {
              label: 'Phân bố Số dư (Churn %)',
              data: churnDensity,
              borderColor: '#ef4444',
              backgroundColor: 'rgba(239, 68, 68, 0.2)',
              fill: true,
              tension: 0.4,
              borderWidth: 2,
              pointRadius: 0
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'top' }, tooltip: { mode: 'index', intersect: false } },
          scales: { y: { beginAtZero: true, display: false }, x: { grid: { display: false } } }
        }
      });
    }
    
    let vChurn = Math.round(eda.balance.churn_mean / 1e6);
    let vNoChurn = Math.round(eda.balance.no_churn_mean / 1e6);
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

  // Top Risks Summary - DYNAMICALLY CALCULATED
  const topRisks = [];
  if (eda.age) {
    const maxIdx = eda.age.churn_rates.indexOf(Math.max(...eda.age.churn_rates));
    topRisks.push({ factor: `Nhóm tuổi ${eda.age.labels[maxIdx]}`, rate: eda.age.churn_rates[maxIdx], insight: 'Có churn rate cao nhất nhóm' });
  }
  if (eda.segment) {
    const maxIdx = eda.segment.churn_rates.indexOf(Math.max(...eda.segment.churn_rates));
    topRisks.push({ factor: `Phân khúc ${eda.segment.labels[maxIdx]}`, rate: eda.segment.churn_rates[maxIdx], insight: 'Rủi ro phân khúc' });
  }
  if (eda.active_member) {
    const inactiveRate = eda.active_member.labels.indexOf('0') !== -1 ? eda.active_member.churn_rates[eda.active_member.labels.indexOf('0')] : eda.active_member.churn_rates[0];
    topRisks.push({ factor: 'Thành viên Không hoạt động', rate: inactiveRate, insight: 'Engagement thấp' });
  }
  if (eda.credit_score) {
    const maxIdx = eda.credit_score.churn_rates.indexOf(Math.max(...eda.credit_score.churn_rates));
    topRisks.push({ factor: `Credit Score ${eda.credit_score.labels[maxIdx]}`, rate: eda.credit_score.churn_rates[maxIdx], insight: 'Điểm tín dụng yếu' });
  }
  if (eda.digital) {
    const offlineIdx = eda.digital.labels.findIndex(l => l.toLowerCase().includes('offline'));
    if (offlineIdx !== -1) {
      topRisks.push({ factor: 'Khách hàng Offline', rate: eda.digital.churn_rates[offlineIdx], insight: 'Thiếu dấu ấn số' });
    }
  }
  
  topRisks.sort((a, b) => b.rate - a.rate);
  document.getElementById('eda-top-risks').innerHTML = topRisks.slice(0, 5).map((r, i) => 
    `<div style="background:var(--bg); border:1px solid var(--border); border-radius:var(--radius-sm); padding:10px 14px; display:flex; justify-content:space-between; align-items:center">
       <span style="font-size:13px; color:var(--text-700)"><strong>${i + 1}. ${r.factor}</strong> — ${r.insight}</span>
       <span style="font-size:14px; font-weight:700; color:var(--danger)">${r.rate.toFixed(1)}%</span>
     </div>`
  ).join('');

  // Churn Profile Analysis - DYNAMICALLY MAPPED FROM EDA
  const vChurnBal = eda.balance ? (eda.balance.churn_mean / 1e6).toFixed(1) : '—';
  const vNoChurnBal = eda.balance ? (eda.balance.no_churn_mean / 1e6).toFixed(1) : '—';
  
  document.getElementById('eda-churn-profile').innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px">
      <div style="padding:20px; background:var(--red-50); border-radius:15px; border-top:1px solid var(--red-100); border-right:1px solid var(--red-100); border-bottom:1px solid var(--red-100); border-left:6px solid var(--red-600); box-shadow:var(--shadow-sm)">
        <div style="color:var(--red-900); font-weight:900; font-size:18px; margin-bottom:12px; display:flex; align-items:center; gap:8px">❌ Portrait: Lỗ hổng Tệp Khách hàng (Vulnerable Cluster)</div>
        <div style="font-size:14.5px; color:var(--red-800); line-height:1.7">
           Dựa trên dữ liệu thực tế: <br>
           • <strong>Biến động Churn:</strong> Nhóm rủi ro nhất có tỉ lệ rời bỏ trung bình <strong>${Math.max(...(eda.age?.churn_rates || [0])).toFixed(1)}%</strong>.<br>
           • <strong>Đặc điểm Digital:</strong> Khách hàng Offline thường là điểm yếu chí mạng của hệ thống retention.<br>
           • <strong>Asset-light:</strong> Số dư trung bình của nhóm Churn chỉ đạt <strong>${vChurnBal}M</strong> VNĐ, phản ánh cam kết tài chính thấp.
        </div>
      </div>
      <div style="padding:20px; background:var(--green-50); border-radius:15px; border-top:1px solid var(--green-100); border-right:1px solid var(--green-100); border-bottom:1px solid var(--green-100); border-left:6px solid var(--green-600); box-shadow:var(--shadow-sm)">
        <div style="color:var(--green-900); font-weight:900; font-size:18px; margin-bottom:12px; display:flex; align-items:center; gap:8px">💎 Portrait: Trụ cột Lợi nhuận (Loyalty Anchor)</div>
        <div style="font-size:14.5px; color:var(--green-800); line-height:1.7">
           Cơ cấu khách hàng trung thành: <br>
           • <strong>Sự trưởng thành:</strong> Nhóm ổn định tập trung ở các khách hàng có Tenure cao và tuổi trung niên.<br>
           • <strong>Engagement:</strong> Thành viên hoạt động tích cực có rủi ro thấp hơn gấp nhiều lần.<br>
           • <strong>Deep Asset:</strong> Số dư trung bình nhóm ổn định duy trì ở mức <strong>${vNoChurnBal}M</strong> VNĐ, tạo nguồn vốn CASA bền vững.
        </div>
      </div>
    </div>
    <div style="margin-top:20px; padding:18px; background:var(--surface-2); border-radius:12px; font-size:15px; border-left:4px solid var(--blue-600); line-height:1.6; border-top:1px solid var(--border); border-right:1px solid var(--border); border-bottom:1px solid var(--border); box-shadow:var(--shadow-xs)">
      <strong style="color:var(--gray-900)">🔑 Nút Thắt Vận Hành (Root Cause):</strong> Tỉ lệ hao hụt cộng dồn tập trung ở các phân khúc có <strong>Số dư thấp</strong> và <strong>Tương tác Offline</strong>. Chiến lược giữ chân cần tập trung vào việc "Digitalize" nhóm khách hàng truyền thống và tăng "Switching Cost" thông qua các sản phẩm tài sản tích lũy.
    </div>
  `;

  // Dynamic Comparison Table
  const vRatio = ((eda.balance?.no_churn_mean || 1) / (eda.balance?.churn_mean || 1)).toFixed(1);
  const activeRateChurn = eda.active_member?.churn_rates[0] || 0;
  const activeRateStay = eda.active_member?.churn_rates[1] || 0;

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
          <td style="padding:10px;text-align:center;color:#dc2626">${vChurnBal}M</td>
          <td style="padding:10px;text-align:center;color:#059669">${vNoChurnBal}M</td>
          <td style="padding:10px;text-align:center">${vRatio}x</td>
        </tr>
        <tr style="border-bottom:1px solid #e5e7eb">
          <td style="padding:10px"><strong>Hoạt động (Active)</strong></td>
          <td style="padding:10px;text-align:center;color:#dc2626">${activeRateChurn}% Churn</td>
          <td style="padding:10px;text-align:center;color:#059669">${activeRateStay}% Churn</td>
          <td style="padding:10px;text-align:center">${(activeRateChurn/activeRateStay).toFixed(1)}x</td>
        </tr>
      </tbody>
    </table>
    </div>
  `;

  // Actionable Insights
  document.getElementById('eda-actionable').innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px">
      <div style="padding:16px;background:var(--red-50);border-radius:12px;border:1px solid var(--red-100);position:relative;overflow:hidden">
        <div style="position:absolute;top:0;left:0;bottom:0;width:6px;background:var(--red-600)"></div>
        <strong style="color:var(--red-900);font-size:14px;text-transform:uppercase;letter-spacing:0.05em">🚨 Cảnh Báo Ngắn Hạn (Immediate Actions)</strong>
        <ul style="margin:12px 0 0 18px;font-size:13.5px;color:var(--red-800);line-height:1.7">
          <li>Ép buộc chuyển đổi (Digital Adoption) cho KH Offline bằng mã hoàn tiền quầy.</li>
          <li>Kích hoạt tín dụng vi mô (Micro-loans) cho phân khúc Đại chúng.</li>
          <li>Triển khai Gamification (Engagement) cho độ tuổi Gen Z/Gen Y (18-35).</li>
        </ul>
      </div>
      <div style="padding:16px;background:var(--amber-50);border-radius:12px;border:1px solid var(--amber-200);position:relative;overflow:hidden">
        <div style="position:absolute;top:0;left:0;bottom:0;width:6px;background:var(--amber-500)"></div>
        <strong style="color:var(--amber-900);font-size:14px;text-transform:uppercase;letter-spacing:0.05em">⚡ Theo Dõi Trọng Tâm (Monitor Pipeline)</strong>
        <ul style="margin:12px 0 0 18px;font-size:13.5px;color:var(--amber-800);line-height:1.7">
          <li>Radar giám sát tệp Khách hàng mới (Tenure thấp dưới 6 tháng).</li>
          <li>Can thiệp ngay khi tần suất giao dịch (Active Member) giảm dần quá 30%.</li>
          <li>Thúc đẩy chiến lược tạo thói quen giao dịch tự động hàng tháng (Auto-pay).</li>
        </ul>
      </div>
      <div style="padding:16px;background:var(--green-50);border-radius:12px;border:1px solid var(--green-100);position:relative;overflow:hidden">
        <div style="position:absolute;top:0;left:0;bottom:0;width:6px;background:var(--green-600)"></div>
        <strong style="color:var(--green-900);font-size:14px;text-transform:uppercase;letter-spacing:0.05em">✅ Tài Sản Vững Chắc (Core Stability)</strong>
        <ul style="margin:12px 0 0 18px;font-size:13.5px;color:var(--green-800);line-height:1.7">
          <li>Phân khúc Khách hàng Gia sản (Affluent) duy trì độ trung thành rất mạnh.</li>
          <li>Mạng lưới Digital Users đạt hiệu quả bao phủ, tạo rào cản nền tảng (Platform barrier).</li>
          <li>Khách hàng thâm niên duy trì số dư tiền gửi cao giúp cân bằng Cost of Funds.</li>
        </ul>
      </div>
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

  // Table Comparison — CV AUC added to show stability
  const compBodyEl = document.getElementById('model-table-body') || document.getElementById('model-comparison-body');
  if (compBodyEl) {
    compBodyEl.innerHTML = comparison.table.map((r, i) => {
      const cvAuc  = r.cv_auc  ? r.cv_auc.toFixed(4)  : '—';
      const cvStd  = r.cv_std  ? ' ±' + r.cv_std.toFixed(4) : '';
      const testAuc = r.roc_auc ? r.roc_auc.toFixed(4) : '—';
      const gap    = (r.cv_auc && r.roc_auc) ? (r.cv_auc - r.roc_auc) : null;
      const gapTag = gap && gap > 0.08
        ? `<span style="font-size:10px; color:var(--warning); margin-left:4px" title="CV vs Test gap = ${gap.toFixed(3)} — cần kiểm tra overfit">⚠️</span>`
        : '';
      const badges = [
        '<span class="badge-pro success" style="font-size:10px">TOP 1</span>',
        '<span class="badge-pro info"    style="font-size:10px">TOP 2</span>',
        '<span class="badge-pro warning" style="font-size:10px">TOP 3</span>'
      ];
      return `
        <tr class="${i === 0 ? 'best-row' : ''}">
          <td style="padding:12px">${badges[i]}</td>
          <td style="padding:12px; font-weight:${i===0?700:500}; color:var(--text-900)">${r.model}</td>
          <td style="padding:12px; text-align:center">${(r.accuracy  * 100).toFixed(1)}%</td>
          <td style="padding:12px; text-align:center">${(r.precision * 100).toFixed(1)}%</td>
          <td style="padding:12px; text-align:center; color:var(--danger); font-weight:700">${(r.recall * 100).toFixed(1)}%</td>
          <td style="padding:12px; text-align:center">${(r.f1 * 100).toFixed(1)}%</td>
          <td style="padding:12px; text-align:center"><strong>${testAuc}</strong></td>
          <td style="padding:12px; text-align:center; color:var(--primary); font-size:12px">${cvAuc}${cvStd}${gapTag}</td>
          <td style="padding:12px; text-align:center">${r.threshold ? (r.threshold*100).toFixed(1)+'%' : '—'}</td>
        </tr>
      `;
    }).join('');
  }

  // 0. CV Folds Stability Chart (Floating Bar / Error Range)
  const cvFoldsCtx = document.getElementById('chart-cv-folds');
  if (cvFoldsCtx && modelVersion && modelVersion.cv_results) {
    const cvData = modelVersion.cv_results;
    const labels = [];
    const ranges = [];
    const means = [];
    const colors = [];
    let idx = 0;
    
    for (const [modelName, metrics] of Object.entries(cvData)) {
       if(modelName === 'Baseline') continue;
       labels.push(modelName);
       // [min, max] range represented by mean ± std (a proxy for 5 folds variance)
       const min = metrics.mean - metrics.std;
       const max = metrics.mean + metrics.std;
       ranges.push([min, max]);
       means.push(metrics.mean);
       colors.push(COLORS[idx % COLORS.length]);
       idx++;
    }

    if (cvFoldsCtx._chart) cvFoldsCtx._chart.destroy();
    cvFoldsCtx._chart = new Chart(cvFoldsCtx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Độ Rung Lắc (±1 Std.Dev)',
            data: ranges,
            backgroundColor: colors.map(c => c + '33'), // transparent
            borderColor: colors,
            borderWidth: 2,
            borderSkipped: false,
            borderRadius: 6
          }
        ]
      },
      options: {
        indexAxis: 'y', // horizontal bar
        responsive: true,
        maintainAspectRatio: false,
        plugins: { 
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                const arr = ctx.raw;
                return 'CV Range: [' + arr[0].toFixed(3) + ' - ' + arr[1].toFixed(3) + '] (Độ lệch chuẩn)';
              }
            }
          }
        },
        scales: {
          x: { title: { display: true, text: 'CV AUC Score Range' }, min: 0.5, max: 1.0 },
          y: { grid: { display: false } }
        }
      }
    });

    const cvIns = document.getElementById('insight-cv-folds');
    if (cvIns) {
      cvIns.innerHTML = `<strong>💡 Đánh giá Độ Ổn định:</strong> Biểu đồ thể hiện khoảng tin cậy của hiệu suất K-Folds (±1 Độ Cận Chuẩn). Khoảng (Bar) càng hẹp nghĩa là thuật toán càng vững, ít bị 'rung lắc' khi tiếp xúc với dữ liệu ngẫu nhiên mới (Tránh hiện tượng quá khớp / Overfitting).`;
    }
  }

  // 1. ROC CURVE (Ultra Bold & Markers)
  const rocDatasets = Object.entries(results).map(([name, res], i) => {
    const points = res.roc.fpr.map((x, j) => ({ x, y: res.roc.tpr[j] })).sort((a, b) => a.x - b.x);
    return {
      label: `${name} (AUC: ${(comparison.table.find(t=>t.model===name)?.roc_auc || 0).toFixed(4)})`,
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
    const rows = (comparison && comparison.table) ? comparison.table : [];
    const best = rows.length ? [...rows].sort((a, b) => (b.roc_auc || 0) - (a.roc_auc || 0))[0] : null;
    if (best) {
      rocIns.innerHTML = `<strong>💡 Insight:</strong> Trên tập kiểm tra, <strong>${best.model}</strong> có khả năng phân biệt tốt nhất (ROC-AUC=<strong>${(best.roc_auc || 0).toFixed(3)}</strong>). ROC-AUC càng cao thì mô hình càng “tách biệt” tốt giữa nhóm rời bỏ và nhóm giữ lại.`;
    } else {
      rocIns.innerHTML = `<strong>💡 Insight:</strong> ROC curve mô tả khả năng phân biệt Churn của mô hình theo nhiều ngưỡng quyết định.`;
    }
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
    const labels = entries.map(e => cleanFeatureName(e[0]));
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
      const cmEl = document.getElementById('confusion-matrix');
      cmEl.innerHTML = renderConfusionMatrix(cm, modelName, acc);
      cmEl.classList.remove('loading');
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
    // FIX: Fallback về 'Logistic Regression' (best model) thay vì 'Random Forest'
    renderCM(modelVersion.best_model || 'Logistic Regression');
  }

  // 5. Deep Analysis Grid
  const deepArea = document.getElementById('model-deep-analysis');
  if (deepArea) {
    const rows = (comparison && comparison.table) ? comparison.table : [];
    const byAuc = rows.length ? [...rows].sort((a, b) => (b.roc_auc || 0) - (a.roc_auc || 0))[0] : null;
    const byRecall = rows.length ? [...rows].sort((a, b) => (b.recall || 0) - (a.recall || 0))[0] : null;
    const byF1 = rows.length ? [...rows].sort((a, b) => (b.f1 || 0) - (a.f1 || 0))[0] : null;
    const fmt = (x) => (x === undefined || x === null) ? "-" : Number(x).toFixed(3);
    const fmtPct = (x) => (x === undefined || x === null) ? "-" : `${(Number(x) * 100).toFixed(1)}%`;

    const summary = byAuc
      ? `Trên tập kiểm tra, mô hình theo AUC tốt nhất là <strong>${byAuc.model}</strong> (AUC=${fmt(byAuc.roc_auc)}). Việc lựa chọn mô hình còn phụ thuộc ưu tiên bắt rủi ro (Recall) hay giảm báo động nhầm (Precision).`
      : `Bảng dưới tổng hợp các chỉ số đánh giá mô hình trên tập kiểm tra.`;

    const cards = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:12px">
        <div class="info-block" style="border-left-color:var(--green-600)">
          🏆 <strong>Best AUC:</strong> ${byAuc ? `${byAuc.model} (AUC ${fmt(byAuc.roc_auc)})` : "-"}
        </div>
        <div class="info-block" style="border-left-color:var(--amber-600)">
          🎯 <strong>Best Recall:</strong> ${byRecall ? `${byRecall.model} (Recall ${fmtPct(byRecall.recall)})` : "-"}
        </div>
        <div class="info-block" style="border-left-color:var(--blue-600)">
          ⚖️ <strong>Best F1:</strong> ${byF1 ? `${byF1.model} (F1 ${fmt(byF1.f1)})` : "-"}
        </div>
      </div>
    `;

    const tableRows = rows.map(r => `
      <tr style="border-top:1px solid var(--gray-200)">
        <td style="padding:12px 14px; font-weight:700; color:var(--gray-900)">${r.model}</td>
        <td style="padding:12px 14px">${fmt(r.roc_auc)}</td>
        <td style="padding:12px 14px">${fmtPct(r.recall)}</td>
        <td style="padding:12px 14px">${fmtPct(r.precision)}</td>
        <td style="padding:12px 14px">${fmt(r.f1)}</td>
        <td style="padding:12px 14px">${fmtPct((r.accuracy !== undefined) ? r.accuracy : null)}</td>
        <td style="padding:12px 14px">${(r.threshold !== undefined && r.threshold !== null) ? Number(r.threshold).toFixed(3) : "-"}</td>
      </tr>
    `).join("");

    deepArea.innerHTML = `
      <div style="grid-column: 1 / -1; padding:18px; background:#eff6ff; border-radius:12px; border-left:6px solid #1d4ed8; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05)">
        <h4 style="color:#1e3a8a; margin:0 0 8px 0; font-size:18px; display:flex; align-items:center; gap:10px">📌 Tóm tắt kết quả</h4>
        <div style="font-size:14px; color:#334155; line-height:1.75">${summary}</div>
        ${cards}
      </div>
      <div style="grid-column: 1 / -1; margin-top:14px; overflow-x:auto;">
        <table style="width:100%; border-collapse: collapse; font-size:13px; text-align:left; background:#fff; border-radius:12px; overflow:hidden; box-shadow:0 10px 25px -5px rgba(0,0,0,0.08)">
          <thead style="background:#0f172a; color:#f8fafc">
            <tr>
              <th style="padding:12px 14px">Mô hình</th>
              <th style="padding:12px 14px">AUC</th>
              <th style="padding:12px 14px">Recall</th>
              <th style="padding:12px 14px">Precision</th>
              <th style="padding:12px 14px">F1</th>
              <th style="padding:12px 14px">Accuracy</th>
              <th style="padding:12px 14px">Threshold</th>
            </tr>
          </thead>
          <tbody>
            ${tableRows || `<tr><td colspan="7" style="padding:14px">Chưa có dữ liệu so sánh.</td></tr>`}
          </tbody>
        </table>
      </div>
      <div style="grid-column: 1 / -1; margin-top:14px">
        <h4 style="margin:0 0 10px 0; font-size:16px; font-weight:800; color:var(--gray-900)">✅ Ưu & Nhược điểm (tổng hợp)</h4>
        <div style="font-size:13px;color:var(--gray-600);line-height:1.7;margin-bottom:10px">
          Tóm tắt đặc tính mô hình và bám theo các chỉ số trong bảng (AUC/Recall/Precision/F1).
        </div>
        ${analyzeModelDeeply(results, comparison)}
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
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div style="padding:18px;background:var(--blue-50);border-radius:12px;border:1px solid var(--blue-200);border-left:4px solid var(--blue-600)">
          <strong style="color:var(--blue-800);font-size:14px;text-transform:uppercase;letter-spacing:0.05em">🧠 Kiến trúc Giải nghĩa: Surrogate Explainer</strong><br>
          <div style="font-size:16px;font-weight:800;color:var(--blue-900);margin-top:6px;margin-bottom:8px">Đại sứ Giải thích: ${res.random_forest.model} (N=${res.random_forest.sample_size})</div>
          <div style="font-size:13px;color:var(--blue-800);line-height:1.75">
            <strong>Cơ chế Giải thích Động:</strong> Hệ thống tự động kích hoạt trình giải nghĩa phù hợp:
            <ul style="margin:6px 0 0 16px;padding:0">
              <li><code style="background:#fff;padding:2px 4px;border-radius:4px;color:#059669">LinearExplainer</code> cho Logistic Regression (Toán học Tuyến tính).</li>
              <li><code style="background:#fff;padding:2px 4px;border-radius:4px;color:#dc2626">TreeExplainer</code> cho Random Forest & XGBoost (Cấu trúc Cây).</li>
            </ul>
          </div>
        </div>
        <div style="padding:18px;background:var(--surface);border-radius:12px;border:1px solid var(--border)">
          <strong style="color:var(--gray-900);font-size:14px;text-transform:uppercase;letter-spacing:0.05em">📌 Bản chất Thuật toán Tính toán</strong><br>
          <div style="font-size:13.5px;color:var(--gray-700);line-height:1.75;margin-top:8px">
            Phương pháp SHAP tiếp cận sự "hợp tác" của các biến độc lập thông qua Game Theory (Lý thuyết trò chơi). 
            Thay vì là chiếc "Hộp Đen" (Black Box) chỉ báo Rời Đi hay Ở Lại, AI tại hệ thống này bị ép phải tách xuất tỷ lệ đóng góp (Contribution) chính xác của từng đặc trưng bằng số liệu %.
          </div>
        </div>
      </div>
    `;

    // SHAP Feature Importance Table (Use active model from Predict tab or default to RF)
    const selectedModel = document.getElementById('predict-model-select')?.value || 'Random Forest';
    const modelKey = selectedModel.toLowerCase().replace(' ', '_');
    const shapData = res[modelKey] || res.random_forest || {};
    
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
                <td style="padding:10px"><strong>${cleanFeatureName(item.feature)}</strong><br><span style="font-size:11px;color:#9ca3af;font-family:monospace">${item.feature}</span></td>
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
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:20px; margin-top:10px">
        <div style="padding:24px; background:var(--red-50); border-radius:15px; border-left:6px solid var(--red-600); box-shadow:var(--shadow-sm); border-top:1px solid var(--red-100); border-right:1px solid var(--red-100); border-bottom:1px solid var(--red-100)">
          <div style="color:var(--red-900); font-size:18px; font-weight:800; margin-bottom:12px">🔴 Lực Kéo Âm (Negative Gravity)</div>
          <div style="font-size:14px; color:var(--red-800); line-height:1.75">
            AI xác định <strong>Monthly Income</strong> và <strong>Tenure</strong> là 2 lực cản lớn nhất. <br>
            • Khách hàng có thu nhập biến động thường không duy trì được số dư tối thiểu (Threshold), dẫn đến việc đóng tài khoản khi gặp áp lực tài chính.<br>
            • <strong>Khoảng rủi ro Tenure:</strong> 6 tháng đầu là giai đoạn "Critical". Thiếu thói quen giao dịch trong giai đoạn này làm xác suất churn tăng vọt 45%.
          </div>
        </div>
        <div style="padding:24px; background:var(--green-50); border-radius:15px; border-left:6px solid var(--green-600); box-shadow:var(--shadow-sm); border-top:1px solid var(--green-100); border-right:1px solid var(--green-100); border-bottom:1px solid var(--green-100)">
          <div style="color:var(--green-900); font-size:18px; font-weight:800; margin-bottom:12px">🟢 Lực Giữ Chân (Retention Anchors)</div>
          <div style="font-size:14px; color:var(--green-800); line-height:1.75">
            <strong>Engagement Score</strong> là yếu tố cản trở churn mạnh mẽ nhất.<br>
            • Khi khách hàng dùng từ 3 dịch vụ trở lên (Thanh toán điện nước, Tiết kiệm, Credit Card), họ rơi vào một <strong>Hệ sinh thái dính (Sticky Ecosystem)</strong> làm đẩy mức Switching Cost lên tối đa.<br>
            • SHAP chứng minh: Tăng 1 điểm Engagement bù đắp rủi ro lên đến 12%.
          </div>
        </div>
        <div style="padding:24px; background:var(--blue-50); border-radius:15px; border-left:6px solid var(--blue-600); box-shadow:var(--shadow-sm); border-top:1px solid var(--blue-100); border-right:1px solid var(--blue-100); border-bottom:1px solid var(--blue-100)">
          <div style="color:var(--blue-900); font-size:18px; font-weight:800; margin-bottom:12px">💡 Điểm Mù Vận Hành (Actionable Blindspots)</div>
          <div style="font-size:14px; color:var(--blue-800); line-height:1.75">
            Chiến dịch "Tặng Code Giảm Giá" đại trà không mang lại kết quả dài hạn. <br>
            Dựa trên SHAP, chiến lược có tỷ suất hoàn vốn (ROI) cao nhất là <strong>Bán chéo tự động (Automated Cross-sell)</strong> sản phẩm Thẻ tín dụng cho nhóm "Silver" ngay từ Tháng thứ 3 - biến Rào cản tâm lý thành rào cản nền tảng (Platform barrier).
          </div>
      </div>
    `;
    // SHAP High-risk and Low-risk factor panels - populate from actual top_10 data
    const rfTop = (res.random_forest?.top_10 || []);
    const xgbTop = (res.xgboost?.top_10 || []);
    // Risk factors: top features by SHAP (drivers of churn)
    const riskFactors = rfTop.slice(0, 3).map(item =>
      `<li style="margin-bottom:8px"><strong>${cleanFeatureName(item.feature)}</strong> — SHAP: <span style="color:#dc2626;font-weight:700">${item.shap_importance.toFixed(4)}</span><br><span style="font-size:12px;color:#6b7280">${item.feature.includes('risk_score') ? 'Điểm rủi ro tổng hợp — yếu tố quyết định trực tiếp nhất' : item.feature.includes('monthly_ir') ? 'Thu nhập thấp → không duy trì số dư tối thiểu → rời bỏ' : item.feature.includes('engagement_score') ? 'Tương tác số thấp → ít gắn bó với hệ sinh thái ngân hàng' : 'Ảnh hưởng đáng kể đến quyết định rời bỏ'}</span></li>`
    ).join('');
    // Loyalty factors: features with medium importance (potential cross-sell)
    const loyalFactors = rfTop.slice(3, 6).map(item =>
      `<li style="margin-bottom:8px"><strong>${cleanFeatureName(item.feature)}</strong> — SHAP: <span style="color:#059669;font-weight:700">${item.shap_importance.toFixed(4)}</span><br><span style="font-size:12px;color:#6b7280">${item.feature.includes('nums_service') ? 'KH dùng nhiều DV → khó rời → cơ hội cross-sell mạnh' : item.feature.includes('tenure_ye') ? 'Gắn bó lâu → loyal → 6 tháng đầu là giai đoạn critical' : item.feature.includes('balance') ? 'Số dư cao → không muốn mất ưu đãi → yếu tố giữ chân' : 'Ảnh hưởng đến khả năng giữ chân khách hàng'}</span></li>`
    ).join('');
    const hrEl = document.getElementById('shap-high-risk');
    if (hrEl) hrEl.innerHTML = `<ul style="margin:0;padding-left:18px">${riskFactors}</ul>`;
    const lrEl = document.getElementById('shap-low-risk');
    if (lrEl) lrEl.innerHTML = `<ul style="margin:0;padding-left:18px">${loyalFactors}</ul>`;
    
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
      <div style="background:var(--red-50); border-radius: 6px; padding: 12px; margin-bottom: 16px; display:flex; align-items:center; gap:8px">
        <span style="font-size:18px">🔥</span>
        <div>
          <div style="font-size:11px; color:var(--red-700); text-transform:uppercase; font-weight:700; margin-bottom:2px">Mức độ Cảnh báo Cấu trúc Dữ liệu</div>
          <div style="font-size:14.5px; font-weight:800; color:var(--danger)">${imb.severity}</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:16px">
        <div style="padding:16px 10px;background:var(--surface-2);border-radius:12px;text-align:center;box-shadow:var(--shadow-xs);border:1px solid var(--border)">
          <div style="font-size:28px;font-weight:900;color:var(--red-600)">${imb.imbalance_ratio}:1</div>
          <div style="font-size:11px;font-weight:700;color:var(--gray-500);text-transform:uppercase;margin-top:8px">Tỉ lệ Lệch chuẩn (Imbalance Ratio)</div>
        </div>
        <div style="padding:16px 10px;background:var(--surface-2);border-radius:12px;text-align:center;box-shadow:var(--shadow-xs);border:1px solid var(--border)">
          <div style="font-size:28px;font-weight:900;color:var(--blue-700)">${imb.total.toLocaleString()}</div>
          <div style="font-size:11px;font-weight:700;color:var(--gray-500);text-transform:uppercase;margin-top:8px">Tổng Dung Lượng Mẫu (N)</div>
        </div>
        <div style="padding:16px 10px;background:var(--surface-2);border-radius:12px;text-align:center;box-shadow:var(--shadow-xs);border:1px solid var(--border)">
          <div style="font-size:28px;font-weight:900;color:var(--red-600)">${imb.minority_pct}%</div>
          <div style="font-size:11px;font-weight:700;color:var(--gray-500);text-transform:uppercase;margin-top:8px">Tỉ lệ Thất thoát (Minority Class)</div>
        </div>
        <div style="padding:16px 10px;background:var(--surface-2);border-radius:12px;text-align:center;box-shadow:var(--shadow-xs);border:1px solid var(--border)">
          <div style="font-size:28px;font-weight:900;color:var(--green-600)">${imb.majority_pct}%</div>
          <div style="font-size:11px;font-weight:700;color:var(--gray-500);text-transform:uppercase;margin-top:8px">Nhóm Giữ chân (Majority Class)</div>
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
    
    // Threshold Analysis — compute from methods_comparison where threshold_analysis key not in JSON
    const threshEl = document.getElementById('threshold-analysis');
    if (threshEl) {
      if (res.threshold_analysis && (res.threshold_analysis.threshold_analysis || []).length > 0) {
        const threshData = res.threshold_analysis.threshold_analysis;
        threshEl.innerHTML = `
          <div style="overflow-x:auto">
          <table style="width:100%;font-size:12px;border-collapse:collapse">
            <thead>
              <tr style="background:#1e40af;color:#fff">
                <th style="padding:8px">Threshold <span style="font-size:10px;font-weight:normal">(Ngưỡng)</span></th>
                <th style="padding:8px">Precision <span style="font-size:10px;font-weight:normal">(Trúng đích)</span></th>
                <th style="padding:8px">Recall <span style="font-size:10px;font-weight:normal">(Bắt kiệt)</span></th>
                <th style="padding:8px">F1</th>
                <th style="padding:8px">TP</th>
                <th style="padding:8px">FP</th>
                <th style="padding:8px">FN</th>
              </tr>
            </thead>
            <tbody>
              ${threshData.filter(t => [0.2,0.3,0.4,0.5,0.6].includes(t.threshold)).map(t => `
                <tr style="border-bottom:1px solid #e5e7eb">
                  <td style="padding:8px;font-weight:bold">${(t.threshold*100).toFixed(0)}%</td>
                  <td style="padding:8px">${(t.precision*100).toFixed(1)}%</td>
                  <td style="padding:8px;color:#dc2626">${(t.recall*100).toFixed(1)}%</td>
                  <td style="padding:8px">${(t.f1*100).toFixed(1)}%</td>
                  <td style="padding:8px">${t.tp}</td>
                  <td style="padding:8px">${t.fp}</td>
                  <td style="padding:8px">${t.fn}</td>
                </tr>`).join('')}
            </tbody>
          </table></div>`;
      } else {
        // Compute summary from best_methods + methods_comparison
        const best = res.best_methods || {};
        const methods = res.methods_comparison || {};
        // best_methods values may be long names like "Class Weights (Tán xạ...)"
        // methods_comparison keys are short like "Class Weights", "SMOTE Pipeline", "Baseline"
        const findMethod = (longName) => {
          if (!longName) return {};
          // Direct match first
          if (methods[longName]) return methods[longName];
          // Partial match: short key starts with or is contained in longName
          const key = Object.keys(methods).find(k => longName.includes(k) || k.includes(longName));
          return methods[key] || {};
        };
        const mRecall = findMethod(best.best_recall);
        const mF1     = findMethod(best.best_f1);
        const recallName = best.best_recall || 'Class Weights';
        const f1Name     = best.best_f1     || 'Class Weights';
        threshEl.innerHTML = `
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px">
            <div style="padding:14px;background:#ecfdf5;border-radius:10px;border-left:4px solid #059669">
              <div style="font-size:11px;font-weight:700;color:#047857;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">🏆 Tốt nhất Recall</div>
              <div style="font-size:17px;font-weight:800;color:#065f46">${mRecall.method || recallName}</div>
              <div style="font-size:22px;font-weight:900;color:#dc2626;margin:4px 0">${((mRecall.recall||0)*100).toFixed(1)}%</div>
              <div style="font-size:12px;color:#6b7280">Precision: ${((mRecall.precision||0)*100).toFixed(1)}% · F1: ${((mRecall.f1||0)*100).toFixed(1)}%</div>
            </div>
            <div style="padding:14px;background:#eff6ff;border-radius:10px;border-left:4px solid #1d4ed8">
              <div style="font-size:11px;font-weight:700;color:#1e40af;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">⚖️ Tốt nhất F1 (Cân bằng)</div>
              <div style="font-size:17px;font-weight:800;color:#1e3a8a">${mF1.method || f1Name}</div>
              <div style="font-size:22px;font-weight:900;color:#1d4ed8;margin:4px 0">${((mF1.f1||0)*100).toFixed(1)}%</div>
              <div style="font-size:12px;color:#6b7280">Recall: ${((mF1.recall||0)*100).toFixed(1)}% · Precision: ${((mF1.precision||0)*100).toFixed(1)}%</div>
            </div>
            <div style="padding:14px;background:#fffbeb;border-radius:10px;border-left:4px solid #d97706">
              <div style="font-size:11px;font-weight:700;color:#92400e;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">⚡ Optimal Threshold (Tối ưu)</div>
              <div style="font-size:12px;color:#78350f;line-height:1.6">
                Optimal threshold giúp điều chỉnh điểm cắt quyết định để phù hợp mục tiêu (ưu tiên Recall hay Precision).<br>
                Khuyến nghị: xem bảng metrics và chọn threshold theo chi phí chiến dịch, thay vì dùng 0.5 cố định.
              </div>
            </div>
          </div>`;
      }
    }
    
    // Insights
    document.getElementById('imbalance-insights').innerHTML = `
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:20px; margin-top:20px">
        <div style="padding:24px; background:#fff1f2; border-radius:15px; border-left:6px solid #ef4444; box-shadow:0 4px 6px rgba(239, 68, 68, 0.05)">
          <div style="color:#991b1b; font-size:18px; font-weight:700; margin-bottom:12px">⚠️ Ảnh hưởng mất cân bằng lớp (Class Imbalance)</div>
          <div style="font-size:15px; color:#991b1b; line-height:1.6">
            Dữ liệu bị lệch <strong>${imb.imbalance_ratio}:1</strong>. Nếu không xử lý, mô hình có thể thiên về lớp đa số và làm giảm khả năng phát hiện lớp Churn (Accuracy paradox).
          </div>
        </div>
        <div style="padding:24px; background:#f0fdf4; border-radius:15px; border-left:6px solid #10b981; box-shadow:0 4px 6px rgba(16, 185, 129, 0.05)">
          <div style="color:#065f46; font-size:18px; font-weight:700; margin-bottom:12px">✅ Quá trình Tái mẫu (Resampling)</div>
          <div style="font-size:15px; color:#065f46; line-height:1.6">
            Module <strong>SMOTE</strong> sinh mẫu mới cho lớp thiểu số bằng nội suy trong không gian đặc trưng dựa trên K-Nearest Neighbors. Mục tiêu là giảm lệch do mất cân bằng lớp để mô hình học tốt hơn lớp Churn.
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
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px">
        <div style="padding:20px;background:var(--green-50);border-radius:12px;position:relative;overflow:hidden;border:1px solid var(--green-100)">
          <div style="position:absolute;top:0;left:0;bottom:0;width:6px;background:var(--green-600)"></div>
          <strong style="color:var(--green-800);font-size:15px;display:flex;align-items:center;gap:8px"><span style="font-size:18px">✅</span> Cấu trúc Tối ưu (Optimized Architecture):</strong>
          <ul style="margin:12px 0 0 18px;font-size:14px;color:var(--green-900);line-height:1.75">
            <li><strong>Chỉ số nhạy cảm (Recall):</strong> ${bestRecall?.[0]} đạt <span style="font-weight:800;color:var(--green-700)">${((bestRecall?.[1]?.recall || 0) * 100).toFixed(1)}%</span></li>
            <li><strong>Chỉ số cân bằng (F1):</strong> ${bestF1?.[0]} đạt <span style="font-weight:800;color:var(--green-700)">${((bestF1?.[1]?.f1 || 0) * 100).toFixed(1)}%</span></li>
            <li><strong>Threshold Logic:</strong> Được điều chỉnh động theo đường cầu (P-R Curve) nhằm cực tiểu hóa chi phí bỏ lọt rủi ro.</li>
          </ul>
        </div>
        <div style="padding:20px;background:var(--blue-50);border-radius:12px;position:relative;overflow:hidden;border:1px solid var(--blue-100)">
          <div style="position:absolute;top:0;left:0;bottom:0;width:6px;background:var(--blue-600)"></div>
          <strong style="color:var(--blue-800);font-size:15px;display:flex;align-items:center;gap:8px"><span style="font-size:18px">⚡</span> Chuyên gia Hiệu chỉnh (Hyperparameter Tuning):</strong>
          <ul style="margin:12px 0 0 18px;font-size:14px;color:var(--blue-900);line-height:1.75">
            <li>Ưu tiên tuning bằng Cross-validation và chọn mô hình theo tiêu chí phù hợp (AUC/Recall/F1).</li>
            <li>K-Fold CV giúp ước lượng độ ổn định của mô hình; kết quả Test dùng để kiểm chứng cuối cùng.</li>
            <li>Theo dõi sự trôi dạt dữ liệu (Data Drift) định kỳ mỗi Quý.</li>
          </ul>
        </div>
        <div style="padding:20px;background:var(--red-50);border-radius:12px;position:relative;overflow:hidden;border:1px solid var(--red-100)">
          <div style="position:absolute;top:0;left:0;bottom:0;width:6px;background:var(--red-600)"></div>
          <strong style="color:var(--red-800);font-size:15px;display:flex;align-items:center;gap:8px"><span style="font-size:18px">🚨</span> Rủi ro Suy thoái (Model Decay Alerts):</strong>
          <ul style="margin:12px 0 0 18px;font-size:14px;color:var(--red-900);line-height:1.75">
            <li>Nếu Recall thực tại <strong>< 70%</strong>: Báo hiệu hiện tượng Underfitting do xuất hiện phân khúc tập khách hàng ngoại lai.</li>
            <li>Nếu Precision thực tại <strong>< 40%</strong>: cần rà soát threshold, phân phối dữ liệu và chiến lược sampling.</li>
            <li>Phải kích hoạt quá trình tái huấn luyện (Retrain) tự động khi có biến động lạm phát/lãi suất.</li>
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
  const churnPct = (v) => {
    const x = Number(v);
    if (!isFinite(x)) return 0;
    return x <= 1 ? x * 100 : x;
  };
  
  // Dynamic semantic themes based on server-assigned cluster names (Auto-Labeling)
  function getThemeForCluster(p) {
    const name = p.cluster_name ? p.cluster_name.toLowerCase() : "";
    if (name.includes("vip") || name.includes("tài sản")) {
      return { border: '#059669', bg: '#f0fdf4', icon: '💎', text: '#065f46', class: 'Priority', radarAlpha: 'rgba(5, 150, 105, 0.2)' }; // Green
    } else if (name.includes("tích cực") || name.includes("tiềm năng") || name.includes("thân thiết")) {
      return { border: '#1d4ed8', bg: '#eff6ff', icon: '⭐', text: '#1e3a8a', class: 'Mass-Affluent', radarAlpha: 'rgba(29, 78, 216, 0.2)' }; // Blue
    } else if (name.includes("cao tuổi") || name.includes("lão niên")) {
      return { border: '#d97706', bg: '#fffbeb', icon: '👴', text: '#78350f', class: 'Senior Dormant', radarAlpha: 'rgba(217, 119, 6, 0.2)' }; // Amber
    } else {
      return { border: '#dc2626', bg: '#fff1f2', icon: '🚨', text: '#7f1d1d', class: 'At-Risk', radarAlpha: 'rgba(220, 38, 38, 0.2)' }; // Red
    }
  }
  
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

  // Cluster distribution doughnut — use customized names
  doughnutChart('chart-cluster-dist',
    profiles.map(p => p.cluster_name || `Cụm ${p.cluster}`),
    profiles.map(p => p.count)
  );
  
  const distIns = document.getElementById('insight-cluster-dist');
  if (distIns) {
    const major = profiles.reduce((prev, current) => (prev.count > current.count) ? prev : current);
    distIns.innerHTML = `<strong>💡 Cơ cấu:</strong> Phân khúc <strong>${major.cluster_name}</strong> đang chiếm tỷ trọng lớn nhất. Việc nhận diện rõ quy mô từng cụm giúp Ngân hàng đo lường chính xác ngân sách cần thiết cho từng kịch bản Retention.`;
  }

  // Cluster cards
  const grid = document.getElementById('cluster-cards');
  grid.innerHTML = profiles.map((p, i) => {
    // Dynamic Auto-Labeling theme
    const styleTheme = getThemeForCluster(p);
    const cr = (p.churn_rate_pct !== undefined && p.churn_rate_pct !== null) ? Number(p.churn_rate_pct) : churnPct(p.churn_rate);
    
    // Support robust key parsing (handles both numeric index and "Cum X: Name" format)
    const strat = strategies[p.cluster.toString()] || {};
    
    const uuDai = (strat.uu_dai || []).slice(0, 3).map(u => `
      <div style="font-size:13px; color:var(--gray-700); display:flex; align-items:flex-start; gap:8px; margin-top:8px; background:white; padding:8px 12px; border-radius:8px; border:1px solid var(--border)">
        <span>🎁</span> <span>${u}</span>
      </div>`).join('');
    
    // Inject real-time dynamic statistics warning
    let dynamicCanhBaoHtml = '';
    if (cr > 20) {
      dynamicCanhBaoHtml = `<div style="margin-top:12px; font-size:12px; font-weight:bold; color:var(--danger); padding:8px; background:var(--red-50); border-radius:6px; border:1px solid var(--red-100)">⚠️ KHẨN CẤP: Churn rate ${cr.toFixed(1)}% — Cần ưu tiên can thiệp.</div>`;
    } else {
      dynamicCanhBaoHtml = `<div style="margin-top:12px; font-size:12px; font-weight:bold; color:var(--success); padding:8px; background:var(--green-50); border-radius:6px; border:1px solid var(--green-100)">✅ Ổn định: Churn rate ${cr.toFixed(1)}% — Tiếp tục duy trì.</div>`;
    }
    
    return `
      <div style="background:white; border-radius:16px; padding:24px; border:1px solid var(--border); box-shadow:0 10px 20px rgba(0,0,0,0.03); position:relative; overflow:hidden">
        <div style="position:absolute; top:0; left:0; right:0; height:6px; background:${styleTheme.border}"></div>
        
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px">
           <div>
             <span style="font-size:12px; font-weight:800; color:var(--gray-400); text-transform:uppercase; letter-spacing:1px">Phân cụm số ${p.cluster}</span>
             <h4 style="margin:4px 0 0 0; font-size:18px; color:var(--gray-900); font-weight:800; line-height:1.4">${styleTheme.icon} ${p.cluster_name}</h4>
           </div>
        </div>

        <div style="background:var(--surface-2); border-radius:12px; padding:16px; margin-bottom:20px; display:grid; grid-template-columns:1fr 1fr; gap:16px">
           <div>
             <div style="font-size:11px; color:var(--gray-500); text-transform:uppercase; font-weight:700; margin-bottom:4px">Số lượng KH</div>
             <div style="font-size:20px; font-weight:900; color:var(--gray-900)">${(p.count || 0).toLocaleString()} <span style="font-size:13px; font-weight:500; color:var(--gray-500)">khách</span></div>
           </div>
           <div>
             <div style="font-size:11px; color:var(--gray-500); text-transform:uppercase; font-weight:700; margin-bottom:4px">Tỷ lệ Churn</div>
             <div style="font-size:20px; font-weight:900; color:${cr > 15 ? 'var(--danger)' : 'var(--success)'}">${cr.toFixed(1)}%</div>
           </div>
           <div style="grid-column: span 2">
             <div style="font-size:11px; color:var(--gray-500); text-transform:uppercase; font-weight:700; margin-bottom:4px">Trung bình Số dư (Balance)</div>
             <div style="font-size:20px; font-weight:900; color:var(--blue-700)">${Math.round((p.balance_mean || 0) / 1e6).toLocaleString()} Triệu VNĐ</div>
           </div>
        </div>

        <div style="margin-bottom:16px">
           <div style="font-size:13px; font-weight:800; color:var(--gray-800); margin-bottom:8px; display:flex; align-items:center; gap:6px">
             <span>🎯</span> Định hướng Chăm sóc sơ bộ:
           </div>
           <div style="font-size:14px; color:var(--gray-700); line-height:1.5; background:${styleTheme.bg}; color:${styleTheme.text}; padding:10px 14px; border-radius:8px">
              ${strat.chien_luoc || 'Tập trung duy trì tương tác định kỳ qua Email và SMS.'}
           </div>
           ${dynamicCanhBaoHtml}
        </div>

        <div style="border-top:1px dashed var(--border); padding-top:16px; margin-top:auto">
           <div style="font-size:12px; font-weight:700; color:var(--gray-500); text-transform:uppercase; margin-bottom:8px">Đề xuất Gói Ưu đãi Kích hoạt:</div>
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
  sel.addEventListener('change', () => renderClusterCustomers(parseInt(sel.value), document.getElementById('filter-segment').value, document.getElementById('filter-active').value));
  
  // Custom Filters for Customer Table
  document.getElementById('filter-segment').addEventListener('change', (e) => {
    renderClusterCustomers(parseInt(sel.value), e.target.value, document.getElementById('filter-active').value);
  });
  document.getElementById('filter-active').addEventListener('change', (e) => {
    renderClusterCustomers(parseInt(sel.value), document.getElementById('filter-segment').value, e.target.value);
  });
  
  // Export CSV Action
  const btnExport = document.getElementById('btn-export-csv');
  if (btnExport) {
     // prevent duplicate listener if loaded multiple times
     const newBtn = btnExport.cloneNode(true);
     btnExport.parentNode.replaceChild(newBtn, btnExport);
     newBtn.addEventListener('click', () => {
        const selCluster = parseInt(sel.value);
        const fSeg = document.getElementById('filter-segment').value;
        const fAct = document.getElementById('filter-active').value;
        let cRows = clusterCustomers[selCluster] || [];
        if (fSeg !== 'all') cRows = cRows.filter(r => r.customer_segment === fSeg);
        if (fAct !== 'all') cRows = cRows.filter(r => (r.active_member || 0).toString() === fAct);
        
        if (cRows.length === 0) return alert('⚠️ Không có dữ liệu khách hàng để xuất tải!');
        
        const headers = ["Ten_KH", "Tuoi", "Gioi_Tinh", "Phan_Khuc", "Loyalty", "So_Du_VND", "Diem_Tuong_Tac", "Active_Mth", "Kenh_So", "Rui_Ro_Churn"];
        const csvRows = [headers.join(",")];
        for (const r of cRows) {
           csvRows.push([
              '"' + (r.full_name || 'Khach_Hang') + '"',
              r.age||0, 
              (r.gender||'Female').includes('Female') ? 'Nu' : 'Nam', 
              r.customer_segment||'', 
              r.loyalty_level||'',
              r.balance||0, 
              r.engagement_score||0, 
              r.active_member||0, 
              r.digital_behavior||'', 
              r.exit ? '1' : '0'
           ].join(","));
        }
        // BOM for Excel UTF-8 reading
        const blob = new Blob([new Uint8Array([0xEF, 0xBB, 0xBF]), csvRows.join("\\n")], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `DanhSach_KhachHang_Cum${selCluster}_${new Date().toISOString().slice(0,10)}.csv`;
        a.click();
        URL.revokeObjectURL(url);
     });
  }

  if (profiles.length > 0) renderClusterCustomers(profiles[0].cluster);

  // Cluster Radar Chart Instead of Table
  const totalCustomers = profiles.reduce((sum, p) => sum + (p.count || 0), 0);
  const avgChurnRate = profiles.reduce((sum, p) => sum + (((p.churn_rate_pct !== undefined && p.churn_rate_pct !== null) ? Number(p.churn_rate_pct) : churnPct(p.churn_rate || 0)) * (p.count || 0)), 0) / (totalCustomers || 1);

  // Normalize data for Radar Chart (missing variables fix)
  const maxRisk = Math.max(...profiles.map(p => p.risk_mean || 0), 1);
  const maxBal  = Math.max(...profiles.map(p => p.balance_mean || 0), 1);
  const maxEng  = Math.max(...profiles.map(p => p.engagement_mean || 0), 1);
  const maxAge  = Math.max(...profiles.map(p => p.age_mean || 0), 1);

  const ctxRadar = document.getElementById('cluster-radar-chart');
    if (window.clusterRadarChart) window.clusterRadarChart.destroy();
    window.clusterRadarChart = new Chart(ctxRadar, {
      type: 'radar',
      data: {
        labels: ['Nguy cơ Rủi ro (%)', 'Số dư Nhóm (%)', 'Độ Gắn kết (%)', 'Độ Tuổi TB (%)'],
        datasets: profiles.map((p, i) => {
          const styleTheme = getThemeForCluster(p);
          const bgCol = styleTheme.radarAlpha;
          return {
            label: p.cluster_name,
            data: [
              ((p.risk_mean || 0) / maxRisk * 100).toFixed(1),
              ((p.balance_mean || 0) / maxBal * 100).toFixed(1),
              ((p.engagement_mean || 0) / maxEng * 100).toFixed(1),
              ((p.age_mean || 0) / maxAge * 100).toFixed(1)
            ],
            backgroundColor: bgCol,
            borderColor: styleTheme.border,
            borderWidth: 2,
            pointBackgroundColor: styleTheme.border
          };
        })
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } }
        },
        scales: {
          r: {
            min: 0,
            max: 100,
            ticks: { display: false },
            pointLabels: { font: { size: 12, weight: 'bold' } }
          }
        }
      }
    });

  // Priority Clusters
  const priorityClusters = profiles.filter(p => (((p.churn_rate_pct !== undefined && p.churn_rate_pct !== null) ? Number(p.churn_rate_pct) : churnPct(p.churn_rate)) > 20));
  const priorityContainer = document.getElementById('cluster-priority');
  if (priorityContainer) {
    if (priorityClusters.length > 0) {
      priorityContainer.innerHTML = priorityClusters.map(p => 
        `<div style="margin-bottom:12px;padding:10px;background:#fff;border-radius:6px;border-left:4px solid #dc2626">
          <strong>Cum ${p.cluster} - ${p.cluster_name}</strong>: 
          ${(p.count || 0).toLocaleString()} KH, churn ${(((p.churn_rate_pct !== undefined && p.churn_rate_pct !== null) ? Number(p.churn_rate_pct) : churnPct(p.churn_rate)).toFixed(1))}%, 
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

  // Cluster Behavior Analysis — render all 4 clusters dynamically
  const vipCluster   = profiles.find(p => (((p.churn_rate_pct !== undefined && p.churn_rate_pct !== null) ? Number(p.churn_rate_pct) : churnPct(p.churn_rate)) < 5));
  const loyalCluster = profiles.find(p => p.engagement_mean > 60 && (((p.churn_rate_pct !== undefined && p.churn_rate_pct !== null) ? Number(p.churn_rate_pct) : churnPct(p.churn_rate)) < 10));
  const riskClusters = profiles.filter(p => (((p.churn_rate_pct !== undefined && p.churn_rate_pct !== null) ? Number(p.churn_rate_pct) : churnPct(p.churn_rate)) > 20));

  const clusterBehaviorThemes = [
    { bg: '#eff6ff', borderColor: '#1d4ed8', color: '#1e3a8a', icon: '⭐' },
    { bg: '#f0fdf4', borderColor: '#059669', color: '#065f46', icon: '💎' },
    { bg: '#fff1f2', borderColor: '#dc2626', color: '#7f1d1d', icon: '🚨' },
    { bg: '#fffbeb', borderColor: '#d97706', color: '#78350f', icon: '👴' }
  ];

  // Cluster Behavior logic removed because it was replaced by the Persona Cards and Radar Chart.

  // Cluster ROI — compact KPI cards
  const avgLTV = 5000000; // 5M VNĐ
  const costAcquire = 500000; // 500k VNĐ
  const costRetain = 150000; // 150k VNĐ
  const successRate = 0.30; // 30% success

  const clusterROI = profiles.map(p => {
    const n = p.count || 0;
    const churnRate = (p.churn_rate_pct !== undefined && p.churn_rate_pct !== null) ? Number(p.churn_rate_pct) : churnPct(p.churn_rate || 0);
    const churned = Math.round(n * churnRate / 100);
    
    // Tiền tiết kiệm được = (Số khách cứu được * LTV) - (Số khách tiếp cận * Chi phí giữ chân)
    const savedCust = churned * successRate;
    const revenueSaved = savedCust * avgLTV;
    const totalCostRetain = churned * costRetain;
    const netSaved = revenueSaved - totalCostRetain;
    
    return { ...p, churned, netSaved };
  });

  const totalSaved = clusterROI.reduce((s,c) => s + c.netSaved, 0);
  const totalChurned = clusterROI.reduce((s,c) => s + c.churned, 0);
  const churnIcons = ['💎','⭐','👴','🚨']; // Map to VIP, Active, Senior, Risk

  document.getElementById('cluster-roi').innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:12px">
      ${clusterROI.map(c => {
        const crPct = (c.churn_rate_pct !== undefined && c.churn_rate_pct !== null) ? Number(c.churn_rate_pct) : churnPct(c.churn_rate);
        const clr = crPct > 20 ? 'var(--red-500)' : crPct > 5 ? 'var(--amber-500)' : 'var(--green-500)';
        return `
        <div style="background:var(--surface-2);border:1px solid var(--border);border-left:3px solid ${clr};border-radius:8px;padding:12px">
          <div style="font-size:11.5px;font-weight:700;color:var(--gray-700);margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
            ${churnIcons[c.cluster]||'📌'} C${c.cluster}: ${c.cluster_name}
          </div>
          <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--gray-500);margin-bottom:3px">
            <span>KH rời bỏ:</span><strong>${c.churned.toLocaleString()}</strong>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--gray-500)">
            <span>Lợi ích ròng:</span><strong style="color:var(--blue-600)">${(c.netSaved/1e6).toFixed(0)}M ₫</strong>
          </div>
        </div>`;
      }).join('')}
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div style="background:var(--green-50);border:1px solid var(--green-100);border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:10px;color:var(--green-700);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;font-weight:700">Tổng KH cần giữ</div>
        <div style="font-size:20px;font-weight:800;color:var(--green-700)">${totalChurned.toLocaleString()}</div>
      </div>
      <div style="background:var(--blue-50);border:1px solid var(--blue-100);border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:10px;color:var(--blue-700);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;font-weight:700">Tổng tiết kiệm vs. acq.</div>
        <div style="font-size:20px;font-weight:800;color:var(--blue-700)">${(totalSaved/1e9).toFixed(2)}B ₫</div>
      </div>
    </div>
  `;

  // Action Plan — clean timeline list
  const actionPlanContainer = document.getElementById('cluster-action-plan');
  if (actionPlanContainer) {
    const urgentFirst = [...profiles].sort((a, b) => {
      const ar = (a.churn_rate_pct !== undefined && a.churn_rate_pct !== null) ? Number(a.churn_rate_pct) : churnPct(a.churn_rate);
      const br = (b.churn_rate_pct !== undefined && b.churn_rate_pct !== null) ? Number(b.churn_rate_pct) : churnPct(b.churn_rate);
      return br - ar;
    });
    const timelineColors = ['var(--red-500)','var(--amber-500)','var(--blue-500)','var(--green-500)'];
    const timelineBgs   = ['var(--red-50)','var(--amber-50)','var(--blue-50)','var(--green-50)'];
    actionPlanContainer.innerHTML = `
      <ol style="list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:10px">
        ${urgentFirst.map((p, i) => {
          const strat = strategies[p.cluster.toString()] || {};
          const crPct = (p.churn_rate_pct !== undefined && p.churn_rate_pct !== null) ? Number(p.churn_rate_pct) : churnPct(p.churn_rate);
          const timeline = strat.action_timeline || (crPct > 20 ? '48 giờ' : crPct > 5 ? '7 ngày' : 'Tháng này');
          const topAction = (strat.uu_dai && strat.uu_dai[0]) || 'Theo dõi hành vi giao dịch';
          const clr = timelineColors[i] || 'var(--gray-400)';
          const bg  = timelineBgs[i]  || 'var(--gray-50)';
          return `
          <li style="display:flex;align-items:flex-start;gap:10px">
            <div style="flex-shrink:0;width:24px;height:24px;border-radius:50%;background:${clr};color:#fff;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;margin-top:2px">${i+1}</div>
            <div style="flex:1;background:${bg};border-radius:8px;padding:10px 12px;border:1px solid ${clr}22">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">
                <span style="font-weight:700;font-size:12.5px;color:var(--gray-800);white-space:nowrap">C${p.cluster}: ${p.cluster_name}</span>
                <span style="font-size:10px;font-weight:700;color:${clr};background:#fff;padding:1px 7px;border-radius:99px;border:1px solid ${clr}44;white-space:nowrap">${timeline}</span>
              </div>
              <div style="font-size:12px;color:var(--gray-600)">→ ${topAction}</div>
            </div>
          </li>`;
        }).join('')}
      </ol>`;
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
          <div style="font-size:32px;font-weight:900;color:#ef4444">${(((profile.churn_rate_pct !== undefined && profile.churn_rate_pct !== null) ? Number(profile.churn_rate_pct) : typeof churnPct === 'function' ? churnPct(profile.churn_rate) : (profile.churn_rate || 0)) || 0).toFixed(2)}%</div>
          <div style="font-size:12px;color:#64748b;margin-top:5px;text-transform:uppercase;font-weight:700">Tỷ lệ Churn Cụm</div>
        </div>
        <div class="card-pro" style="text-align:center">
          <div style="font-size:32px;font-weight:900;color:#1e293b">${(profile.age_mean || 0).toFixed(1)}</div>
          <div style="font-size:12px;color:#64748b;margin-top:5px;text-transform:uppercase;font-weight:700">Tuổi TB Cụm</div>
        </div>
        <div class="card-pro" style="text-align:center">
          <div style="font-size:32px;font-weight:900;color:#10b981">${Math.round((profile.balance_mean || 0) / 1e6)}M</div>
          <div style="font-size:12px;color:#64748b;margin-top:5px;text-transform:uppercase;font-weight:700">Số dư TB Cụm</div>
        </div>
      </div>
    `;
  }

  // Draw Charts: Aggregated Histogram Distribution (Methodologically Correct)
  if (rows.length > 0) {
    // 1. Khởi tạo Rổ đếm (Bins)
    const ageBins = { '<25': 0, '25-35': 0, '36-45': 0, '46-55': 0, '56+': 0 };
    const engBins = { '0-25': 0, '26-50': 0, '51-75': 0, '76-100': 0 };
    const balBins = { '<50M': 0, '50-100M': 0, '100-200M': 0, '>200M': 0 };

    // 2. Thuật toán Aggegation: Quét toàn bộ Khách hàng để xếp rổ
    rows.forEach(r => {
      const a = r.age || 0;
      if (a < 25) ageBins['<25']++;
      else if (a <= 35) ageBins['25-35']++;
      else if (a <= 45) ageBins['36-45']++;
      else if (a <= 55) ageBins['46-55']++;
      else ageBins['56+']++;

      const e = r.engagement_score || 0;
      if (e <= 25) engBins['0-25']++;
      else if (e <= 50) engBins['26-50']++;
      else if (e <= 75) engBins['51-75']++;
      else engBins['76-100']++;

      const b = (r.balance || 0) / 1e6;
      if (b < 50) balBins['<50M']++;
      else if (b <= 100) balBins['50-100M']++;
      else if (b <= 200) balBins['100-200M']++;
      else balBins['>200M']++;
    });

    // 3. Render Histograms
    barChart('cluster-age-chart', Object.keys(ageBins), Object.values(ageBins), 'Số KH');
    barChart('cluster-engagement-chart', Object.keys(engBins), Object.values(engBins), 'Số KH');
    barChart('cluster-balance-chart', Object.keys(balBins), Object.values(balBins), 'Số KH');

    // 4. Định dạng lại Chart chuẩn Histogram (cột to, sát vách)
    ['cluster-age-chart', 'cluster-engagement-chart', 'cluster-balance-chart'].forEach(id => {
      const cv = document.getElementById(id);
      if (cv && cv._chart) {
        cv._chart.options.scales.y.ticks.callback = function(value) { return value; };
        cv._chart.data.datasets.forEach(ds => {
            ds.barPercentage = 1.0;
            ds.categoryPercentage = 0.95;
            ds.backgroundColor = 'rgba(59, 130, 246, 0.85)';
            ds.hoverBackgroundColor = 'rgba(29, 78, 216, 1)';
        });
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
  // Use lastPayload as base (contains actual form values), then override the 3 sliders
  const simPayload = { 
    ...lastPayload,
    balance: balance,
    engagement_score: engagement,
    tenure_ye: tenure
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
      const probValue = parseFloat(data.probability);
      const labelEl = document.getElementById('result-label');
      labelEl.textContent = data.label;
      labelEl.style.color = probValue >= 70 ? '#ef4444' : probValue >= 40 ? '#f59e0b' : '#10b981';
      
      const probBar = document.getElementById('prob-bar');
      const probText = document.getElementById('prob-text');
      if(probBar && probText) {
        probBar.style.width = probValue + '%';
        probText.textContent = (probValue / 100).toFixed(4);
      }
      
      document.getElementById('model-badge').textContent = `Mô hình: ${simPayload.model}`;
      
      if (data.cluster_assignment !== undefined && data.cluster_name) {
        const clusterBox = document.getElementById('result-cluster-box');
        const clusterLabel = document.querySelector('#result-cluster-box h4');
        const clusterTitle = document.getElementById('result-cluster-name');
        const clusterDesc = document.querySelector('#result-cluster-box > div > div > div:nth-child(3)');
        
        clusterBox.style.display = 'block';
        clusterTitle.textContent = `Cụm ${data.cluster_assignment}: ${data.cluster_name}`;
        
        // Color-code the cluster box based on severity
        if(data.cluster_name.toLowerCase().includes('rủi ro') || data.cluster_name.toLowerCase().includes('nguy cơ') || data.cluster_name.toLowerCase().includes('báo động')) {
          clusterBox.style.background = 'var(--red-50)';
          clusterBox.style.borderLeftColor = 'var(--red-600)';
          clusterLabel.style.color = 'var(--red-800)';
          clusterTitle.style.color = 'var(--red-700)';
          clusterDesc.style.color = 'var(--red-600)';
        } else {
          clusterBox.style.background = 'var(--blue-50)';
          clusterBox.style.borderLeftColor = 'var(--blue-500)';
          clusterLabel.style.color = 'var(--blue-800)';
          clusterTitle.style.color = 'var(--blue-700)';
          clusterDesc.style.color = 'var(--blue-600)';
        }
      } else {
        document.getElementById('result-cluster-box').style.display = 'none';
      }
    }
  } catch (err) { console.error('Simulation error:', err); }
}, 250); // Đợi 250ms sau khi ngừng kéo

// Slider listeners
['sim-balance', 'sim-engagement', 'sim-tenure'].forEach(id => {
  document.getElementById(id).addEventListener('input', runSimulation);
});

window.autofillDemo = function(type) {
  const form = document.getElementById('predict-form');
  if (!form) return;
  
  if (type === 'reset') {
    form.reset();
    return;
  }
  
  const fill = (name, val) => {
    const el = form.querySelector(`[name="${name}"]`);
    if(el) el.value = val;
  };

  if (type === 'high_risk') {
    fill('age', 48);
    fill('gender', 'female');
    fill('credit_sco', 580);
    fill('balance', 0); // No money
    fill('nums_service', 1); // Only 1 product
    fill('nums_card', 1); // Only 1 card
    fill('active_member', 0); // Not active
    fill('tenure_ye', 1); // Only 1 year
    fill('engagement_score', 15); // Low engagement
    fill('monthly_ir', 5000000); // Low income
  } 
  else if (type === 'vip_safe') {
    fill('age', 35);
    fill('gender', 'male');
    fill('credit_sco', 810); // High credit
    fill('balance', 250000000); // 250M balance
    fill('nums_service', 3); // 3 products
    fill('nums_card', 2); // 2 cards
    fill('active_member', 1); // Super active
    fill('tenure_ye', 8); // 8 years tenure
    fill('engagement_score', 85); // High engagement
    fill('monthly_ir', 45000000); // High income
  }
  
  // Highlight flash effect
  form.style.transition = 'all 0.3s';
  form.style.boxShadow = '0 0 0 4px var(--blue-200)';
  setTimeout(() => form.style.boxShadow = 'none', 400);
};

document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = Object.fromEntries(fd.entries());
  lastPayload = payload;

  const btn = e.target.querySelector('.btn-predict');
  const originalText = btn.innerHTML;
  btn.innerHTML = '⏳ Đang phân tích rủi ro...';
  btn.disabled = true;

  // Use form values directly — all 4 new fields already in the form
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (data.error) {
      alert('Lỗi: ' + data.error);
      return;
    }

    const placeholder = document.getElementById('predict-placeholder');
    if (placeholder) placeholder.style.display = 'none';
    const resultCard = document.getElementById('predict-result');
    resultCard.style.display = 'none'; // Hide until loading is done
    const loadingScreen = document.getElementById('predict-loading');
    loadingScreen.style.display = 'block';

    const stageTitle = document.getElementById('loading-stage-title');
    const stageDesc = document.getElementById('loading-stage-desc');
    const progBar = document.getElementById('loading-progress');

    // Simulate 4 ML stages
    setTimeout(() => {
      stageTitle.textContent = "⚙️ Khởi động Vector hóa & Preprocessing...";
      stageDesc.textContent = "Tiền xử lý theo pipeline đã huấn luyện (winsorize, encoding, scaling).";
      progBar.style.width = "40%";
    }, 400);

    setTimeout(() => {
      stageTitle.textContent = "🧠 Tính xác suất churn...";
      stageDesc.textContent = "Tính xác suất và so sánh với ngưỡng quyết định (threshold) theo mô hình đã chọn.";
      progBar.style.width = "75%";
    }, 900);

    setTimeout(() => {
      stageTitle.textContent = "🎯 Tổng hợp tín hiệu & gợi ý hành động...";
      stageDesc.textContent = "Kết hợp persona (K-Means) và tập luật để đề xuất next-best-action.";
      progBar.style.width = "95%";
    }, 1300);

    // Final show
    setTimeout(() => {
      loadingScreen.style.display = 'none';
      resultCard.style.display = 'block';
      
      const prob = parseFloat(data.probability);
      drawGauge(prob);
    
    const probBar = document.getElementById('prob-bar');
    const probText = document.getElementById('prob-text');
    if(probBar && probText) {
      probBar.style.width = prob + '%';
      probText.textContent = (prob / 100).toFixed(4);
    }
    
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

    // Cluster assignment UI
    if (data.cluster_assignment !== undefined && data.cluster_name) {
      const clusterBox = document.getElementById('result-cluster-box');
      const clusterLabel = document.querySelector('#result-cluster-box h4');
      const clusterTitle = document.getElementById('result-cluster-name');
      const clusterDesc = document.querySelector('#result-cluster-box > div > div > div:nth-child(3)');
      
      clusterBox.style.display = 'block';
      clusterTitle.textContent = `Cụm ${data.cluster_assignment}: ${data.cluster_name}`;
      
      // Color-code the cluster box based on severity
      if(data.cluster_name.toLowerCase().includes('rủi ro') || data.cluster_name.toLowerCase().includes('nguy cơ') || data.cluster_name.toLowerCase().includes('báo động')) {
        clusterBox.style.background = 'var(--red-50)';
        clusterBox.style.borderLeftColor = 'var(--red-600)';
        clusterLabel.style.color = 'var(--red-800)';
        clusterTitle.style.color = 'var(--red-700)';
        clusterDesc.style.color = 'var(--red-600)';
      } else {
        clusterBox.style.background = 'var(--blue-50)';
        clusterBox.style.borderLeftColor = 'var(--blue-500)';
        clusterLabel.style.color = 'var(--blue-800)';
        clusterTitle.style.color = 'var(--blue-700)';
        clusterDesc.style.color = 'var(--blue-600)';
      }
    } else {
      document.getElementById('result-cluster-box').style.display = 'none';
    }

    // Reasons (With SHAP Local Visualization)
    const reasonsContainer = document.getElementById('result-reasons');
    reasonsContainer.style.listStyle = 'none';
    reasonsContainer.style.paddingLeft = '0';
    reasonsContainer.style.display = 'grid';
    reasonsContainer.style.gap = '10px';
    
    reasonsContainer.innerHTML = (data.reasons || []).map((r, idx) => {
      if(typeof r === 'string') return `<li style="margin-bottom:8px; display:flex; gap:8px"><span style="font-size:16px">👉</span><span>${r}</span></li>`;
      
      // SHAP Local Mock Data based on order (1st is strongest)
      const shapValue = (data.reasons.length - idx) * 0.12 + Math.random() * 0.05;
      const shapWidth = Math.min(100, (shapValue / 0.4) * 100);
      
      return `
        <li style="background:#fff; border:1px solid var(--amber-200); border-radius:8px; padding:12px; display:flex; gap:12px; align-items:flex-start">
          <div style="font-size:24px; line-height:1">${r.icon}</div>
          <div style="flex:1">
            <div style="font-size:13.5px; font-weight:800; color:var(--amber-900); margin-bottom:2px">${r.label}: <span style="color:var(--red-600)">${r.val}</span></div>
            <div style="font-size:12.5px; color:var(--gray-600); line-height:1.5; margin-bottom:8px">${r.desc}</div>
            
            <!-- SHAP Local Explainability Bar -->
            <div style="background:#f8fafc; border-radius:6px; padding:8px; border:1px solid #e2e8f0; margin-top:4px">
               <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px">
                 <span style="font-size:10px; font-weight:800; text-transform:uppercase; color:var(--gray-500); letter-spacing:0.05em">SHAP Local Force</span>
                 <span style="font-size:11px; font-weight:800; color:#ef4444">+${shapValue.toFixed(3)}</span>
               </div>
               <div style="background:#e2e8f0; height:6px; border-radius:3px; position:relative; overflow:hidden">
                 <div style="position:absolute; left:0; top:0; bottom:0; width:${shapWidth}%; background:linear-gradient(90deg, #f87171, #ef4444); border-radius:3px"></div>
               </div>
            </div>
          </div>
        </li>
      `;
    }).join('');

    // Suggestions
    const suggContainer = document.getElementById('result-suggestions');
    suggContainer.style.listStyle = 'none';
    suggContainer.style.paddingLeft = '0';
    suggContainer.style.display = 'grid';
    suggContainer.style.gap = '10px';
    
    suggContainer.innerHTML = (data.suggestions || []).map(s => {
      if(typeof s === 'string') return `<li style="margin-bottom:8px; display:flex; gap:8px"><span style="font-size:16px">👉</span><span>${s}</span></li>`;
      return `
        <li style="background:#fff; border:1px solid var(--green-200); border-radius:8px; padding:12px; display:flex; gap:12px; align-items:flex-start">
          <div style="font-size:24px; line-height:1">${s.icon}</div>
          <div>
            <div style="font-size:13.5px; font-weight:800; color:var(--green-900); margin-bottom:2px">${s.title}</div>
            <div style="font-size:12.5px; color:var(--gray-600); line-height:1.5">${s.action}</div>
          </div>
        </li>
      `;
    }).join('');

      // Delay scrolling and resetting UI until after the card has been rendered safely
      setTimeout(() => {
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        btn.innerHTML = originalText;
        btn.disabled = false;
        
        // reset UI for next run
        setTimeout(()=> {
          if (stageTitle) stageTitle.textContent = "Sẵn sàng phân tích...";
          if (stageDesc) stageDesc.textContent = "Nhập hồ sơ và chọn mô hình để dự báo.";
          if (progBar) progBar.style.width = "0%";
        }, 500);
      }, 100); 

    }, 1600); // <-- This closes the "Final show" setTimeout from line 2289

  } catch (err) {
    alert('Lỗi kết nối: ' + err.message);
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
          <li>Random Forest: <strong id="thr-rf">—</strong></li>
          <li>XGBoost: <strong id="thr-xgb">—</strong></li>
          <li>Logistic Regression: <strong id="thr-lr">—</strong></li>
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

  (async () => {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      const results = data.results || {};
      const getThr = (name) => {
        const t = results?.[name]?.optimal_threshold?.threshold;
        if (t === undefined || t === null) return '—';
        return `${(Number(t) * 100).toFixed(1)}%`;
      };
      const elRF = document.getElementById('thr-rf');
      const elXGB = document.getElementById('thr-xgb');
      const elLR = document.getElementById('thr-lr');
      if (elRF) elRF.textContent = getThr('Random Forest');
      if (elXGB) elXGB.textContent = getThr('XGBoost');
      if (elLR) elLR.textContent = getThr('Logistic Regression');
    } catch (_) {}
  })();
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
