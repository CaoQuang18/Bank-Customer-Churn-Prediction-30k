import pandas as pd
import numpy as np
import json
import os
from pipeline.data_cleaning import load_data, clean_pipeline

# ══════════════════════════════════════════════════════════════════════════════
# PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)
# Mục tiêu: Hiểu sâu từng đặc trưng, mối quan hệ với churn,
# từ đó đưa ra suy luận có căn cứ trước khi xây dựng mô hình.
# ══════════════════════════════════════════════════════════════════════════════

def eda_overview(df):
    """Tổng quan nhanh về dataset sau làm sạch."""
    total       = len(df)
    churn       = int(df['exit'].sum())
    no_churn    = total - churn
    churn_rate  = round(churn / total * 100, 2)
    return {
        "total":      total,
        "churn":      churn,
        "no_churn":   no_churn,
        "churn_rate": churn_rate,
        # Suy luận: Tỉ lệ churn phản ánh mức độ giữ chân khách hàng
        # của ngân hàng. Nếu > 20% là dấu hiệu đáng lo ngại.
        "insight": (
            f"Tỉ lệ churn {churn_rate}% cho thấy cứ 100 khách hàng thì có "
            f"{round(churn_rate)} người rời bỏ. "
            + ("Đây là mức đáng lo ngại, cần chiến lược giữ chân ngay." if churn_rate > 20
               else "Mức tương đối kiểm soát được nhưng vẫn cần cải thiện.")
        )
    }

def eda_age(df):
    """
    Nguyên nhân: Độ tuổi quyết định nhu cầu tài chính và mức độ gắn bó.
    Lý do tại sao: Mỗi nhóm tuổi có mục tiêu tài chính khác nhau,
    nếu ngân hàng không đáp ứng đúng nhu cầu -> khách hàng rời bỏ.
    """
    bins   = [18, 30, 40, 50, 60, 70, 100]
    labels = ['18-30', '31-40', '41-50', '51-60', '61-70', '70+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    age_churn = df.groupby('age_group', observed=True)['exit'].agg(['mean', 'count']).reset_index()
    age_churn['churn_rate'] = (age_churn['mean'] * 100).round(2)

    result = {
        "labels": labels,
        "churn_rates": age_churn['churn_rate'].tolist(),
        "counts": age_churn['count'].tolist(),
    }

    # Tìm nhóm churn cao nhất
    max_idx  = age_churn['churn_rate'].idxmax()
    max_grp  = age_churn.loc[max_idx, 'age_group']
    max_rate = age_churn.loc[max_idx, 'churn_rate']

    result["insight"] = {
        "so_lieu": f"Nhóm tuổi {max_grp} có churn rate cao nhất: {max_rate}%",
        "nguyen_nhan": "Độ tuổi ảnh hưởng trực tiếp đến nhu cầu tài chính và hành vi sử dụng dịch vụ ngân hàng",
        "ly_do": f"Nhóm {max_grp} đang ở giai đoạn chuyển tiếp tài chính quan trọng, dễ tìm kiếm sản phẩm phù hợp hơn ở nơi khác",
        "huong_xu_ly": "Thiết kế sản phẩm tài chính đặc thù theo từng nhóm tuổi",
        "giai_phap": f"Với nhóm {max_grp}: tư vấn tài chính cá nhân, ưu đãi sản phẩm phù hợp giai đoạn sống"
    }
    return result

def eda_gender(df):
    """
    Nguyên nhân: Giới tính có thể phản ánh sự khác biệt trong hành vi
    tài chính và mức độ trung thành với ngân hàng.
    """
    gender_map = {0: 'Nữ', 1: 'Nam'} if df['gender'].dtype != object else {}
    g = df.groupby('gender')['exit'].agg(['mean', 'count']).reset_index()
    g['churn_rate'] = (g['mean'] * 100).round(2)

    labels = g['gender'].astype(str).tolist()
    rates  = g['churn_rate'].tolist()
    counts = g['count'].tolist()

    max_idx = g['churn_rate'].idxmax()
    max_gender = labels[max_idx]
    max_rate   = rates[max_idx]

    return {
        "labels": labels,
        "churn_rates": rates,
        "counts": counts,
        "insight": {
            "so_lieu": f"Giới tính '{max_gender}' có churn rate cao hơn: {max_rate}%",
            "nguyen_nhan": "Sự khác biệt giới tính dẫn đến nhu cầu và kỳ vọng dịch vụ khác nhau",
            "ly_do": "Ngân hàng có thể chưa có sản phẩm/dịch vụ đủ hấp dẫn cho nhóm này",
            "huong_xu_ly": "Nghiên cứu hành vi tài chính theo giới tính để thiết kế sản phẩm phù hợp",
            "giai_phap": "Chương trình ưu đãi riêng, marketing cá nhân hóa theo giới tính"
        }
    }

def eda_segment(df):
    """
    Nguyên nhân: Phân khúc khách hàng phản ánh giá trị và mức độ
    quan trọng của từng nhóm với ngân hàng.
    """
    seg = df.groupby('customer_segment')['exit'].agg(['mean', 'count']).reset_index()
    seg['churn_rate'] = (seg['mean'] * 100).round(2)

    max_idx  = seg['churn_rate'].idxmax()
    max_seg  = seg.loc[max_idx, 'customer_segment']
    max_rate = seg.loc[max_idx, 'churn_rate']

    return {
        "labels": seg['customer_segment'].tolist(),
        "churn_rates": seg['churn_rate'].tolist(),
        "counts": seg['count'].tolist(),
        "insight": {
            "so_lieu": f"Phân khúc '{max_seg}' có churn rate cao nhất: {max_rate}%",
            "nguyen_nhan": "Mỗi phân khúc có kỳ vọng dịch vụ và ngưỡng hài lòng khác nhau",
            "ly_do": f"Phân khúc {max_seg} có thể chưa nhận được đủ giá trị tương xứng từ ngân hàng",
            "huong_xu_ly": "Rà soát lại gói dịch vụ và quyền lợi cho từng phân khúc",
            "giai_phap": "Nâng cấp quyền lợi, tạo lộ trình thăng hạng rõ ràng cho khách hàng"
        }
    }

def eda_loyalty(df):
    """
    Nguyên nhân: Loyalty level phản ánh mức độ gắn bó lịch sử
    của khách hàng với ngân hàng.
    """
    loy = df.groupby('loyalty_level')['exit'].agg(['mean', 'count']).reset_index()
    loy['churn_rate'] = (loy['mean'] * 100).round(2)

    max_idx  = loy['churn_rate'].idxmax()
    max_loy  = loy.loc[max_idx, 'loyalty_level']
    max_rate = loy.loc[max_idx, 'churn_rate']

    return {
        "labels": loy['loyalty_level'].tolist(),
        "churn_rates": loy['churn_rate'].tolist(),
        "counts": loy['count'].tolist(),
        "insight": {
            "so_lieu": f"Hạng loyalty '{max_loy}' có churn rate cao nhất: {max_rate}%",
            "nguyen_nhan": "Khách hàng hạng thấp chưa cảm nhận đủ giá trị từ chương trình loyalty",
            "ly_do": "Điểm thưởng và quyền lợi chưa đủ hấp dẫn để giữ chân nhóm này",
            "huong_xu_ly": "Cải thiện chương trình loyalty, tạo động lực tích điểm rõ ràng hơn",
            "giai_phap": "Tặng điểm bonus, ưu đãi phí dịch vụ, lộ trình lên hạng nhanh hơn"
        }
    }

def eda_digital(df):
    """
    Nguyên nhân: Hành vi digital phản ánh mức độ tương tác và
    sự tiện lợi mà khách hàng cảm nhận từ ngân hàng.
    """
    dig = df.groupby('digital_behavior')['exit'].agg(['mean', 'count']).reset_index()
    dig['churn_rate'] = (dig['mean'] * 100).round(2)

    max_idx  = dig['churn_rate'].idxmax()
    max_dig  = dig.loc[max_idx, 'digital_behavior']
    max_rate = dig.loc[max_idx, 'churn_rate']

    return {
        "labels": dig['digital_behavior'].tolist(),
        "churn_rates": dig['churn_rate'].tolist(),
        "counts": dig['count'].tolist(),
        "insight": {
            "so_lieu": f"Nhóm '{max_dig}' có churn rate cao nhất: {max_rate}%",
            "nguyen_nhan": "Mức độ sử dụng kênh digital ảnh hưởng trực tiếp đến tần suất tương tác",
            "ly_do": "Khách hàng ít dùng digital ít tiếp xúc với sản phẩm/ưu đãi mới của ngân hàng",
            "huong_xu_ly": "Khuyến khích chuyển đổi số, đơn giản hóa trải nghiệm app",
            "giai_phap": "Ưu đãi phí 0đ khi dùng app, cashback giao dịch online, hỗ trợ onboarding digital"
        }
    }

def eda_active_member(df):
    """
    Nguyên nhân: Thành viên không hoạt động là dấu hiệu sớm nhất
    của việc sắp rời bỏ ngân hàng.
    """
    act = df.groupby('active_member')['exit'].agg(['mean', 'count']).reset_index()
    act['churn_rate'] = (act['mean'] * 100).round(2)

    inactive_rate = act.loc[act['active_member'] == 0, 'churn_rate'].values
    active_rate   = act.loc[act['active_member'] == 1, 'churn_rate'].values

    inactive_r = float(inactive_rate[0]) if len(inactive_rate) > 0 else 0
    active_r   = float(active_rate[0])   if len(active_rate) > 0   else 0

    return {
        "labels": act['active_member'].astype(str).tolist(),
        "churn_rates": act['churn_rate'].tolist(),
        "counts": act['count'].tolist(),
        "insight": {
            "so_lieu": f"Thành viên không hoạt động: churn {inactive_r}% vs hoạt động: {active_r}%",
            "nguyen_nhan": "Không hoạt động = không có nhu cầu hoặc đã chuyển sang ngân hàng khác",
            "ly_do": f"Chênh lệch {round(inactive_r - active_r, 1)}% cho thấy engagement là yếu tố bảo vệ mạnh nhất",
            "huong_xu_ly": "Phát hiện sớm khách hàng ngừng hoạt động và can thiệp kịp thời",
            "giai_phap": "Gửi thông báo cá nhân hóa, ưu đãi kích hoạt lại tài khoản, gọi điện tư vấn"
        }
    }

def eda_balance(df):
    """
    Nguyên nhân: Số dư tài khoản phản ánh mức độ cam kết tài chính
    của khách hàng với ngân hàng.
    """
    churn_bal    = df[df['exit'] == 1]['balance']
    no_churn_bal = df[df['exit'] == 0]['balance']

    result = {
        "churn_mean":    round(float(churn_bal.mean()), 0),
        "no_churn_mean": round(float(no_churn_bal.mean()), 0),
        "churn_median":  round(float(churn_bal.median()), 0),
        "no_churn_median": round(float(no_churn_bal.median()), 0),
        # Histogram bins
        "churn_hist":    np.histogram(churn_bal, bins=20)[0].tolist(),
        "no_churn_hist": np.histogram(no_churn_bal, bins=20)[0].tolist(),
        "bin_edges":     np.histogram(df['balance'], bins=20)[1].tolist(),
    }

    diff = round((result['churn_mean'] - result['no_churn_mean']) / 1e6, 1)
    result["insight"] = {
        "so_lieu": f"Khách hàng churn có balance trung bình {result['churn_mean']/1e6:.1f}M vs không churn {result['no_churn_mean']/1e6:.1f}M",
        "nguyen_nhan": "Số dư tài khoản phản ánh mức độ phụ thuộc tài chính vào ngân hàng",
        "ly_do": "Khách hàng balance thấp ít bị ràng buộc, dễ chuyển sang ngân hàng khác hơn",
        "huong_xu_ly": "Khuyến khích tăng số dư thông qua lãi suất hấp dẫn",
        "giai_phap": "Gói tiết kiệm lãi suất bậc thang, thưởng khi duy trì số dư tối thiểu"
    }
    return result

def eda_credit_score(df):
    """
    Nguyên nhân: Điểm tín dụng phản ánh lịch sử tài chính và
    mức độ tin tưởng của ngân hàng với khách hàng.
    """
    bins   = [300, 500, 600, 650, 700, 750, 800]
    labels = ['<500', '500-600', '600-650', '650-700', '700-750', '750+']
    df['credit_group'] = pd.cut(df['credit_sco'], bins=bins, labels=labels)

    cr = df.groupby('credit_group', observed=True)['exit'].agg(['mean', 'count']).reset_index()
    cr['churn_rate'] = (cr['mean'] * 100).round(2)

    max_idx  = cr['churn_rate'].idxmax()
    max_grp  = cr.loc[max_idx, 'credit_group']
    max_rate = cr.loc[max_idx, 'churn_rate']

    return {
        "labels": labels,
        "churn_rates": cr['churn_rate'].tolist(),
        "counts": cr['count'].tolist(),
        "insight": {
            "so_lieu": f"Nhóm credit score '{max_grp}' có churn rate cao nhất: {max_rate}%",
            "nguyen_nhan": "Điểm tín dụng thấp thường đi kèm với trải nghiệm dịch vụ hạn chế",
            "ly_do": "Khách hàng điểm thấp bị từ chối nhiều sản phẩm -> cảm thấy không được phục vụ tốt",
            "huong_xu_ly": "Tạo lộ trình cải thiện điểm tín dụng, sản phẩm phù hợp cho từng mức điểm",
            "giai_phap": "Tư vấn tài chính miễn phí, sản phẩm vay có bảo đảm cho nhóm điểm thấp"
        }
    }

def eda_occupation(df):
    """
    Nguyên nhân: Nghề nghiệp quyết định thu nhập, sự ổn định tài chính
    và nhu cầu sản phẩm ngân hàng.
    """
    occ = df.groupby('occupation')['exit'].agg(['mean', 'count']).reset_index()
    occ['churn_rate'] = (occ['mean'] * 100).round(2)
    occ = occ.sort_values('churn_rate', ascending=False)

    max_occ  = occ.iloc[0]['occupation']
    max_rate = occ.iloc[0]['churn_rate']

    return {
        "labels": occ['occupation'].tolist(),
        "churn_rates": occ['churn_rate'].tolist(),
        "counts": occ['count'].tolist(),
        "insight": {
            "so_lieu": f"Nghề '{max_occ}' có churn rate cao nhất: {max_rate}%",
            "nguyen_nhan": "Nghề nghiệp ảnh hưởng đến thu nhập, nhu cầu vay vốn và đầu tư",
            "ly_do": "Một số nghề có thu nhập không ổn định -> khó duy trì cam kết tài chính dài hạn",
            "huong_xu_ly": "Thiết kế sản phẩm linh hoạt phù hợp với đặc thù từng nghề",
            "giai_phap": "Gói vay linh hoạt cho freelancer, bảo hiểm thu nhập cho lao động phổ thông"
        }
    }

def eda_province(df):
    """
    Nguyên nhân: Địa lý ảnh hưởng đến khả năng tiếp cận dịch vụ
    và mức độ cạnh tranh từ các ngân hàng khác.
    """
    prov = df.groupby('origin_province')['exit'].agg(['mean', 'count']).reset_index()
    prov['churn_rate'] = (prov['mean'] * 100).round(2)
    prov = prov.sort_values('churn_rate', ascending=False).head(10)

    max_prov = prov.iloc[0]['origin_province']
    max_rate = prov.iloc[0]['churn_rate']

    return {
        "labels": prov['origin_province'].tolist(),
        "churn_rates": prov['churn_rate'].tolist(),
        "counts": prov['count'].tolist(),
        "insight": {
            "so_lieu": f"Tỉnh '{max_prov}' có churn rate cao nhất trong top 10: {max_rate}%",
            "nguyen_nhan": "Mức độ cạnh tranh ngân hàng và khả năng tiếp cận chi nhánh khác nhau theo vùng",
            "ly_do": "Vùng có nhiều ngân hàng cạnh tranh -> khách hàng có nhiều lựa chọn hơn",
            "huong_xu_ly": "Tăng cường hiện diện và chất lượng dịch vụ tại các vùng churn cao",
            "giai_phap": "Mở thêm điểm giao dịch, tăng cường digital banking tại vùng xa"
        }
    }

def eda_correlation(df):
    """
    Nguyên nhân: Tương quan giữa các đặc trưng giúp phát hiện
    multicollinearity và các yếu tố liên quan nhất đến churn.
    """
    num_cols = ['credit_sco', 'age', 'balance', 'monthly_ir', 'tenure_ye',
                'nums_card', 'nums_service', 'engagement_score', 'risk_score', 'exit']
    corr = df[num_cols].corr()['exit'].drop('exit').sort_values(key=abs, ascending=False)

    top_pos = corr[corr > 0].head(3)
    top_neg = corr[corr < 0].head(3)

    return {
        "features": corr.index.tolist(),
        "correlations": corr.round(4).tolist(),
        "insight": {
            "so_lieu": f"Top tương quan dương: {top_pos.index.tolist()} | Âm: {top_neg.index.tolist()}",
            "nguyen_nhan": "Các đặc trưng tương quan cao với exit là yếu tố dự báo churn quan trọng nhất",
            "ly_do": "Tương quan dương -> tăng giá trị đặc trưng -> tăng khả năng churn và ngược lại",
            "huong_xu_ly": "Ưu tiên các đặc trưng tương quan cao trong feature selection",
            "giai_phap": "Tập trung can thiệp vào các yếu tố có tương quan âm mạnh để giảm churn"
        }
    }

def run_eda(df):
    eda_data = {}
    eda_data['overview']      = eda_overview(df)
    eda_data['age']           = eda_age(df)
    eda_data['gender']        = eda_gender(df)
    eda_data['segment']       = eda_segment(df)
    eda_data['loyalty']       = eda_loyalty(df)
    eda_data['digital']       = eda_digital(df)
    eda_data['active_member'] = eda_active_member(df)
    eda_data['balance']       = eda_balance(df)
    eda_data['credit_score']  = eda_credit_score(df)
    eda_data['occupation']    = eda_occupation(df)
    eda_data['province']      = eda_province(df)
    eda_data['correlation']   = eda_correlation(df)
    return eda_data

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("powerbi", exist_ok=True)

    df_raw = load_data()
    df, _  = clean_pipeline(df_raw)

    eda_data = run_eda(df)

    with open("outputs/eda.json", "w", encoding="utf-8") as f:
        json.dump(eda_data, f, ensure_ascii=False, indent=2)

    # Xuất PowerBI
    powerbi_rows = []
    for feature in ['age_group', 'gender', 'customer_segment', 'loyalty_level',
                     'digital_behavior', 'active_member', 'occupation', 'origin_province']:
        if feature not in df.columns:
            continue
        g = df.groupby(feature)['exit'].agg(['mean', 'count']).reset_index()
        g.columns = ['group_value', 'churn_rate', 'count']
        g['feature']    = feature
        g['churn_rate'] = (g['churn_rate'] * 100).round(2)
        powerbi_rows.append(g)

    pd.concat(powerbi_rows).to_csv("powerbi/churn_by_feature.csv", index=False)

    print("[OK] EDA hoàn tất -> outputs/eda.json")
    print("[OK] PowerBI data -> powerbi/churn_by_feature.csv")
    print(f"\n[PLOT] Tổng quan: {eda_data['overview']['total']:,} KH | Churn rate: {eda_data['overview']['churn_rate']}%")
