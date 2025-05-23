import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

# ---------- データ処理関数群 ----------

def parse_time_to_seconds(time_str):
    if not isinstance(time_str, str):
        raise ValueError(f"Invalid time value: {time_str}")
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {time_str}")
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s

def aggregate_conditions(df, temp_col='T_c1', ripple_col='I_ripple'):
    df['seconds'] = df['Time'].apply(parse_time_to_seconds)
    df['delta_seconds'] = df['seconds'].diff().fillna(10)
    df['temp_change'] = df[temp_col].diff().fillna(0).abs() > 1e-6
    df['ripple_change'] = df[ripple_col].diff().fillna(0).abs() > 1e-6
    df['condition_change'] = df['temp_change'] | df['ripple_change'] | (df['delta_seconds'] != 10)
    df['group'] = df['condition_change'].cumsum()
    grouped = df.groupby('group').agg(
        T_n=('T_c1', 'mean'),
        I_n=('I_ripple', 'mean'),
        start_s=('seconds', 'min'),
        end_s=('seconds', 'max'),
        count=('seconds', 'count')
    ).reset_index(drop=True)
    grouped['used_seconds'] = grouped['end_s'] - grouped['start_s'] + 10
    grouped['used_hours'] = grouped['used_seconds'] / 3600
    return grouped[['T_n', 'I_n', 'used_hours']]

def get_k_coefficient(T_n):
    temps = [45, 55, 65, 75, 85]
    coeffs = [2, 2, 2, 2, 1.6]
    return np.interp(T_n, temps, coeffs)

def estimate_lifetime(T_n, I_n, L_0=3000, T_0=85, delta_t_0=8, K=5, I_m=3.7):
    k_coeff = get_k_coefficient(T_n)
    K_eff = K * k_coeff
    delta_t_n = delta_t_0 * (I_n / I_m) ** 2
    temp_factor = 2 ** ((T_0 - (40 if T_n <= 40 else T_n)) / 10)
    L_n = L_0 * temp_factor * 2 ** (1 - (delta_t_n / K_eff))
    return L_n

def process_single_file(file, C_0=900):
    df_raw = pd.read_csv(file, dtype={'Time': str})
    df_conditions = aggregate_conditions(df_raw)
    h_0 = df_conditions['used_hours'].sum()
    df_conditions['L_n'] = df_conditions.apply(lambda row: estimate_lifetime(row['T_n'], row['I_n']), axis=1)
    df_conditions['h_i_over_L_i'] = df_conditions['used_hours'] / df_conditions['L_n']
    sum_h_over_L = df_conditions['h_i_over_L_i'].sum()
    L_total = h_0 / sum_h_over_L if sum_h_over_L > 0 else float('inf')
    remaining_life_hours = max(L_total - h_0, 0)
    remaining_life_years = remaining_life_hours / 24 / 365
    degradation_ratio = min(h_0 / L_total, 1.0) if L_total != float('inf') else 0
    C_now = C_0 * (1 - 0.2 * degradation_ratio)
    remaining_percent = (C_now / C_0) * 100
    degradation_percent = 100 - remaining_percent
    return df_conditions, h_0, L_total, remaining_life_hours, remaining_life_years, C_now, remaining_percent, degradation_percent

def process_multiple_files(files, C_0=900):
    total_h_0 = 0
    total_sum_h_over_L = 0
    all_conditions = []
    for file in files:
        df_cond, h_0, L_total, rem_hours, rem_years, C_now, rem_pct, deg_pct = process_single_file(file, C_0)
        h_over_L = 0 if L_total == float('inf') else h_0 / L_total
        total_h_0 += h_0
        total_sum_h_over_L += h_over_L
        df_cond['source_file'] = os.path.basename(file.name)
        all_conditions.append(df_cond)
    L_total_all = total_h_0 / total_sum_h_over_L if total_sum_h_over_L > 0 else float('inf')
    remaining_life_hours_all = max(L_total_all - total_h_0, 0)
    remaining_life_years_all = remaining_life_hours_all / 24 / 365
    degradation_ratio_all = min(total_h_0 / L_total_all, 1.0) if L_total_all != float('inf') else 0
    C_now_all = C_0 * (1 - 0.2 * degradation_ratio_all)
    remaining_percent_all = (C_now_all / C_0) * 100
    degradation_percent_all = 100 - remaining_percent_all
    all_conditions_df = pd.concat(all_conditions, ignore_index=True)
    return all_conditions_df, total_h_0, L_total_all, remaining_life_hours_all, remaining_life_years_all, C_now_all, remaining_percent_all, degradation_percent_all

# -------- Streamlit UI --------

st.title("コンデンサ寿命推定ツール")

uploaded_files = st.file_uploader("CSVファイルを複数選択してください", type=["csv"], accept_multiple_files=True)

capacity_input = st.text_input("初期容量 [μF]", "900")

if st.button("寿命推定を実行"):
    if not uploaded_files:
        st.error("エラー: ファイルが選択されていません。")
    else:
        try:
            C_0 = float(capacity_input)
        except ValueError:
            st.error("エラー: 初期容量に数値を入力してください。")
            st.stop()

        try:
            with st.spinner("計算中..."):
                df_cond, h_0, L_total, rem_hours, rem_years, C_now, rem_pct, deg_pct = process_multiple_files(uploaded_files, C_0)
            
            st.subheader("📊 結果")
            st.write(f"トータル使用時間 h₀ = {h_0:.2f} 時間")
            st.write(f"推定寿命 Lₙ = {L_total:.2f} 時間")
            st.write(f"残余寿命 = {rem_hours:.2f} 時間（約 {rem_years:.2f} 年）")
            st.write(f"残存容量 = {C_now:.2f} μF（{rem_pct:.2f}%）")
            st.write(f"劣化割合 = {deg_pct:.2f}%")

            # グラフ描画
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            x = np.arange(len(df_cond))
            ax1.bar(x, df_cond['used_hours'], width=0.6, color='tab:blue', label='使用時間 [h]')
            ax1.set_title("条件別 使用時間")
            ax1.set_xlabel("グループ番号")
            ax1.set_ylabel("使用時間 [h]")
            ax1.grid(True)
            ax1.legend(loc='upper left')

            ax1_2 = ax1.twinx()
            ax1_2.plot(x, df_cond['T_n'], 'r-o', label="温度 [℃]")
            ax1_2.plot(x, df_cond['I_n'], 'g-x', label="リプル電流 [A]")
            ax1_2.set_ylabel("温度 / リプル電流")
            ax1_2.legend(loc='upper right')

            ax2.bar(['残存容量'], [rem_pct], color='orange')
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('残存容量 [%]')
            ax2.set_title(f'残存容量: {C_now:.2f} μF ({rem_pct:.2f}%)')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)

            fig.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"処理中にエラーが発生しました: {e}")
