import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
import hashlib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- 웹 앱 기본 설정 ---
st.set_page_config(page_title="QRA 대화형 매핑 및 자동화 대시보드", layout="wide")
st.title("🔥 QRA Interactive Mapping & Analysis Dashboard")
st.markdown("도면 상의 누출점 및 작업구역을 직접 지정하고 QRA 분석 엔진을 통해 위험도 등고선을 렌더링합니다.")

# --- CSS를 통한 툴박스 글씨 크기 확대 ---
st.markdown("""
<style>
div[role="radiogroup"] > label > div > p {
    font-size: 1.15rem !important; 
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# --- 사이드바: 입력 패널 ---
st.sidebar.header("📁 Data Input")
uploaded_image = st.sidebar.file_uploader("1. 도면 이미지 업로드 (.png, .jpg)", type=['png', 'jpg', 'jpeg'])
real_width_m = st.sidebar.number_input("2. 도면 실제 가로 길이 (m)", min_value=1.0, value=1000.0, step=10.0)
uploaded_excel = st.sidebar.file_uploader("3. 마스터 엑셀 업로드 (.xlsx)", type=['xlsx'])

def clean_name(n):
    return str(n).strip().lower().replace('_', ' ')

# --- 세션 초기화 ---
if 'qra_calculated' not in st.session_state:
    st.session_state.qra_calculated = False

# --- 메인 화면: 대화형 캔버스 및 매핑 시스템 ---
if uploaded_image is not None and uploaded_excel is not None:
    
    try:
        df_freq = pd.read_excel(uploaded_excel, sheet_name='Leak_Frequencies')
        df_area = pd.read_excel(uploaded_excel, sheet_name='Area_Coordinates')
        
        is_id_list = df_freq['IS_ID'].dropna().astype(str).unique().tolist()
        area_name_list = df_area['Area_Name'].dropna().astype(str).unique().tolist()
    except Exception as e:
        st.error(f"엑셀 데이터 로드 실패. 시트 및 데이터를 확인하십시오.\nError: {e}")
        st.stop()

    # --- 업로드 도면 이미지 처리 ---
    from PIL import ImageOps

    img_original = Image.open(uploaded_image)

    # 휴대폰/스캔 이미지의 EXIF 회전 정보 보정
    img_original = ImageOps.exif_transpose(img_original)

    # 투명 배경 PNG 대응을 위해 RGBA로 변환
    img_original = img_original.convert("RGBA")

    # 투명 배경을 흰색 배경으로 합성
    white_bg = Image.new("RGBA", img_original.size, (255, 255, 255, 255))
    white_bg.alpha_composite(img_original)

    # canvas에는 RGB 이미지로 넘기는 것이 안정적
    bg_image = white_bg.convert("RGB")

    target_width = 800
    original_width, original_height = bg_image.size

    scale_ratio = target_width / original_width
    canvas_width = target_width
    canvas_height = int(original_height * scale_ratio)

    bg_image_resized = bg_image.resize(
        (canvas_width, canvas_height),
        Image.Resampling.LANCZOS
    )

    # 마지막 안전장치: RGB 강제 보장
    bg_image_resized = bg_image_resized.convert("RGB")
    
    st.markdown("---")
    
    col_tool, col_canvas = st.columns([0.7, 2.5])
    
    with col_tool:
        st.subheader("🛠️ Step 1. Toolbox")
        drawing_type = st.radio(
            "그리기 모드 선택",
            ("📍 누출점 (Point)", "🟦 작업구역 (Rect)", "🔄 수정/삭제 (Transform)"),
            help="※ '수정/삭제' 모드에서는 그려진 도형을 마우스로 선택하여 이동하거나 회전할 수 있으며, 키보드의 Delete 키를 눌러 지울 수 있습니다."
        )
        
        if drawing_type == "📍 누출점 (Point)":
            drawing_mode = "point"
            stroke_color = "red" 
            fill_color = "rgba(255, 0, 0, 0.5)"
        elif drawing_type == "🟦 작업구역 (Rect)":
            drawing_mode = "rect"
            stroke_color = "blue" 
            fill_color = "rgba(0, 0, 255, 0.3)"
        else:
            drawing_mode = "transform" 
            stroke_color = "black"
            fill_color = "rgba(0, 0, 0, 0.3)"

    with col_canvas:
        canvas_result = st_canvas(
            fill_color=fill_color,
            stroke_width=3 if drawing_mode == "rect" else 5,
            stroke_color=stroke_color,
            background_color="#FFFFFF",
            background_image=bg_image_resized,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
            display_toolbar=True,
            key="qra_canvas_v2",
        )

    mapped_is_data = {}
    mapped_area_data = {}
    has_duplicate = False 

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        points = [obj for obj in objects if obj["type"] == "circle"]
        rects = [obj for obj in objects if obj["type"] == "rect"]
        
        if points or rects:
            st.markdown("---")
            st.subheader("📝 Step 2. Data Mapping")
            
            col_map_img, col_pt, col_rect = st.columns([2, 1, 1])
            
            with col_map_img:
                if canvas_result.image_data is not None:
                    canvas_array = np.array(canvas_result.image_data, dtype=np.uint8)
                    canvas_pil = Image.fromarray(canvas_array, "RGBA")
                    base_img = bg_image_resized.convert("RGBA")
                    base_img.alpha_composite(canvas_pil)
                    annotated_img = np.array(base_img.convert("RGB"))
                else:
                    annotated_img = np.array(bg_image_resized).copy()
                
                def draw_transparent_label(img, text, x, y, color):
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.5
                    thickness = 1
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    overlay = img.copy()
                    cv2.rectangle(overlay, (x, y - th - 5), (x + tw + 10, y + 5), (255, 255, 255), -1)
                    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
                    cv2.rectangle(img, (x, y - th - 5), (x + tw + 10, y + 5), color, 1)
                    cv2.putText(img, text, (x + 5, y), font, font_scale, color, thickness, cv2.LINE_AA)

                for i, pt in enumerate(points):
                    cx = int(pt["left"] + pt["radius"])
                    cy = int(pt["top"] + pt["radius"])
                    draw_transparent_label(annotated_img, f"Point {i+1}", cx + 12, cy - 12, (255, 0, 0))

                for i, r in enumerate(rects):
                    rx = int(r["left"])
                    ry = int(r["top"])
                    draw_transparent_label(annotated_img, f"Area {i+1}", rx + 12, ry - 12, (0, 0, 255))
                
                st.image(annotated_img, use_container_width=True, caption="Live Mapping Guide")
            
            with col_pt:
                st.markdown("##### 📍 Points")
                if not points: st.info("도면에 누출점이 없습니다.")
                for i, pt in enumerate(points):
                    cx = int(pt["left"] + pt["radius"])
                    cy = int(pt["top"] + pt["radius"])
                    default_idx = i if i < len(is_id_list) else 0
                    user_name = st.selectbox(f"Point {i+1}", options=is_id_list, index=default_idx, key=f"pt_{i}")
                    if user_name in mapped_is_data:
                        st.error(f"'{user_name}' 중복!")
                        has_duplicate = True
                    else:
                        mapped_is_data[user_name] = (cx, cy)
            
            with col_rect:
                st.markdown("##### 🟦 Areas")
                if not rects: st.info("도면에 작업구역이 없습니다.")
                for i, r in enumerate(rects):
                    cx = int(r["left"] + (r["width"] * r["scaleX"]) / 2)
                    cy = int(r["top"] + (r["height"] * r["scaleY"]) / 2)
                    default_idx = i if i < len(area_name_list) else 0
                    user_name = st.selectbox(f"Area {i+1}", options=area_name_list, index=default_idx, key=f"rect_{i}")
                    if user_name in mapped_area_data:
                        st.error(f"'{user_name}' 중복!")
                        has_duplicate = True
                    else:
                        mapped_area_data[str(user_name)] = (cx, cy)

            # ❗ [NEW] 버튼 외부에서 고해상도 좌표 스케일링 사전 준비
            engine_target_width = 4000 
            coord_scale = engine_target_width / canvas_width
            mapped_is_highres = {k: (int(v[0]*coord_scale), int(v[1]*coord_scale)) for k, v in mapped_is_data.items()}
            mapped_areas_highres = {clean_name(k): (int(v[0]*coord_scale), int(v[1]*coord_scale)) for k, v in mapped_area_data.items()}

            # ==========================================
            # Step 3: 엔진 통합 (초고해상도 QRA 분석)
            # ==========================================
            st.write("---")
            if has_duplicate:
                st.warning("⚠️ 중복 선택된 항목을 수정해야 분석을 실행할 수 있습니다.")
            else:
                st.markdown("### 🚀 Step 3. High-Res QRA Analysis Engine")
                
                # 버튼을 누르면 연산 실행 및 세션 저장소(st.session_state)에 결과 기록
                if st.button("QRA 물리 모델 초정밀 연산 및 렌더링 실행", type="primary", use_container_width=True):
                    with st.spinner("AI가 고해상도 환경에서 QRA 모델을 초정밀 연산 중입니다... (잠시만 기다려주세요)"):
                        try:
                            # 엑셀 데이터 로드
                            df_met = pd.read_excel(uploaded_excel, sheet_name='Meteorology')
                            df_cons = pd.read_excel(uploaded_excel, sheet_name='PHAST_Distances')
                            df_occ = pd.read_excel(uploaded_excel, sheet_name='Occupancy')
                            df_vuln = pd.read_excel(uploaded_excel, sheet_name='Vulnerability_Criteria')

                            vuln_vals = df_vuln['Outdoor_Vulnerability'].tolist()
                            vuln_thermal_12, vuln_thermal_37 = float(vuln_vals[0]), float(vuln_vals[1])
                            vuln_flash = float(vuln_vals[2])
                            vuln_exp_03, vuln_exp_05 = float(vuln_vals[3]), float(vuln_vals[4])

                            area_population = {}
                            for col in df_occ.columns[2:]:
                                if 'Unnamed' in str(col): continue
                                clean_col = clean_name(col)
                                area_population[clean_col] = (df_occ['No_of_Personnel'].fillna(0) * df_occ[col].fillna(0)).sum()

                            df_area['Area_Clean'] = df_area['Area_Name'].apply(clean_name)
                            df_area['Population'] = df_area['Area_Clean'].map(area_population).fillna(0.0)

                            engine_scale_ratio = engine_target_width / original_width
                            engine_h = int(original_height * engine_scale_ratio)
                            engine_bg_pil = bg_image.resize((engine_target_width, engine_h), Image.Resampling.LANCZOS)
                            engine_bg_image = np.array(engine_bg_pil) 
                            
                            PIXELS_PER_METER = engine_target_width / real_width_m
                            new_w, new_h = engine_target_width, engine_h

                            lsir_map = np.zeros((new_h, new_w), dtype=np.float32)
                            thermal_map = np.zeros((new_h, new_w), dtype=np.float32)
                            explosion_map = np.zeros((new_h, new_w), dtype=np.float32)
                            y_idx, x_idx = np.indices((new_h, new_w), dtype=np.float32)
                            fn_data = []

                            def parse_val(val):
                                if pd.isna(val) or str(val).strip() in ['NR', 'NA', '']: return 0.0
                                try: return float(val)
                                except: return 0.0

                            for _, row in df_cons.iterrows():
                                is_id = str(row['IS_ID'])
                                if is_id not in mapped_is_highres: continue
                                freq_col = f"Freq_{float(row['Leak_Size_mm']):g}mm"
                                freq_info = df_freq[df_freq['IS_ID'] == is_id]
                                if freq_info.empty or freq_col not in freq_info.columns: continue
                                scenario_freq = float(freq_info[freq_col].values[0])
                                if scenario_freq == 0: continue

                                sx, sy = mapped_is_highres[is_id]
                                r_p12, r_p37 = parse_val(row.get('Pool_12.5')), parse_val(row.get('Pool_37.5'))
                                r_e03, r_e05 = parse_val(row.get('Exp_0.3')), parse_val(row.get('Exp_0.5'))
                                r_j12, r_j37 = parse_val(row.get('Jet_12.5')), parse_val(row.get('Jet_37.5'))
                                r_flash = parse_val(row.get('Flash_LFL'))

                                def apply_circle(r_m, val, target_map, is_max=False):
                                    if r_m <= 0: return
                                    r_px = r_m * PIXELS_PER_METER
                                    x_min, x_max = max(0, int(sx - r_px - 2)), min(new_w, int(sx + r_px + 2))
                                    y_min, y_max = max(0, int(sy - r_px - 2)), min(new_h, int(sy + r_px + 2))
                                    if x_min >= x_max or y_min >= y_max: return
                                    mask = np.sqrt((x_idx[y_min:y_max, x_min:x_max] - sx)**2 + (y_idx[y_min:y_max, x_min:x_max] - sy)**2) <= r_px
                                    if is_max: target_map[y_min:y_max, x_min:x_max][mask] = np.maximum(target_map[y_min:y_max, x_min:x_max][mask], val)
                                    else: target_map[y_min:y_max, x_min:x_max][mask] += val

                                apply_circle(r_p12, scenario_freq * vuln_thermal_12, lsir_map)
                                apply_circle(r_p37, scenario_freq * (vuln_thermal_37 - vuln_thermal_12), lsir_map)
                                apply_circle(r_e03, scenario_freq * vuln_exp_03, lsir_map)
                                apply_circle(r_e05, scenario_freq * (vuln_exp_05 - vuln_exp_03), lsir_map)

                                if r_p12 > 0: apply_circle(r_p12, 12.5, thermal_map, True)
                                if r_p37 > 0: apply_circle(r_p37, 37.5, thermal_map, True)
                                if r_e03 > 0: apply_circle(r_e03, 0.3, explosion_map, True)
                                if r_e05 > 0: apply_circle(r_e05, 0.5, explosion_map, True)

                                met_data = df_met[df_met['Weather_Class'].str.strip() == str(row['Weather_Class']).strip()]
                                for _, m_row in met_data.iterrows():
                                    w_prob = float(m_row['Probability'])
                                    rad = np.radians((float(m_row['Angle_deg']) + 180) % 360)
                                    math_rad = np.radians((90 - ((float(m_row['Angle_deg']) + 180) % 360)) % 360)
                                    cos_a, sin_a = np.cos(math_rad), np.sin(math_rad)

                                    def apply_ellipse(r_m, val, target_map, is_max=False):
                                        if r_m <= 0: return None
                                        a = r_m * PIXELS_PER_METER; b = a * 0.4
                                        cx, cy = sx + a * np.sin(rad), sy - a * np.cos(rad)
                                        x_min, x_max = max(0, int(cx - a*2.5)), min(new_w, int(cx + a*2.5))
                                        y_min, y_max = max(0, int(cy - a*2.5)), min(new_h, int(cy + a*2.5))
                                        if x_min >= x_max or y_min >= y_max: return (cx, cy, a, b)
                                        x_rot = (x_idx[y_min:y_max, x_min:x_max] - cx) * cos_a + (y_idx[y_min:y_max, x_min:x_max] - cy) * sin_a
                                        y_rot = -(x_idx[y_min:y_max, x_min:x_max] - cx) * sin_a + (y_idx[y_min:y_max, x_min:x_max] - cy) * cos_a
                                        mask = (x_rot/a)**2 + (y_rot/b)**2 <= 1.0
                                        if is_max: target_map[y_min:y_max, x_min:x_max][mask] = np.maximum(target_map[y_min:y_max, x_min:x_max][mask], val)
                                        else: target_map[y_min:y_max, x_min:x_max][mask] += (scenario_freq * w_prob * val)
                                        return (cx, cy, a, b)

                                    apply_ellipse(r_j12, vuln_thermal_12, lsir_map)
                                    apply_ellipse(r_j37, (vuln_thermal_37 - vuln_thermal_12), lsir_map)
                                    apply_ellipse(r_flash, vuln_flash, lsir_map)
                                    params_j12 = apply_ellipse(r_j12, 12.5, thermal_map, True) if r_j12 > 0 else None
                                    params_j37 = apply_ellipse(r_j37, 37.5, thermal_map, True) if r_j37 > 0 else None

                                    N_fat = 0.0
                                    for _, area_row in df_area.iterrows():
                                        area_key = area_row['Area_Clean']
                                        pop = area_row['Population']
                                        if pop <= 0 or area_key not in mapped_areas_highres: continue
                                        ax, ay = mapped_areas_highres[area_key]
                                        d_sq = (ax - sx)**2 + (ay - sy)**2
                                        max_v = 0.0
                                        if r_e05 > 0 and d_sq <= (r_e05*PIXELS_PER_METER)**2: max_v = max(max_v, vuln_exp_05)
                                        elif r_e03 > 0 and d_sq <= (r_e03*PIXELS_PER_METER)**2: max_v = max(max_v, vuln_exp_03)
                                        if r_p37 > 0 and d_sq <= (r_p37*PIXELS_PER_METER)**2: max_v = max(max_v, vuln_thermal_37)
                                        elif r_p12 > 0 and d_sq <= (r_p12*PIXELS_PER_METER)**2: max_v = max(max_v, vuln_thermal_12)
                                        
                                        def in_ell(p):
                                            if not p: return False
                                            cx, cy, a, b = p
                                            xr = (ax - cx)*cos_a + (ay - cy)*sin_a
                                            yr = -(ax - cx)*sin_a + (ay - cy)*cos_a
                                            return (xr/a)**2 + (yr/b)**2 <= 1.0
                                        if in_ell(params_j37): max_v = max(max_v, vuln_thermal_37)
                                        elif in_ell(params_j12): max_v = max(max_v, vuln_thermal_12)
                                        N_fat += pop * max_v
                                    if N_fat > 0.0: fn_data.append((scenario_freq * w_prob, N_fat))

                            # 차트 생성
                            area_lsir_vals = {row['Area_Clean']: lsir_map[mapped_areas_highres[row['Area_Clean']][1], mapped_areas_highres[row['Area_Clean']][0]] if row['Area_Clean'] in mapped_areas_highres else 0.0 for _, row in df_area.iterrows()}

                            irpa_results = []
                            for _, row in df_occ.iterrows():
                                category = row['Personnel_Category']
                                if pd.isna(category): continue
                                irpa = sum(float(row[col]) * area_lsir_vals[clean_name(col)] for col in df_occ.columns[2:] if not 'Unnamed' in str(col) and pd.notna(row[col]) and row[col]>0 and clean_name(col) in area_lsir_vals)
                                irpa_results.append((category, irpa))

                            df_irpa = pd.DataFrame(irpa_results, columns=['Category', 'IRPA']).sort_values(by='IRPA', ascending=True)
                            df_irpa['IRPA_Plot'] = df_irpa['IRPA'].apply(lambda x: x if x > 1e-8 else 1e-8)

                            fig_irpa, ax_irpa = plt.subplots(figsize=(12, 8))
                            ax_irpa.barh(df_irpa['Category'], df_irpa['IRPA_Plot'], color='dodgerblue')
                            ax_irpa.axvline(x=1e-3, color='red', linestyle='--', linewidth=2, label='1E-3 (Red Limit)')
                            ax_irpa.axvline(x=1e-4, color='orange', linestyle='--', linewidth=2, label='1E-4 (Orange Limit)')
                            ax_irpa.axvline(x=1e-5, color='gold', linestyle='--', linewidth=2, label='1E-5 (Yellow Limit)')
                            ax_irpa.axvline(x=1e-6, color='blue', linestyle='--', linewidth=2, label='1E-6 (Blue Limit)')
                            ax_irpa.axvline(x=1e-7, color='green', linestyle='--', linewidth=2, label='1E-7 (Green Limit)')
                            ax_irpa.set_xscale('log'); ax_irpa.set_xlim(left=1e-8, right=1e-2)
                            for i, v in enumerate(df_irpa['IRPA']):
                                if v == 0: ax_irpa.text(1.2e-8, i, " No Risk (0.0)", color='dimgray', va='center', fontsize=10, fontweight='bold')
                            ax_irpa.set_xlabel('Individual Risk Per Annum (IRPA)', fontsize=14)
                            ax_irpa.set_title('IRPA per Worker Category', fontsize=18, fontweight='bold')
                            ax_irpa.grid(axis='x', which='both', linestyle='--', alpha=0.5)
                            ax_irpa.legend(loc='lower right', fontsize=10)
                            plt.tight_layout()
                            buf_irpa = io.BytesIO(); fig_irpa.savefig(buf_irpa, format='png', dpi=300); buf_irpa.seek(0)

                            fig_fn, ax_fn = plt.subplots(figsize=(10, 8))
                            if fn_data:
                                df_fn = pd.DataFrame(fn_data, columns=['F', 'N'])
                                df_fn = df_fn.groupby('N').sum().reset_index().sort_values(by='N', ascending=False)
                                df_fn['Cum_F'] = df_fn['F'].cumsum()
                                X_plot, Y_plot = [df_fn['N'].iloc[0]], [df_fn['Cum_F'].iloc[0]]
                                for i in range(1, len(df_fn)):
                                    X_plot.extend([df_fn['N'].iloc[i], df_fn['N'].iloc[i]])
                                    Y_plot.extend([df_fn['Cum_F'].iloc[i-1], df_fn['Cum_F'].iloc[i]])
                                max_n = max(10.0, df_fn['N'].max() * 1.2)
                                max_f = max(1e-4, df_fn['Cum_F'].max() * 1.5)
                                min_f = min(1e-5, df_fn['Cum_F'].min() / 1.5)
                                n_range = np.logspace(0, np.log10(max_n), 100) 
                                ax_fn.plot(n_range, 1e-3 / n_range, color='red', linestyle='--', linewidth=2, label='Unacceptable Limit (1E-3)')
                                ax_fn.plot(n_range, 1e-5 / n_range, color='green', linestyle='--', linewidth=2, label='Broadly Acceptable (1E-5)')
                                ax_fn.fill_between(n_range, 1e-5 / n_range, 1e-3 / n_range, color='yellow', alpha=0.15, label='ALARP Region')
                                ax_fn.plot(X_plot, Y_plot, color='blue', linewidth=3, label='Calculated Facility Risk') 
                                ax_fn.set_xscale('log'); ax_fn.set_yscale('log')
                                ax_fn.set_title('F-N Curve with ALARP Criteria', fontsize=18, fontweight='bold')
                                ax_fn.set_xlabel('Number of Fatalities (N)', fontsize=14); ax_fn.set_ylabel('Cumulative Frequency (F) > N', fontsize=14)
                                ax_fn.grid(True, which='both', ls='--', alpha=0.6)
                                ax_fn.set_xlim(left=1, right=max_n); ax_fn.set_ylim(bottom=min_f, top=max_f)
                                ax_fn.legend(loc='upper right', fontsize=12)
                            plt.tight_layout()
                            buf_fn = io.BytesIO(); fig_fn.savefig(buf_fn, format='png', dpi=300); buf_fn.seek(0)

                            def get_rendered_image(data_map, levels, title):
                                out = engine_bg_image.copy()
                                thickness = max(3, int(new_w / 800)) 
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                fs = new_w / 4500.0 
                                text_thick = max(2, int(thickness / 3))
                                
                                for limit, color, label in levels:
                                    mask = np.uint8(data_map >= limit) * 255
                                    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(out, contours, -1, color, thickness, cv2.LINE_AA)
                                    
                                max_text_w, th_max = 0, 0
                                for limit, color, text in levels:
                                    (tw, th), _ = cv2.getTextSize(text, font, fs, text_thick)
                                    max_text_w = max(max_text_w, tw)
                                    th_max = max(th_max, th)
                                    
                                (title_w, title_h), _ = cv2.getTextSize(title, font, fs * 1.2, text_thick)
                                
                                line_len = int(150 * fs)
                                padding = int(50 * fs)
                                row_h = int(100 * fs)
                                cont_leg_w = line_len + int(40*fs) + max(max_text_w, title_w)
                                cont_leg_h = int(80 * fs) + len(levels) * row_h
                                
                                r_max = int(new_w * 0.045) 
                                wr_w = int(r_max * 2.5)    
                                wr_leg_w = int(450 * fs)   
                                
                                panel_w = padding * 5 + wr_w + wr_leg_w + cont_leg_w
                                panel_h = padding * 2 + max(cont_leg_h, int(r_max * 2.8))
                                px = int(new_w * 0.02) 
                                py = new_h - panel_h - int(new_h * 0.02) 
                                
                                overlay = out.copy()
                                cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (250, 250, 250), -1)
                                cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
                                cv2.rectangle(out, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), max(1, int(thickness/2)))

                                cx = px + padding + int(wr_w / 2)
                                cy = py + int(panel_h / 2.2)
                                for scale in [0.33, 0.66, 1.0]:
                                    cv2.circle(out, (cx, cy), int(r_max * scale), (200, 200, 200), max(1, int(r_max/150)))
                                    
                                grouped = df_met.groupby(['Angle_deg', 'Wind_Speed_m_s'])['Probability'].sum().reset_index()
                                total_probs = df_met.groupby('Angle_deg')['Probability'].sum()
                                max_prob = total_probs.max() if not total_probs.empty else 1.0
                                
                                for angle in grouped['Angle_deg'].unique():
                                    subset = grouped[grouped['Angle_deg'] == angle].sort_values(by='Wind_Speed_m_s', ascending=False)
                                    cv2_angle = (angle - 90) % 360
                                    cum_prob = subset['Probability'].sum()
                                    for _, row in subset.iterrows():
                                        speed, prob = row['Wind_Speed_m_s'], row['Probability']
                                        layer_r = int(r_max * (cum_prob / max_prob))
                                        if layer_r <= 0: continue
                                        if speed <= 3.3: color = (200, 200, 200) 
                                        elif speed <= 7.9: color = (65, 105, 225) 
                                        else: color = (0, 0, 128) 
                                        cv2.ellipse(out, (cx, cy), (layer_r, layer_r), 0, cv2_angle - 15, cv2_angle + 15, color, -1)
                                        cv2.ellipse(out, (cx, cy), (layer_r, layer_r), 0, cv2_angle - 15, cv2_angle + 15, (255, 255, 255), max(1, int(r_max/150)))
                                        cum_prob -= prob
                                        
                                cv2.putText(out, "N", (cx - int(r_max*0.1), cy - int(r_max*1.15)), font, fs*1.2, (255, 0, 0), max(2, int(r_max/25)))
                                cv2.putText(out, "Wind Rose", (cx - int(r_max*0.7), cy + int(r_max*1.35)), font, fs, (0, 0, 0), text_thick)

                                wr_leg_x = px + padding * 2 + wr_w
                                wr_leg_y = py + padding + int(title_h * 1.5)
                                box_s = int(r_max * 0.25)
                                cv2.putText(out, "Wind Speed", (wr_leg_x, py + padding + title_h), font, fs * 1.2, (0,0,0), text_thick)
                                
                                legends = [("<= 3.3 m/s", (200, 200, 200)), ("3.4 ~ 7.9 m/s", (65, 105, 225)), (">= 8.0 m/s", (0, 0, 128))]
                                for text, color in legends:
                                    cv2.rectangle(out, (wr_leg_x, wr_leg_y), (wr_leg_x + box_s, wr_leg_y + box_s), color, -1)
                                    cv2.rectangle(out, (wr_leg_x, wr_leg_y), (wr_leg_x + box_s, wr_leg_y + box_s), (150, 150, 150), max(1, int(r_max/200)))
                                    cv2.putText(out, text, (wr_leg_x + box_s + int(r_max*0.2), wr_leg_y + int(box_s*0.8)), font, fs*0.9, (50, 50, 50), text_thick)
                                    wr_leg_y += int(box_s * 1.8)

                                cont_leg_x = px + padding * 4 + wr_w + wr_leg_w
                                cv2.putText(out, title, (cont_leg_x, py + padding + title_h), font, fs * 1.2, (0,0,0), text_thick)
                                sy = py + padding + title_h + int(100 * fs)
                                for limit, color, text in reversed(levels):
                                    cv2.line(out, (cont_leg_x, sy), (cont_leg_x + line_len, sy), color, thickness + 4, cv2.LINE_AA)
                                    cv2.putText(out, text, (cont_leg_x + line_len + int(40*fs), sy + int(th_max/2)), font, fs, (0,0,0), text_thick)
                                    sy += row_h

                                return out

                            lvl_uni = [(1e-7, [0, 255, 0], "1.0 E-07 (Green)"), (1e-6, [0, 0, 255], "1.0 E-06 (Blue)"), (1e-5, [255, 255, 0], "1.0 E-05 (Yellow)"), (1e-4, [255, 165, 0], "1.0 E-04 (Orange)"), (1e-3, [255, 0, 0], "1.0 E-03 (Red)")]
                            lvl_therm = [(12.0, [255, 255, 0], "12.5 kW/m2"), (37.0, [255, 0, 0], "37.5 kW/m2")]
                            lvl_exp = [(0.29, [255, 255, 0], "0.3 bar"), (0.49, [255, 0, 0], "0.5 bar")]

                            # ❗ [NEW] 계산된 기본 도면을 세션 스토리지에 영구 저장
                            st.session_state.base_lsir = get_rendered_image(lsir_map, lvl_uni, "LSIR Contour")
                            st.session_state.base_therm = get_rendered_image(thermal_map, lvl_therm, "Thermal Radiation")
                            st.session_state.base_exp = get_rendered_image(explosion_map, lvl_exp, "Explosion Overpressure")
                            st.session_state.base_irpa = get_rendered_image(lsir_map, lvl_uni, "IRPA Contour")
                            
                            st.session_state.bytes_fn = buf_fn.getvalue() if fn_data else None
                            st.session_state.bytes_irpa = buf_irpa.getvalue()
                            
                            st.session_state.engine_target_width = engine_target_width
                            st.session_state.qra_calculated = True

                        except Exception as e:
                            st.error(f"❌ 엔진 연산 중 오류가 발생했습니다.\nError: {e}")

                # ❗ [NEW] 계산이 완료된 상태라면 다운로드를 해도 화면이 초기화되지 않음
                if st.session_state.get('qra_calculated', False):
                    st.success("✅ 고해상도 QRA 연산 및 등고선 매핑이 완료되었습니다!")
                    
                    # ❗ [NEW] 결과물 마커 표시 전환 토글 버튼 (화면을 새로고침하며 오버레이 적용)
                    show_markers = st.toggle("📍 도면 위에 누출점 및 작업구역 마커 표시하기", value=False)
                    
                    def add_markers(base_img, mapped_is, mapped_areas, new_w):
                        out = base_img.copy()
                        font = cv2.FONT_HERSHEY_DUPLEX
                        fs = new_w / 4500.0
                        text_thick = max(2, int((new_w / 800) / 3))
                        
                        for is_id, (cx, cy) in mapped_is.items():
                            r = int(new_w * 0.005)
                            cv2.circle(out, (cx, cy), r, (255, 0, 0), -1) 
                            cv2.circle(out, (cx, cy), r, (255, 255, 255), max(1, int(r/4))) 
                            cv2.putText(out, f"{is_id}", (cx + int(r*1.2), cy - int(r*1.2)), font, fs*1.5, (255, 0, 0), text_thick+1, cv2.LINE_AA)
                            
                        for area_id, (cx, cy) in mapped_areas.items():
                            r = int(new_w * 0.005)
                            cv2.rectangle(out, (cx - r, cy - r), (cx + r, cy + r), (0, 0, 255), -1) 
                            cv2.rectangle(out, (cx - r, cy - r), (cx + r, cy + r), (255, 255, 255), max(1, int(r/4)))
                            cv2.putText(out, f"{area_id}", (cx + int(r*1.2), cy - int(r*1.2)), font, fs*1.5, (0, 0, 255), text_thick+1, cv2.LINE_AA)
                        return out

                    # 토글 상태에 따라 마커가 오버레이된 이미지로 교체
                    if show_markers:
                        final_lsir = add_markers(st.session_state.base_lsir, mapped_is_highres, mapped_areas_highres, st.session_state.engine_target_width)
                        final_therm = add_markers(st.session_state.base_therm, mapped_is_highres, mapped_areas_highres, st.session_state.engine_target_width)
                        final_exp = add_markers(st.session_state.base_exp, mapped_is_highres, mapped_areas_highres, st.session_state.engine_target_width)
                        final_irpa = add_markers(st.session_state.base_irpa, mapped_is_highres, mapped_areas_highres, st.session_state.engine_target_width)
                    else:
                        final_lsir = st.session_state.base_lsir
                        final_therm = st.session_state.base_therm
                        final_exp = st.session_state.base_exp
                        final_irpa = st.session_state.base_irpa

                    st.markdown("### 📊 분석 결과")
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        "🗺️ LSIR 도면", "🔥 Thermal 도면", "💥 Explosion 도면", 
                        "👤 IRPA 도면", "📉 F-N Curve", "📊 IRPA 차트"
                    ])
                    
                    def get_img_bytes(img_rgb):
                        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                        return io.BytesIO(buffer)

                    # ❗ 결과물 90% 크기 고정 및 가운데 정렬 유지
                    with tab1:
                        col_l, col_c, col_r = st.columns([0.05, 0.9, 0.05])
                        with col_c:
                            st.image(final_lsir, use_container_width=True)
                            st.download_button("⬇️ LSIR 도면 다운로드", get_img_bytes(final_lsir), "LSIR_Contour.png", "image/png")
                    with tab2:
                        col_l, col_c, col_r = st.columns([0.05, 0.9, 0.05])
                        with col_c:
                            st.image(final_therm, use_container_width=True)
                            st.download_button("⬇️ Thermal 도면 다운로드", get_img_bytes(final_therm), "Thermal_Contour.png", "image/png")
                    with tab3:
                        col_l, col_c, col_r = st.columns([0.05, 0.9, 0.05])
                        with col_c:
                            st.image(final_exp, use_container_width=True)
                            st.download_button("⬇️ Explosion 도면 다운로드", get_img_bytes(final_exp), "Explosion_Contour.png", "image/png")
                    with tab4:
                        col_l, col_c, col_r = st.columns([0.05, 0.9, 0.05])
                        with col_c:
                            st.image(final_irpa, use_container_width=True)
                            st.download_button("⬇️ IRPA 도면 다운로드", get_img_bytes(final_irpa), "IRPA_Contour.png", "image/png")
                    with tab5:
                        col_l, col_c, col_r = st.columns([0.05, 0.9, 0.05])
                        with col_c:
                            if st.session_state.bytes_fn:
                                st.image(st.session_state.bytes_fn, use_container_width=True)
                                st.download_button("⬇️ F-N Curve 다운로드", st.session_state.bytes_fn, "FN_Curve.png", "image/png")
                            else:
                                st.info("조건에 부합하는 사상자(N) 발생 시나리오가 없어 FN Curve가 생성되지 않았습니다.")
                    with tab6:
                        col_l, col_c, col_r = st.columns([0.05, 0.9, 0.05])
                        with col_c:
                            st.image(st.session_state.bytes_irpa, use_container_width=True)
                            st.download_button("⬇️ IRPA 막대그래프 다운로드", st.session_state.bytes_irpa, "IRPA_Chart.png", "image/png")

elif uploaded_image is None or uploaded_excel is None:
    st.info("좌측 패널에서 도면과 엑셀 마스터 파일을 업로드하여 주십시오.")