import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image

# 💡 [NEW] 말썽 많던 Canvas를 버리고, 빠르고 안정적인 이미지 클릭 좌표 라이브러리 도입
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CSS를 통한 툴박스 글씨 크기 확대 ---
st.markdown("""
<style>
div[role="radiogroup"] > label > div > p {
    font-size: 1.15rem !important; 
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# --- 웹 앱 기본 설정 ---
st.set_page_config(page_title="QRA 대화형 매핑 및 자동화 대시보드", layout="wide")
st.title("🔥 QRA Interactive Mapping & Analysis Dashboard")
st.markdown("도면 상의 누출점 및 작업구역을 클릭하여 지정하고, QRA 분석 엔진을 통해 위험도 등고선을 렌더링합니다.")

# --- 사이드바: 입력 패널 ---
st.sidebar.header("📁 Data Input")
uploaded_image = st.sidebar.file_uploader("1. 도면 이미지 업로드 (.png, .jpg)", type=['png', 'jpg', 'jpeg'])
real_width_m = st.sidebar.number_input("2. 도면 실제 가로 길이 (m)", min_value=1.0, value=1000.0, step=10.0)
uploaded_excel = st.sidebar.file_uploader("3. 마스터 엑셀 업로드 (.xlsx)", type=['xlsx'])

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ 리스크 허용 기준 (F-N Curve)")
limit_unacc = st.sidebar.select_slider("허용 불가 기준 (Unacceptable)", options=[1e-2, 1e-3, 1e-4], value=1e-3)
limit_acc = st.sidebar.select_slider("허용 가능 기준 (Acceptable)", options=[1e-4, 1e-5, 1e-6, 1e-7], value=1e-5)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 그래프 축 범위 설정")
max_n_axis = st.sidebar.number_input("F-N Curve 사상자 수(N) 최대값", min_value=10, max_value=10000, value=100, step=10)

def clean_name(n):
    return str(n).strip().lower().replace('_', ' ')

# --- 상태 관리 (세션 초기화) ---
if 'qra_calculated' not in st.session_state:
    st.session_state.qra_calculated = False
if 'markers' not in st.session_state:
    st.session_state.markers = {'points': [], 'areas': []}
if 'last_click' not in st.session_state:
    st.session_state.last_click = None

# --- 데이터 캐싱 ---
@st.cache_data
def load_excel_cached(file_buffer):
    xls = pd.ExcelFile(file_buffer)
    df_is = pd.read_excel(xls, sheet_name='IS_Coordinates')
    df_phast = pd.read_excel(xls, sheet_name='PHAST_Distances')
    wind_sheet = 'Wind_rose' if 'Wind_rose' in xls.sheet_names else 'Wind rose(Meteorology)'
    df_met = pd.read_excel(xls, sheet_name=wind_sheet)
    df_occ = pd.read_excel(xls, sheet_name='Occupancy')
    df_vuln = pd.read_excel(xls, sheet_name='Vulnerability_Criteria')
    names = [str(col) for col in df_occ.columns[2:] if 'Unnamed' not in str(col)]
    df_area = pd.DataFrame({'Area_Name': names})
    return df_is, df_phast, df_met, df_occ, df_vuln, df_area

@st.cache_data
def process_image_cached(image_bytes):
    img_orig = Image.open(io.BytesIO(image_bytes)).convert("RGBA") 
    white_bg = Image.new("RGBA", img_orig.size, (255, 255, 255, 255)) 
    white_bg.alpha_composite(img_orig) 
    bg_img = white_bg.convert("RGB") 
    t_width = 800
    orig_w, orig_h = bg_img.size
    s_ratio = t_width / orig_w
    c_width = t_width
    c_height = int(orig_h * s_ratio)
    bg_resized = bg_img.resize((c_width, c_height), Image.Resampling.LANCZOS)
    return bg_img, bg_resized, c_width, c_height, orig_w, orig_h

# --- 메인 화면: 새로운 Click & Pin 매핑 시스템 ---
if uploaded_image is not None and uploaded_excel is not None:
    try:
        df_is, df_phast, df_met, df_occ, df_vuln, df_area = load_excel_cached(uploaded_excel)
        is_id_list = df_is['IS_ID'].dropna().astype(str).unique().tolist()
        area_name_list = df_area['Area_Name'].dropna().astype(str).unique().tolist()
    except Exception as e:
        st.error(f"엑셀 데이터 로드 실패. 시트 및 데이터를 확인하십시오.\nError: {e}")
        st.stop()

    bg_image, bg_image_resized, canvas_width, canvas_height, original_width, original_height = process_image_cached(uploaded_image.getvalue())
    
    st.markdown("---")
    col_tool, col_map = st.columns([1, 2.5])
    
    with col_tool:
        st.subheader("🛠️ Step 1. Click & Pin")
        st.info("우측 도면 위를 마우스로 직접 **클릭**하여 핀을 꽂아주세요!")
        mapping_mode = st.radio("마커 종류 선택", ("📍 누출점 (Point)", "🟦 작업구역 (Area)"))
        
        st.markdown("---")
        if st.button("↩️ 마지막 핀 지우기", use_container_width=True):
            if mapping_mode == "📍 누출점 (Point)" and st.session_state.markers['points']:
                st.session_state.markers['points'].pop()
            elif mapping_mode == "🟦 작업구역 (Area)" and st.session_state.markers['areas']:
                st.session_state.markers['areas'].pop()
            st.session_state.last_click = None # 초기화
            st.rerun()
            
        if st.button("🗑️ 전체 핀 지우기", type="secondary", use_container_width=True):
            st.session_state.markers = {'points': [], 'areas': []}
            st.session_state.last_click = None
            st.rerun()

    with col_map:
        # 도면 복사본에 마커들을 직접 그림 (절대 사라질 수 없음)
        display_img = np.array(bg_image_resized).copy()
        
        for i, (x, y) in enumerate(st.session_state.markers['points']):
            cv2.circle(display_img, (x, y), 6, (255, 0, 0), -1)
            cv2.circle(display_img, (x, y), 6, (255, 255, 255), 2)
            cv2.putText(display_img, f"P{i+1}", (x+10, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), 2)
            
        for i, (x, y) in enumerate(st.session_state.markers['areas']):
            cv2.rectangle(display_img, (x-8, y-8), (x+8, y+8), (0, 0, 255), -1)
            cv2.rectangle(display_img, (x-8, y-8), (x+8, y+8), (255, 255, 255), 2)
            cv2.putText(display_img, f"A{i+1}", (x+10, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)

        # 클릭 이벤트 수신
        click_val = streamlit_image_coordinates(Image.fromarray(display_img), key="interactive_map")
        
        if click_val is not None and click_val != st.session_state.last_click:
            st.session_state.last_click = click_val
            cx, cy = click_val['x'], click_val['y']
            if mapping_mode == "📍 누출점 (Point)":
                st.session_state.markers['points'].append((cx, cy))
            else:
                st.session_state.markers['areas'].append((cx, cy))
            st.rerun()

    mapped_is_data = {}
    mapped_area_data = {}
    has_duplicate = False 

    if st.session_state.markers['points'] or st.session_state.markers['areas']:
        st.markdown("---")
        st.subheader("📝 Step 2. Data Mapping")
        col_pt, col_rect = st.columns(2)
        
        with col_pt:
            st.markdown("##### 📍 Points (누출점)")
            pts = st.session_state.markers['points']
            if not pts: st.info("도면에 누출점이 없습니다.")
            for i, (cx, cy) in enumerate(pts):
                default_idx = i if i < len(is_id_list) else 0
                user_name = st.selectbox(f"Point {i+1} 마핑", options=is_id_list, index=default_idx, key=f"pt_{i}")
                if user_name in mapped_is_data: st.error(f"'{user_name}' 중복!"); has_duplicate = True
                else: mapped_is_data[user_name] = (cx, cy)
        
        with col_rect:
            st.markdown("##### 🟦 Areas (작업구역)")
            ars = st.session_state.markers['areas']
            if not ars: st.info("도면에 작업구역이 없습니다.")
            for i, (cx, cy) in enumerate(ars):
                default_idx = i if i < len(area_name_list) else 0
                user_name = st.selectbox(f"Area {i+1} 마핑", options=area_name_list, index=default_idx, key=f"rect_{i}")
                if user_name in mapped_area_data: st.error(f"'{user_name}' 중복!"); has_duplicate = True
                else: mapped_area_data[str(user_name)] = (cx, cy)

        engine_target_width = 2000 
        coord_scale = engine_target_width / canvas_width
        mapped_is_highres = {k: (int(v[0]*coord_scale), int(v[1]*coord_scale)) for k, v in mapped_is_data.items()}
        mapped_areas_highres = {clean_name(k): (int(v[0]*coord_scale), int(v[1]*coord_scale)) for k, v in mapped_area_data.items()}

        st.write("---")
        if has_duplicate:
            st.warning("⚠️ 중복 선택된 항목을 수정해야 분석을 실행할 수 있습니다.")
        else:
            st.markdown("### 🚀 Step 3. High-Res QRA Analysis Engine")
            if st.button("QRA 물리 모델 초정밀 연산 및 렌더링 실행", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="초기화 및 데이터 로드 중...")
                try:
                    progress_bar.progress(10, text="1/5: 마스터 엑셀 파라미터 매핑 중...")
                    try:
                        vuln_vals = df_vuln['Outdoor_Vulnerability'].tolist()
                        vuln_thermal_12, vuln_thermal_37 = float(vuln_vals[0]), float(vuln_vals[1])
                        vuln_flash = float(vuln_vals[2])
                        vuln_exp_03, vuln_exp_05 = float(vuln_vals[3]), float(vuln_vals[4])
                    except:
                        vuln_thermal_12, vuln_thermal_37 = 0.0, 1.0; vuln_flash = 1.0; vuln_exp_03, vuln_exp_05 = 0.2, 0.5

                    df_occ_calc = df_occ[~df_occ['Personnel_Category'].astype(str).str.lower().str.contains('building', na=False)].copy()
                    df_occ_calc['No_of_Personnel'] = pd.to_numeric(df_occ_calc['No_of_Personnel'], errors='coerce').fillna(0)
                    area_population = {}
                    for col in df_occ_calc.columns[2:]:
                        if 'Unnamed' in str(col): continue
                        clean_col = clean_name(col)
                        col_data = pd.to_numeric(df_occ_calc[col], errors='coerce').fillna(0)
                        area_population[clean_col] = (df_occ_calc['No_of_Personnel'] * col_data).sum()

                    df_area['Area_Clean'] = df_area['Area_Name'].apply(clean_name)
                    df_area['Population'] = df_area['Area_Clean'].map(area_population).fillna(0.0)

                    engine_scale_ratio = engine_target_width / original_width
                    engine_h = int(original_height * engine_scale_ratio)
                    engine_bg_pil = bg_image.resize((engine_target_width, engine_h), Image.Resampling.LANCZOS)
                    engine_bg_image = np.array(engine_bg_pil) 
                    
                    PIXELS_PER_METER = engine_target_width / real_width_m
                    new_w, new_h = engine_target_width, engine_h

                    lsir_map = np.zeros((new_h, new_w), dtype=np.float32)
                    old_lsir_map = np.zeros((new_h, new_w), dtype=np.float32)
                    thermal_map = np.zeros((new_h, new_w), dtype=np.float32)
                    explosion_map = np.zeros((new_h, new_w), dtype=np.float32)
                    y_idx, x_idx = np.indices((new_h, new_w), dtype=np.float32)
                    fn_data = []

                    def parse_val(val):
                        if pd.isna(val) or str(val).strip() in ['NR', 'NA', '']: return 0.0
                        try: return float(val)
                        except: return 0.0

                    progress_bar.progress(30, text="2/5: 열복사(Thermal) 및 폭발(Explosion) 확산 모델 연산 중...")
                    
                    for _, row in df_phast.iterrows():
                        is_id = str(row['IS_ID'])
                        if is_id not in mapped_is_highres: continue
                        freq_col = f"Freq_{float(row['Leak_Size_mm']):g}mm"
                        freq_info = df_is[df_is['IS_ID'] == is_id]
                        if freq_info.empty or freq_col not in freq_info.columns: continue
                        scenario_freq = float(freq_info[freq_col].values[0])
                        if scenario_freq == 0: continue

                        sx, sy = mapped_is_highres[is_id]
                        r_p12, r_p37 = parse_val(row.get('Pool_12.5')), parse_val(row.get('Pool_37.5'))
                        r_e03, r_e05 = parse_val(row.get('Exp_0.3')), parse_val(row.get('Exp_0.5'))
                        r_j12, r_j37 = parse_val(row.get('Jet_12.5')), parse_val(row.get('Jet_37.5'))
                        r_flash = parse_val(row.get('Flash_LFL'))
                        
                        p_jet = parse_val(row.get('P_jet'))
                        p_pool = parse_val(row.get('P_pool'))
                        p_flash = parse_val(row.get('P_flash'))
                        p_vce = parse_val(row.get('P_vce'))

                        total_w_prob = float(df_met['Probability'].sum()) if not df_met.empty else 0.0
                        freq_pool = scenario_freq * total_w_prob * p_pool
                        freq_vce = scenario_freq * total_w_prob * p_vce

                        def apply_circle(r_m, val, target_map, is_max=False):
                            if r_m <= 0: return
                            r_px = r_m * PIXELS_PER_METER
                            x_min, x_max = max(0, int(sx - r_px - 2)), min(new_w, int(sx + r_px + 2))
                            y_min, y_max = max(0, int(sy - r_px - 2)), min(new_h, int(sy + r_px + 2))
                            if x_min >= x_max or y_min >= y_max: return
                            mask = np.sqrt((x_idx[y_min:y_max, x_min:x_max] - sx)**2 + (y_idx[y_min:y_max, x_min:x_max] - sy)**2) <= r_px
                            if is_max: target_map[y_min:y_max, x_min:x_max][mask] = np.maximum(target_map[y_min:y_max, x_min:x_max][mask], val)
                            else: target_map[y_min:y_max, x_min:x_max][mask] += val

                        apply_circle(r_p12, freq_pool * vuln_thermal_12, lsir_map)
                        apply_circle(r_p37, freq_pool * (vuln_thermal_37 - vuln_thermal_12), lsir_map)
                        apply_circle(r_e03, freq_vce * vuln_exp_03, lsir_map)
                        apply_circle(r_e05, freq_vce * (vuln_exp_05 - vuln_exp_03), lsir_map)
                        
                        apply_circle(r_p12, scenario_freq * vuln_thermal_12, old_lsir_map)
                        apply_circle(r_p37, scenario_freq * (vuln_thermal_37 - vuln_thermal_12), old_lsir_map)
                        apply_circle(r_e03, scenario_freq * vuln_exp_03, old_lsir_map)
                        apply_circle(r_e05, scenario_freq * (vuln_exp_05 - vuln_exp_03), old_lsir_map)

                        if r_p12 > 0: apply_circle(r_p12, 12.5, thermal_map, True)
                        if r_p37 > 0: apply_circle(r_p37, 37.5, thermal_map, True)
                        if r_e03 > 0: apply_circle(r_e03, 0.3, explosion_map, True)
                        if r_e05 > 0: apply_circle(r_e05, 0.5, explosion_map, True)

                        met_data = df_met[df_met['Weather_Class'].astype(str).str.strip() == str(row['Weather_Class']).strip()]
                        for _, m_row in met_data.iterrows():
                            w_prob = float(m_row['Probability'])
                            freq_jet_dir = scenario_freq * w_prob * p_jet
                            freq_flash_dir = scenario_freq * w_prob * p_flash
                            
                            rad = np.radians((float(m_row['Angle_degree']) + 180) % 360)
                            math_rad = np.radians((90 - ((float(m_row['Angle_degree']) + 180) % 360)) % 360)
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
                                else: target_map[y_min:y_max, x_min:x_max][mask] += val
                                return (cx, cy, a, b)

                            apply_ellipse(r_j12, freq_jet_dir * vuln_thermal_12, lsir_map)
                            apply_ellipse(r_j37, freq_jet_dir * (vuln_thermal_37 - vuln_thermal_12), lsir_map)
                            apply_ellipse(r_flash, freq_flash_dir * vuln_flash, lsir_map)
                            
                            apply_ellipse(r_j12, (scenario_freq * w_prob * vuln_thermal_12), old_lsir_map)
                            apply_ellipse(r_j37, (scenario_freq * w_prob * (vuln_thermal_37 - vuln_thermal_12)), old_lsir_map)
                            apply_ellipse(r_flash, (scenario_freq * w_prob * vuln_flash), old_lsir_map)

                            params_j12 = apply_ellipse(r_j12, 12.5, thermal_map, True) if r_j12 > 0 else None
                            params_j37 = apply_ellipse(r_j37, 37.5, thermal_map, True) if r_j37 > 0 else None

                            N_fat = 0.0
                            for _, area_row in df_area.iterrows():
                                area_key = area_row['Area_Clean']
                                pop = area_row['Population']
                                if pop <= 0 or area_key not in mapped_areas_highres: continue
                                
                                b_type = 'Outdoor'
                                try:
                                    b_row = df_occ[df_occ['Personnel_Category'].astype(str).str.lower().str.contains('building', na=False)]
                                    if not b_row.empty:
                                        b_type = str(b_row[area_row['Area_Name']].values[0]).strip()
                                except: pass

                                if "RC" in b_type or "내폭" in b_type: v_e03, v_e05 = 0.05, 0.20
                                elif "PEB" in b_type or "경량" in b_type: v_e03, v_e05 = 0.40, 0.90
                                else: v_e03, v_e05 = vuln_exp_03, vuln_exp_05

                                ax, ay = mapped_areas_highres[area_key]
                                d_sq = (ax - sx)**2 + (ay - sy)**2
                                max_v = 0.0
                                if r_e05 > 0 and d_sq <= (r_e05*PIXELS_PER_METER)**2: max_v = max(max_v, v_e05)
                                elif r_e03 > 0 and d_sq <= (r_e03*PIXELS_PER_METER)**2: max_v = max(max_v, v_e03)
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
                            
                            if N_fat > 0.0: 
                                total_ign_freq = scenario_freq * w_prob * (p_jet + p_pool + p_flash + p_vce)
                                fn_data.append((total_ign_freq, N_fat))

                    progress_bar.progress(60, text="3/5: 인구 밀집도(IRPA) 기초 데이터 세팅 중...")
                    area_lsir_vals = {row['Area_Clean']: lsir_map[mapped_areas_highres[row['Area_Clean']][1], mapped_areas_highres[row['Area_Clean']][0]] if row['Area_Clean'] in mapped_areas_highres else 0.0 for _, row in df_area.iterrows()}

                    irpa_results = []
                    for _, row in df_occ_calc.iterrows():
                        category = row['Personnel_Category']
                        if pd.isna(category): continue
                        irpa = sum(float(row[col]) * area_lsir_vals[clean_name(col)] for col in df_occ_calc.columns[2:] if not 'Unnamed' in str(col) and pd.notna(row[col]) and row[col]>0 and clean_name(col) in area_lsir_vals)
                        irpa_results.append((category, irpa))

                    df_irpa = pd.DataFrame(irpa_results, columns=['Category', 'IRPA']).sort_values(by='IRPA', ascending=True)
                    df_irpa['IRPA_Plot'] = df_irpa['IRPA'].apply(lambda x: x if x > 1e-8 else 1e-8)
                    
                    st.session_state.df_irpa = df_irpa
                    st.session_state.fn_data = fn_data
                    st.session_state.engine_target_width = engine_target_width

                    progress_bar.progress(80, text="4/5: 초고해상도 등고선 및 정보 패널 렌더링 중...")
                    def get_rendered_image(data_map, levels, title):
                        out = engine_bg_image.copy()
                        thickness = max(3, int(new_w / 800)); font = cv2.FONT_HERSHEY_SIMPLEX; fs = new_w / 4500.0; text_thick = max(2, int(thickness / 3))
                        for limit, color, label in levels:
                            mask = np.uint8(data_map >= limit) * 255
                            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(out, contours, -1, color, thickness, cv2.LINE_AA)
                            
                        max_text_w, th_max = 0, 0
                        for limit, color, text in levels:
                            (tw, th), _ = cv2.getTextSize(text, font, fs, text_thick)
                            max_text_w = max(max_text_w, tw); th_max = max(th_max, th)
                            
                        (title_w, title_h), _ = cv2.getTextSize(title, font, fs * 1.2, text_thick)
                        line_len = int(150 * fs); padding = int(50 * fs); row_h = int(100 * fs)
                        cont_leg_w = line_len + int(40*fs) + max(max_text_w, title_w)
                        cont_leg_h = int(80 * fs) + len(levels) * row_h
                        r_max = int(new_w * 0.045); wr_w = int(r_max * 2.5); wr_leg_w = int(450 * fs)   
                        panel_w = padding * 5 + wr_w + wr_leg_w + cont_leg_w
                        panel_h = padding * 2 + max(cont_leg_h, int(r_max * 2.8))
                        px = int(new_w * 0.02); py = new_h - panel_h - int(new_h * 0.02) 
                        
                        overlay = out.copy()
                        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (250, 250, 250), -1)
                        cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
                        cv2.rectangle(out, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), max(1, int(thickness/2)))

                        cx = px + padding + int(wr_w / 2); cy = py + int(panel_h / 2.2)
                        for scale in [0.33, 0.66, 1.0]:
                            cv2.circle(out, (cx, cy), int(r_max * scale), (200, 200, 200), max(1, int(r_max/150)))
                            
                        grouped = df_met.groupby(['Angle_degree', 'Wind_Speed_(m/s)'])['Probability'].sum().reset_index()
                        total_probs = df_met.groupby('Angle_degree')['Probability'].sum()
                        max_prob = total_probs.max() if not total_probs.empty else 1.0
                        for angle in grouped['Angle_degree'].unique():
                            subset = grouped[grouped['Angle_degree'] == angle].sort_values(by='Wind_Speed_(m/s)', ascending=False)
                            cv2_angle = (angle - 90) % 360
                            cum_prob = subset['Probability'].sum()
                            for _, row in subset.iterrows():
                                speed, prob = row['Wind_Speed_(m/s)'], row['Probability']
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

                    st.session_state.base_lsir = get_rendered_image(lsir_map, lvl_uni, "LSIR Contour")
                    st.session_state.base_therm = get_rendered_image(thermal_map, lvl_therm, "Thermal Radiation")
                    st.session_state.base_exp = get_rendered_image(explosion_map, lvl_exp, "Explosion Overpressure")
                    st.session_state.base_comb = get_rendered_image(old_lsir_map, lvl_uni, "Combined Contour (Conservative)")
                    st.session_state.base_irpa = get_rendered_image(lsir_map, lvl_uni, "IRPA Contour")
                    
                    st.session_state.qra_calculated = True
                    progress_bar.empty() 

                except Exception as e:
                    st.error(f"❌ 엔진 연산 중 오류가 발생했습니다.\nError: {e}")

            if st.session_state.get('qra_calculated', False):
                
                df_irpa = st.session_state.df_irpa
                fn_data = st.session_state.fn_data
                
                fig_irpa, ax_irpa = plt.subplots(figsize=(12, 8))
                ax_irpa.barh(df_irpa['Category'], df_irpa['IRPA_Plot'], color='dodgerblue')
                ax_irpa.axvline(x=limit_unacc, color='red', linestyle='--', linewidth=2, label=f'Unacceptable ({limit_unacc:.0E})')
                ax_irpa.axvline(x=limit_acc, color='green', linestyle='--', linewidth=2, label=f'Acceptable ({limit_acc:.0E})')
                ax_irpa.set_xscale('log'); ax_irpa.set_xlim(left=1e-8, right=1e-2)
                for i, v in enumerate(df_irpa['IRPA']):
                    if v == 0: ax_irpa.text(1.2e-8, i, " No Risk (0.0)", color='dimgray', va='center', fontsize=10, fontweight='bold')
                ax_irpa.set_xlabel('Individual Risk Per Annum (IRPA)', fontsize=14)
                ax_irpa.set_title('IRPA per Worker Category', fontsize=18, fontweight='bold')
                ax_irpa.grid(axis='x', which='both', linestyle='--', alpha=0.5)
                ax_irpa.legend(loc='lower right', fontsize=10)
                plt.tight_layout()
                buf_irpa = io.BytesIO(); fig_irpa.savefig(buf_irpa, format='png', dpi=300); buf_irpa.seek(0)
                bytes_irpa = buf_irpa.getvalue()

                fig_fn, ax_fn = plt.subplots(figsize=(10, 8))
                max_n = max_n_axis 
                n_range = np.logspace(0, np.log10(max_n), 100) 
                
                ax_fn.plot(n_range, limit_unacc / n_range, color='red', linestyle='--', linewidth=2, label=f'Unacceptable ({limit_unacc:.0E})')
                ax_fn.plot(n_range, limit_acc / n_range, color='green', linestyle='--', linewidth=2, label=f'Broadly Acceptable ({limit_acc:.0E})')
                ax_fn.fill_between(n_range, limit_acc / n_range, limit_unacc / n_range, color='yellow', alpha=0.15, label='ALARP Region')

                df_fn = pd.DataFrame()
                min_y = 1e-5 
                
                if fn_data:
                    df_fn = pd.DataFrame(fn_data, columns=['F', 'N'])
                    df_fn = df_fn.groupby('N').sum().reset_index().sort_values(by='N', ascending=False)
                    df_fn['Cum_F'] = df_fn['F'].cumsum()
                    X_plot, Y_plot = [df_fn['N'].iloc[0]], [df_fn['Cum_F'].iloc[0]]
                    for i in range(1, len(df_fn)):
                        X_plot.extend([df_fn['N'].iloc[i], df_fn['N'].iloc[i]])
                        Y_plot.extend([df_fn['Cum_F'].iloc[i-1], df_fn['Cum_F'].iloc[i]])
                    
                    ax_fn.plot(X_plot, Y_plot, color='blue', linewidth=3, label='Calculated Facility Risk') 
                    min_y = min(1e-5, df_fn['Cum_F'].min() / 5)

                ax_fn.set_xscale('log'); ax_fn.set_yscale('log')
                ax_fn.set_title('F-N Curve with ALARP Criteria', fontsize=18, fontweight='bold')
                ax_fn.set_xlabel('Number of Fatalities (N)', fontsize=14); ax_fn.set_ylabel('Cumulative Frequency (F) > N', fontsize=14)
                ax_fn.grid(True, which='both', ls='--', alpha=0.6)
                
                ax_fn.set_xlim(left=1, right=max_n)
                ax_fn.set_ylim(bottom=1e-5, top=1e-1)
                ax_fn.legend(loc='upper right', fontsize=12)
                
                plt.tight_layout()
                buf_fn = io.BytesIO(); fig_fn.savefig(buf_fn, format='png', dpi=300); buf_fn.seek(0)
                bytes_fn = buf_fn.getvalue()

                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    df_fn_out = df_fn[['N', 'Cum_F']] if not df_fn.empty else pd.DataFrame(columns=['N', 'Cum_F'])
                    df_fn_out.to_excel(writer, sheet_name='FN_Curve', index=False)
                    ws_fn = writer.sheets['FN_Curve']
                    ws_fn.insert_image('D2', 'fn.png', {'image_data': io.BytesIO(bytes_fn), 'x_scale': 0.8, 'y_scale': 0.8})
                        
                    df_irpa[['Category', 'IRPA']].to_excel(writer, sheet_name='IRPA_Chart', index=False)
                    ws_irpa = writer.sheets['IRPA_Chart']
                    ws_irpa.insert_image('D2', 'irpa.png', {'image_data': io.BytesIO(bytes_irpa), 'x_scale': 0.8, 'y_scale': 0.8})
                excel_data = excel_buffer.getvalue()

                images_to_pdf = []
                for img_array in [st.session_state.base_lsir, st.session_state.base_therm, st.session_state.base_exp, st.session_state.base_comb, st.session_state.base_irpa]:
                    images_to_pdf.append(Image.fromarray(img_array))
                images_to_pdf.append(Image.open(io.BytesIO(bytes_fn)).convert("RGB"))
                images_to_pdf.append(Image.open(io.BytesIO(bytes_irpa)).convert("RGB"))
                
                pdf_bytes = io.BytesIO()
                images_to_pdf[0].save(pdf_bytes, format="PDF", resolution=100.0, save_all=True, append_images=images_to_pdf[1:])
                pdf_report = pdf_bytes.getvalue()

                st.success("✅ 고해상도 QRA 연산 및 리포트 생성이 완료되었습니다!")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.download_button(
                        label="📄 7종 통합 분석 리포트 다운로드 (PDF)",
                        data=pdf_report,
                        file_name="QRA_Integrated_Report.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                with col_btn2:
                    st.download_button(
                        label="📊 F-N 및 IRPA 그래프 보고서용 다운로드 (Excel)",
                        data=excel_data,
                        file_name="QRA_Presentation_Charts.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                st.markdown("#### 🎯 도면 마커 표시 옵션")
                col_t1, col_t2 = st.columns(2)
                with col_t1: show_points = st.toggle("📍 누출점(Point) 마커 켜기", value=False)
                with col_t2: show_areas = st.toggle("🟦 작업구역(Area) 마커 켜기", value=False)
                
                def add_markers(base_img, mapped_is, mapped_areas, new_w, draw_pts, draw_areas):
                    out = base_img.copy()
                    font = cv2.FONT_HERSHEY_DUPLEX; fs = new_w / 4500.0; text_thick = max(2, int((new_w / 800) / 3))
                    if draw_pts:
                        for is_id, (cx, cy) in mapped_is.items():
                            r = int(new_w * 0.005)
                            cv2.circle(out, (cx, cy), r, (255, 0, 0), -1); cv2.circle(out, (cx, cy), r, (255, 255, 255), max(1, int(r/4))) 
                            cv2.putText(out, f"{is_id}", (cx + int(r*1.2), cy - int(r*1.2)), font, fs*1.5, (255, 0, 0), text_thick+1, cv2.LINE_AA)
                    if draw_areas:
                        for area_id, (cx, cy) in mapped_areas.items():
                            r = int(new_w * 0.005)
                            cv2.rectangle(out, (cx - r, cy - r), (cx + r, cy + r), (0, 0, 255), -1); cv2.rectangle(out, (cx - r, cy - r), (cx + r, cy + r), (255, 255, 255), max(1, int(r/4)))
                            cv2.putText(out, f"{area_id}", (cx + int(r*1.2), cy - int(r*1.2)), font, fs*1.5, (0, 0, 255), text_thick+1, cv2.LINE_AA)
                    return out

                if show_points or show_areas:
                    final_lsir = add_markers(st.session_state.base_lsir, mapped_is_highres, mapped_areas_highres, st.session_state.engine_target_width, show_points, show_areas)
                    final_therm = add_markers(st.session_state.base_therm, mapped_is_highres, mapped_areas_highres, st.session_state.engine_target_width, show_points, show_areas)
                    final_exp = add_markers(st.session_state.base_exp, mapped_is_highres, mapped_areas_highres, st.session_state.engine_target_width, show_points, show_areas)
                    final_comb = add_markers(st.session_state.base_comb, mapped_is_highres, mapped_areas_highres, st.session_state.engine_target_width, show_points, show_areas)
                    final_irpa = add_markers(st.session_state.base_irpa, mapped_is_highres, mapped_areas_highres, st.session_state.engine_target_width, show_points, show_areas)
                else:
                    final_lsir, final_therm, final_exp, final_comb, final_irpa = st.session_state.base_lsir, st.session_state.base_therm, st.session_state.base_exp, st.session_state.base_comb, st.session_state.base_irpa

                st.markdown("### 📊 개별 분석 결과")
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "🗺️ LSIR 도면", "🔥 Thermal 도면", "💥 Explosion 도면", "🌪️ Combined (Old) 도면", 
                    "👤 IRPA 도면", "📉 F-N Curve", "📊 IRPA 차트"
                ])
                
                def get_img_bytes(img_rgb):
                    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                    return io.BytesIO(buffer)

                layout_ratio = [0.22, 0.56, 0.22]

                with tab1:
                    col_l, col_c, col_r = st.columns(layout_ratio)
                    with col_c: st.image(final_lsir, use_container_width=True); st.download_button("⬇️ 개별 다운로드", get_img_bytes(final_lsir), "LSIR_Contour.png", "image/png")
                with tab2:
                    col_l, col_c, col_r = st.columns(layout_ratio)
                    with col_c: st.image(final_therm, use_container_width=True); st.download_button("⬇️ 개별 다운로드", get_img_bytes(final_therm), "Thermal_Contour.png", "image/png")
                with tab3:
                    col_l, col_c, col_r = st.columns(layout_ratio)
                    with col_c: st.image(final_exp, use_container_width=True); st.download_button("⬇️ 개별 다운로드", get_img_bytes(final_exp), "Explosion_Contour.png", "image/png")
                with tab4:
                    col_l, col_c, col_r = st.columns(layout_ratio)
                    with col_c: st.image(final_comb, use_container_width=True); st.download_button("⬇️ 개별 다운로드", get_img_bytes(final_comb), "Combined_Contour_Conservative.png", "image/png")
                with tab5:
                    col_l, col_c, col_r = st.columns(layout_ratio)
                    with col_c: st.image(final_irpa, use_container_width=True); st.download_button("⬇️ 개별 다운로드", get_img_bytes(final_irpa), "IRPA_Contour.png", "image/png")
                with tab6:
                    col_l, col_c, col_r = st.columns(layout_ratio)
                    with col_c:
                        if bytes_fn: st.image(bytes_fn, use_container_width=True); st.download_button("⬇️ 개별 다운로드", bytes_fn, "FN_Curve.png", "image/png")
                        else: st.info("조건에 부합하는 사상자(N) 발생 시나리오가 없어 FN Curve가 생성되지 않았습니다.")
                with tab7:
                    col_l, col_c, col_r = st.columns(layout_ratio)
                    with col_c: st.image(bytes_irpa, use_container_width=True); st.download_button("⬇️ 개별 다운로드", bytes_irpa, "IRPA_Chart.png", "image/png")

elif uploaded_image is None or uploaded_excel is None:
    st.info("좌측 패널에서 도면과 엑셀 마스터 파일을 업로드하여 주십시오.")
