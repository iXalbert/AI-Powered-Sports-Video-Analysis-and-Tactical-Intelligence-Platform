import json
import time
import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"

SPORTS_LIST = [
    "Fotbal (Football)",
    "Baschet (Basketball)",
    "Handbal (Handball)",
    "Volei (Volleyball)",
    "Tenis (Tennis)",
    "Rugby (Rugby)",
    "Hochei pe gheață (Ice Hockey)",
    "Tenis de masă / Badminton"
]

COLOR_OPTIONS = ["Auto-detect", "Alb", "Negru", "Roșu", "Albastru", "Galben", "Verde", "Portocaliu", "Mov"]

st.set_page_config(
    page_title="Tactical AI",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Tactical AI">
    <style>
        .stButton > button {
            min-height: 46px;
            border-radius: 10px;
            font-weight: 600;
        }
        video {
            width: 100% !important;
            border-radius: 12px;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1.2rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


def render_backend_video(relative_url: str):
    if not relative_url:
        return
    try:
        r = requests.get(f"{API_URL}{relative_url}")
        if r.status_code == 200:
            st.video(r.content)
    except Exception as e:
        st.error(f"Nu s-a putut încărca videoclipul: {e}")


def render_backend_image(relative_url: str, caption: str = None):
    if not relative_url:
        return
    try:
        r = requests.get(f"{API_URL}{relative_url}")
        if r.status_code == 200:
            st.image(r.content, caption=caption, use_container_width=True)
    except Exception as e:
        st.error(f"Nu s-a putut încărca imaginea: {e}")


st.title("🏆 AI Sports Video Analysis & Tactical Intelligence")

with st.sidebar:
    st.header("🗂️ Istoric Meciuri Analizate")
    try:
        saved_sessions = requests.get(f"{API_URL}/api/sessions", timeout=2).json()
    except Exception:
        saved_sessions = []

    if saved_sessions:
        session_map = {s["label"]: s["session_id"] for s in saved_sessions}
        chosen_label = st.selectbox("Încarcă o analiză salvată:", list(session_map.keys()))
        if st.button("📂 Deschide Analiza", use_container_width=True):
            sid = session_map[chosen_label]
            st.session_state["results"] = requests.get(f"{API_URL}/api/results/{sid}").json()
            st.session_state["last_session_id"] = sid
            st.session_state.pop("active_phase_clip", None)
    else:
        st.caption("Nicio analiză salvată încă.")

tab_match, tab_bench = st.tabs(["🎥 Analiză Tactică Multi-Sport", "📊 Benchmark & Evaluare Licență"])

# ==========================================
# TAB 1: ANALIZA TACTICA MULTI-SPORT
# ==========================================
with tab_match:
    selected_sport = st.selectbox("🏅 Selectează sportul din videoclip:", SPORTS_LIST, index=0)

    col_in1, col_col1, col_in2, col_col2 = st.columns([2, 1, 2, 1])
    with col_in1:
        team_a_input = st.text_input("Nume Echipa 1 (sau 'Auto' din tabela scor)", value="Auto (Tabela Scor)")
    with col_col1:
        color_a_input = st.selectbox("Culoare 1:", COLOR_OPTIONS, index=0)
    with col_in2:
        team_b_input = st.text_input("Nume Echipa 2 (sau 'Auto' din tabela scor)", value="Auto (Tabela Scor)")
    with col_col2:
        color_b_input = st.selectbox("Culoare 2:", COLOR_OPTIONS, index=0)

    source_type = st.radio(
        "Alege sursa videoclipului:",
        ["📁 Încarcă fișier video (MP4 / AVI / MOV)", "▶️ Link YouTube (URL)"],
        horizontal=True
    )

    uploaded_file = None
    youtube_url = ""

    if source_type.startswith("📁"):
        uploaded_file = st.file_uploader("Selectează clipul video", type=["mp4", "avi", "mov"], key="match_uploader")
    else:
        youtube_url = st.text_input("Lipește link-ul YouTube al meciului/rezumatului:", placeholder="https://www.youtube.com/watch?v=...")

    with st.expander("⚙️ Opțiuni avansate: Interval timp, Model AI & Calibrare Homografică 2D", expanded=False):
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            start_sec = st.number_input("Secunda de start", min_value=0, value=0, step=5)
        with t_col2:
            duration_sec = st.number_input("Durată analizată (secunde, 0 = tot clipul / max 15 min)", min_value=0, value=0, step=10)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            selected_match_model = st.selectbox(
                "Model Detecție:",
                ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt", "yolov5su.pt", "rtdetr-l.pt", "models/football_yolov8.pt"],
                index=0
            )
        with m_col2:
            selected_match_tracker = st.selectbox(
                "Algoritm Tracking:",
                ["bytetrack.yaml", "botsort.yaml"],
                index=0
            )

        st.markdown("**📐 Calibrare Perspectivă Teren 2D (Homografie — % din ecran):**")
        st.caption("Ajustează dacă camera filmează dintr-un unghi foarte înclinat sau de aproape:")
        h_c1, h_c2, h_c3, h_c4 = st.columns(4)
        with h_c1:
            tl_x = st.number_input("Stânga-Sus X%", 0, 50, 18)
            tl_y = st.number_input("Stânga-Sus Y%", 0, 60, 22)
        with h_c2:
            tr_x = st.number_input("Dreapta-Sus X%", 50, 100, 82)
            tr_y = st.number_input("Dreapta-Sus Y%", 0, 60, 22)
        with h_c3:
            bl_x = st.number_input("Stânga-Jos X%", 0, 40, 2)
            bl_y = st.number_input("Stânga-Jos Y%", 50, 100, 94)
        with h_c4:
            br_x = st.number_input("Dreapta-Jos X%", 60, 100, 98)
            br_y = st.number_input("Dreapta-Jos Y%", 50, 100, 94)

    can_start = (uploaded_file is not None) or (len(youtube_url.strip()) > 0)

    if can_start and st.button("Pornește Analiza Tactică", type="primary"):
        with st.spinner("Se inițializează sesiunea de analiză..."):
            homography_cfg = {
                "tl_x": tl_x, "tl_y": tl_y,
                "tr_x": tr_x, "tr_y": tr_y,
                "bl_x": bl_x, "bl_y": bl_y,
                "br_x": br_x, "br_y": br_y
            }
            payload = {
                "team_a": team_a_input,
                "team_b": team_b_input,
                "color_a": color_a_input,
                "color_b": color_b_input,
                "sport": selected_sport,
                "homography_json": json.dumps(homography_cfg),
                "start_sec": str(start_sec),
                "duration_sec": str(duration_sec),
                "model_name": selected_match_model,
                "tracker_name": selected_match_tracker
            }
            if uploaded_file is not None:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
                res = requests.post(f"{API_URL}/api/analyze", files=files, data=payload)
            else:
                payload["youtube_url"] = youtube_url.strip()
                res = requests.post(f"{API_URL}/api/analyze", data=payload)

            session_id = res.json()["session_id"]
            st.session_state["last_session_id"] = session_id

        progress_bar = st.progress(0)
        status_text = st.empty()

        while True:
            status_res = requests.get(f"{API_URL}/api/status/{session_id}").json()
            state = status_res["status"]
            pct = status_res.get("progress", 0)
            progress_bar.progress(pct)
            status_text.text(f"Status procesare: {state.upper()} ({pct}%)")

            if state == "completed":
                st.session_state["results"] = requests.get(f"{API_URL}/api/results/{session_id}").json()
                st.session_state.pop("active_phase_clip", None)
                break
            elif state == "failed":
                st.error(f"Eroare la procesare: {status_res.get('error')}")
                break
            time.sleep(2)

    if "results" in st.session_state:
        data = st.session_state["results"]
        stats = data["stats"]
        visuals = data["visuals"]
        report = data["tactical_report"]
        sid = data["session_id"]
        sp_label = data.get("sport", stats.get("sport", "Fotbal (Football)"))

        st.divider()
        ball_frames = stats["video_info"].get("ball_detected_frames", 0)
        model_tag = data.get("model_used", "yolov8n.pt")
        tracker_tag = data.get("tracker_used", "bytetrack.yaml")
        st.markdown(
            f"### 📋 [{sp_label}] **{data.get('video_title', 'Match Clip')}** "
            f"(ID: `{sid}` | Model: `{model_tag}` | Tracker: `{tracker_tag}` | Minge: `{ball_frames}` cadre)"
        )

        sb_det = stats.get("scoreboard_detected")
        tact_f = stats["video_info"].get("tactical_frames", 0)
        filt_f = stats["video_info"].get("filtered_closeup_frames", 0)
        if sb_det:
            st.success(
                f"📺 **Tabelă de scor citită automat ({sb_det['source']}):** "
                f"`{sb_det['team_0_abbr']}` vs `{sb_det['team_1_abbr']}` | "
                f"Cadre tactice valide: `{tact_f}` (filtrate `{filt_f}` cadre close-up/reluări)"
            )
        else:
            st.caption(f"Cadre tactice valide: `{tact_f}` | Cadre close-up/reluări filtrate din radar: `{filt_f}`")

        st.subheader("🎥 Tracking Live + Radar Tactic 2D Homografic Orizontal")
        render_backend_video(visuals["video_url"])

        st.divider()

        # ==========================================
        # EXPLORER FAZE 10s (5s INAINTE + 5s DUPA) & CAUTARE
        # ==========================================
        st.subheader("🎬 Explorer Evenimente & Faze (Secvențe de 10s: 5s înainte + 5s după)")
        st.caption("Caută pase, șuturi/aruncări, ocazii pe flancuri sau sari direct la orice secundă din meci.")

        events = stats.get("timeline_events", [])
        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            search_query = st.text_input(
                "🔎 Caută în evenimente (ex. 'pasă', 'șut', 'dreapta', 'stânga' sau numele echipei):",
                placeholder="ex: șut pe dreapta..."
            ).strip().lower()
        with f_col2:
            flank_filter = st.selectbox("Filtru rapid flanc:", ["Toate flancurile", "Stânga", "Centru", "Dreapta"])

        filtered_events = []
        for ev in events:
            lbl = ev.get("label", "").lower()
            flk = ev.get("flank", "")
            if flank_filter != "Toate flancurile" and flk != flank_filter:
                continue
            if search_query:
                words = [w for w in search_query.split() if w not in ("pe", "de", "la", "din")]
                if words and not any(w in lbl for w in words):
                    continue
            filtered_events.append(ev)

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("**1️⃣ Alege un eveniment detectat automat:**")
            if filtered_events:
                ev_map = {e["label"]: e["timestamp_sec"] for e in filtered_events}
                chosen_ev = st.selectbox("Evenimente filtrate:", list(ev_map.keys()))
                if st.button("▶️ Vezi Faza Selectată (10s)"):
                    r = requests.post(f"{API_URL}/api/extract_phase/{sid}", data={"timestamp_sec": str(ev_map[chosen_ev])})
                    if r.status_code == 200:
                        st.session_state["active_phase_clip"] = r.json()
            else:
                st.info("Niciun eveniment nu corespunde filtrului curent.")

        with p_col2:
            st.markdown("**2️⃣ Sau sari la orice secundă din meci (Gol / Coș / Fault / Raliu):**")
            v_start = float(stats["video_info"].get("start_sec", 0.0))
            v_end = float(stats["video_info"].get("end_sec", 30.0))
            custom_sec = st.slider(
                "Selectează secunda evenimentului (se decupează ±5s):",
                min_value=v_start,
                max_value=max(v_start + 1.0, v_end),
                value=round((v_start + v_end) / 2, 1),
                step=0.5
            )
            if st.button("✂️ Extrage Faza de 10s din jurul acestei secunde"):
                r = requests.post(f"{API_URL}/api/extract_phase/{sid}", data={"timestamp_sec": str(custom_sec)})
                if r.status_code == 200:
                    st.session_state["active_phase_clip"] = r.json()

        if "active_phase_clip" in st.session_state:
            clip_info = st.session_state["active_phase_clip"]
            st.success(f"Secvență extrasă: **{clip_info['window_start']}s — {clip_info['window_end']}s** (5s înainte + 5s după)")
            render_backend_video(clip_info["clip_url"])

        if visuals["key_frames"]:
            with st.expander("🏆 Top 3 Momente Cheie Extrase Automat (Clipuri 10s)", expanded=False):
                kf_cols = st.columns(len(visuals["key_frames"]))
                for idx, kf in enumerate(visuals["key_frames"]):
                    with kf_cols[idx]:
                        st.markdown(f"**Moment #{idx+1} — Sec {kf['timestamp_sec']}s** ({kf['attacking_team']} | {kf.get('flank', '')})")
                        if kf.get("clip_url"):
                            render_backend_video(kf["clip_url"])
                        else:
                            render_backend_image(kf["url"])

        st.divider()

        # ==========================================
        # STATISTICI ECHIPE & HARTI TACTICE 2D
        # ==========================================
        top_ctrl1, top_ctrl2 = st.columns([3, 1])
        with top_ctrl2:
            st.write("")
            if st.button("🔄 Inversează Echipele (inclusiv în Video)"):
                with st.spinner("Se inversează echipele și se re-randează videoclipul + radarul 2D..."):
                    r_swap = requests.post(f"{API_URL}/api/swap_teams/{sid}")
                    if r_swap.status_code == 200:
                        st.session_state["results"] = r_swap.json()
                        st.rerun()

        t0 = stats["teams"]["0"]
        t1 = stats["teams"]["1"]
        h0_url = visuals["heatmap_team_0_url"]
        h1_url = visuals["heatmap_team_1_url"]
        ev_lbls = stats.get("event_labels", {"pass": "Pase", "shot": "Șuturi"})

        with top_ctrl1:
            view_mode = st.selectbox(
                "🔍 Selectează modul de vizualizare a statisticilor tactice:",
                [
                    f"Comparație: {t0['name']} vs. {t1['name']}",
                    f"Doar {t0['name']}",
                    f"Doar {t1['name']}"
                ]
            )

        def render_team_card(team_data, pitch_map_url):
            st.markdown(
                f"### {team_data['name']} "
                f"<span style='background-color:{team_data['jersey_hex']};padding:2px 12px;border-radius:4px;border:1px solid #888;'>&nbsp;</span>",
                unsafe_allow_html=True
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Timp pe Atac", f"{team_data['attacking_share_pct']}%")
            m2.metric("Avans Teren", f"{team_data['avg_line_height_pct']}%")
            m3.metric(f"{ev_lbls.get('pass', 'Pase')} est.", f"{team_data.get('estimated_passes', 0)}")
            m4.metric(f"{ev_lbls.get('shot', 'Șuturi')} est.", f"{team_data.get('estimated_shots', 0)}")

            dist = team_data["attacking_distribution_pct"]
            st.write("**Distribuția Reală pe Flancuri (Stânga / Centru / Dreapta terenului):**")
            d1, d2, d3 = st.columns(3)
            d1.metric("Flanc Stânga", f"{dist['left']}%")
            d2.metric("Zonă Centrală", f"{dist['center']}%")
            d3.metric("Flanc Dreapta", f"{dist['right']}%")

            render_backend_image(pitch_map_url, caption=f"Hartă Tactică 2D Teren — {team_data['name']}")

            st.write("**Top Jucători Activi:**")
            for p in team_data["top_active_players"]:
                st.write(f"• Jucător **#{p['track_id']}** — Indice deplasare 2D: `{p['movement_score']}`")

        if view_mode.startswith("Comparație"):
            c_left, c_right = st.columns(2)
            with c_left:
                render_team_card(t0, h0_url)
            with c_right:
                render_team_card(t1, h1_url)
        elif view_mode == f"Doar {t0['name']}":
            render_team_card(t0, h0_url)
        else:
            render_team_card(t1, h1_url)

        st.divider()

        st.subheader("📈 Timeline Dinamică Atac / Apărare")
        render_backend_image(visuals["timeline_url"])

        st.divider()

        st.subheader("🧠 Raport Tactic Comparativ")
        st.info(f"**Strategie Dominantă:** {report.get('dominant_strategy')}")
        for obs in report.get("key_observations", []):
            st.write(f"- {obs}")
        st.write(f"**Recomandare Antrenor:** {report.get('tactical_recommendation')}")

# ==========================================
# TAB 2: BENCHMARK & EVALUARE LICENTA
# ==========================================
with tab_bench:
    st.subheader("🔬 Evaluare Cantitativă & Studiu de Ablație (Modele, Trackere & Hardware)")
    st.write("Compară arhitecturi diferite (YOLOv8, YOLOv11, YOLOv5u, RT-DETR Transformer) și algoritmi de tracking (ByteTrack vs. BoT-SORT).")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        selected_models = st.multiselect(
            "Modele de Detecție de comparat:",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt", "yolov5su.pt", "rtdetr-l.pt"],
            default=["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "rtdetr-l.pt"]
        )
    with b_col2:
        selected_trackers = st.multiselect(
            "Algoritmi Tracking (ByteTrack vs. BoT-SORT):",
            ["bytetrack.yaml", "botsort.yaml"],
            default=["bytetrack.yaml", "botsort.yaml"]
        )

    b_col3, b_col4 = st.columns(2)
    with b_col3:
        selected_devices = st.multiselect(
            "Dispozitive Hardware:",
            ["mps", "cpu"],
            default=["mps"]
        )
    with b_col4:
        eval_frames = st.slider("Număr cadre evaluate per configurație:", min_value=60, max_value=240, value=90, step=30)

    bench_file = st.file_uploader("Încarcă un clip de referință (sau lasă gol pentru a folosi ultimul meci analizat)", type=["mp4", "avi", "mov"], key="bench_uploader")
    last_sid = st.session_state.get("last_session_id", "")

    if st.button("🚀 Rulează Benchmark Comparativ", type="primary"):
        if not bench_file and not last_sid:
            st.warning("Încarcă un clip video aici sau analizează mai întâi un clip în primul tab.")
        elif not selected_models or not selected_trackers or not selected_devices:
            st.warning("Selectează cel puțin un model, un tracker și un dispozitiv.")
        else:
            data_payload = {
                "models": ",".join(selected_models),
                "trackers": ",".join(selected_trackers),
                "devices": ",".join(selected_devices),
                "max_frames": str(eval_frames),
                "session_id": last_sid
            }
            if bench_file is not None:
                files = {"file": (bench_file.name, bench_file.getvalue(), "video/mp4")}
                res = requests.post(f"{API_URL}/api/benchmark", files=files, data=data_payload)
            else:
                res = requests.post(f"{API_URL}/api/benchmark", data=data_payload)

            bench_id = res.json()["bench_id"]
            b_prog = st.progress(0)
            b_msg = st.empty()

            while True:
                st_res = requests.get(f"{API_URL}/api/benchmark/status/{bench_id}").json()
                b_prog.progress(st_res.get("progress", 0))
                b_msg.text(st_res.get("message", ""))

                if st_res["status"] == "completed":
                    st.session_state["bench_output"] = st_res
                    break
                elif st_res["status"] == "failed":
                    st.error(f"Eroare benchmark: {st_res.get('error')}")
                    break
                time.sleep(2)

    if "bench_output" in st.session_state:
        b_out = st.session_state["bench_output"]
        df = pd.DataFrame(b_out["results"])

        st.divider()
        st.subheader("📋 Tabel Rezultate Experimentale")
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descarcă Tabelul CSV (benchmark_results.csv)",
            data=csv_bytes,
            file_name="benchmark_results.csv",
            mime="text/csv"
        )

        st.subheader("📈 Grafice Comparative (salvate automat în folderul benchmarks/)")
        render_backend_image(b_out["urls"]["perf_chart"], caption="Comparație FPS și Latență (Modele, Trackere & Hardware)")
        render_backend_image(b_out["urls"]["acc_chart"], caption="Comparație Detecție și Stabilitate Tracking")

    # ==========================================
    # EVALUARE GROUND TRUTH MANUAL VS SISTEM
    # ==========================================
    st.divider()
    st.subheader("🎯 Validare Acuratețe Evenimente (Ground Truth Manual vs. Sistem)")
    st.write("Compară fazele detectate automat pe meciul activ cu adnotările tale manuale (Playbook Secțiunea 7).")

    if "results" not in st.session_state:
        st.info("Deschide sau analizează mai întâi un meci în Tab 1 (sau din Istoric) pentru a-l evalua.")
    else:
        res_data = st.session_state["results"]
        active_sid = res_data["session_id"]
        t0_n = res_data["stats"]["teams"]["0"]["name"]
        t1_n = res_data["stats"]["teams"]["1"]["name"]

        tol_sec = st.slider("Fereastră de toleranță timestamp (± secunde):", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

        default_gt = pd.DataFrame([
            {"timestamp_sec": 4.0, "attacking_team": t0_n, "flank": "Dreapta"},
            {"timestamp_sec": 10.0, "attacking_team": t1_n, "flank": "Centru"},
            {"timestamp_sec": 16.0, "attacking_team": t0_n, "flank": "Stânga"},
        ])
        st.caption("Editează tabelul de mai jos cu momentele reale observate de tine în videoclip (poți adăuga/șterge rânduri):")
        edited_gt = st.data_editor(default_gt, num_rows="dynamic", use_container_width=True)

        if st.button("📐 Calculează Metricile de Acuratețe (Precision / Recall / F1)"):
            gt_records = edited_gt.dropna(subset=["timestamp_sec"]).to_dict(orient="records")
            r_gt = requests.post(
                f"{API_URL}/api/evaluate_ground_truth/{active_sid}",
                data={"ground_truth_json": pd.Series(gt_records).to_json(orient="values"), "tolerance_sec": str(tol_sec)}
            )
            if r_gt.status_code == 200:
                m = r_gt.json()
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("Precision", f"{m['precision_pct']}%")
                g2.metric("Recall", f"{m['recall_pct']}%")
                g3.metric("F1-Score", f"{m['f1_score_pct']}%")
                g4.metric("Eroare Medie Timp", f"{m['mean_timestamp_error_sec']} s")

                g5, g6 = st.columns(2)
                g5.metric("Acuratețe Atribuire Echipă", f"{m['team_attribution_acc_pct']}%")
                g6.metric("Acuratețe Clasificare Flanc", f"{m['flank_classification_acc_pct']}%")

                gt_df = pd.DataFrame([m])
                st.download_button(
                    "⬇️ Descarcă Raport Ground Truth (CSV)",
                    data=gt_df.to_csv(index=False).encode("utf-8"),
                    file_name="ground_truth_evaluation.csv",
                    mime="text/csv"
                )