import time
import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Sports Tactical Intelligence", layout="wide")
st.title("⚽ AI Sports Video Analysis & Tactical Intelligence")

# Sidebar pentru Istoric Sesiuni Salvate
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
            st.session_state["swapped"] = False
            st.session_state.pop("active_phase_clip", None)
    else:
        st.caption("Nicio analiză salvată încă.")

tab_match, tab_bench = st.tabs(["🎥 Analiză Tactică Meci", "📊 Benchmark & Evaluare Licență"])

# ==========================================
# TAB 1: ANALIZA TACTICA MECI
# ==========================================
with tab_match:
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        team_a_input = st.text_input("Nume Echipa 1 (opțional)", value="Team A")
    with col_in2:
        team_b_input = st.text_input("Nume Echipa 2 (opțional)", value="Team B")

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

    with st.expander("⏱️ Opțiuni interval analizat (opțional — util pentru rezumate lungi)", expanded=False):
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            start_sec = st.number_input("Secunda de start", min_value=0, value=0, step=5)
        with t_col2:
            duration_sec = st.number_input("Durată analizată (secunde, 0 = tot clipul / max 5 min)", min_value=0, value=0, step=10)

    can_start = (uploaded_file is not None) or (len(youtube_url.strip()) > 0)

    if can_start and st.button("Pornește Analiza Tactică", type="primary"):
        with st.spinner("Se inițializează sesiunea de analiză..."):
            payload = {
                "team_a": team_a_input,
                "team_b": team_b_input,
                "start_sec": str(start_sec),
                "duration_sec": str(duration_sec)
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
                st.session_state["swapped"] = False
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

        st.divider()
        ball_frames = stats["video_info"].get("ball_detected_frames", 0)
        st.markdown(
            f"### 📋 Meci analizat: **{data.get('video_title', 'Match Clip')}** "
            f"(ID: `{sid}` | Cadre minge detectată: `{ball_frames}`)"
        )

        st.subheader("🎥 Tracking Live + Radar Tactic 2D Sincronizat")
        st.video(f"{API_URL}{visuals['video_url']}")

        st.divider()

        # ==========================================
        # EXPLORER FAZE 10s (5s INAINTE + 5s DUPA) & CAUTARE
        # ==========================================
        st.subheader("🎬 Explorer Faze & Momente (Secvențe de 10s: 5s înainte + 5s după)")
        st.caption("Caută o fază după echipă/flanc sau alege orice moment (ex. înainte de un gol sau fault) pentru a viziona secvența de 10 secunde.")

        events = stats.get("timeline_events", [])
        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            search_query = st.text_input(
                "🔎 Caută în evenimente (ex. 'dreapta', 'stânga', 'centru' sau numele echipei):",
                placeholder="ex: atac pe dreapta..."
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
                words = [w for w in search_query.split() if w not in ("pe", "de", "la", "din", "atac", "atacuri", "faza", "faze")]
                if words and not any(w in lbl for w in words):
                    continue
            filtered_events.append(ev)

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("**1️⃣ Alege o fază detectată automat:**")
            if filtered_events:
                ev_map = {e["label"]: e["timestamp_sec"] for e in filtered_events}
                chosen_ev = st.selectbox("Faze filtrate:", list(ev_map.keys()))
                if st.button("▶️ Vezi Faza Selectată (10s)"):
                    r = requests.post(f"{API_URL}/api/extract_phase/{sid}", data={"timestamp_sec": str(ev_map[chosen_ev])})
                    if r.status_code == 200:
                        st.session_state["active_phase_clip"] = r.json()
            else:
                st.info("Nicio fază nu corespunde filtrului curent.")

        with p_col2:
            st.markdown("**2️⃣ Sau sari la orice secundă din meci (Gol / Fault / Ocazie):**")
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
            st.video(f"{API_URL}{clip_info['clip_url']}")

        if visuals["key_frames"]:
            with st.expander("🏆 Top 3 Momente Cheie Extrase Automat (Clipuri 10s)", expanded=False):
                kf_cols = st.columns(len(visuals["key_frames"]))
                for idx, kf in enumerate(visuals["key_frames"]):
                    with kf_cols[idx]:
                        st.markdown(f"**Moment #{idx+1} — Sec {kf['timestamp_sec']}s** ({kf['attacking_team']} | {kf.get('flank', '')})")
                        if kf.get("clip_url"):
                            st.video(f"{API_URL}{kf['clip_url']}")
                        else:
                            st.image(f"{API_URL}{kf['url']}")

        st.divider()

        # ==========================================
        # STATISTICI ECHIPE & HARTI TACTICE 2D
        # ==========================================
        top_ctrl1, top_ctrl2 = st.columns([3, 1])
        with top_ctrl2:
            st.write("")
            if st.button("🔄 Inversează Echipele"):
                st.session_state["swapped"] = not st.session_state.get("swapped", False)

        if st.session_state.get("swapped", False):
            t0 = stats["teams"]["1"]
            t1 = stats["teams"]["0"]
            h0_url = visuals["heatmap_team_1_url"]
            h1_url = visuals["heatmap_team_0_url"]
        else:
            t0 = stats["teams"]["0"]
            t1 = stats["teams"]["1"]
            h0_url = visuals["heatmap_team_0_url"]
            h1_url = visuals["heatmap_team_1_url"]

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
            m1, m2 = st.columns(2)
            m1.metric("Timp pe Atac", f"{team_data['attacking_share_pct']}%")
            m2.metric("Poziționare Medie Teren", f"{team_data['avg_line_height_pct']}%")

            dist = team_data["attacking_distribution_pct"]
            st.write("**Distribuția Atacului (Stânga / Centru / Dreapta):**")
            d1, d2, d3 = st.columns(3)
            d1.metric("Stânga", f"{dist['left']}%")
            d2.metric("Centru", f"{dist['center']}%")
            d3.metric("Dreapta", f"{dist['right']}%")

            st.image(f"{API_URL}{pitch_map_url}", caption=f"Hartă Tactică 2D Teren — {team_data['name']}")

            st.write("**Top Jucători Activi:**")
            for p in team_data["top_active_players"]:
                st.write(f"• Jucător **#{p['track_id']}** — Mișcare: `{p['movement_score']} px`")

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
        st.image(f"{API_URL}{visuals['timeline_url']}")

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
    st.subheader("🔬 Evaluare Cantitativă & Studiu de Ablație (Hardware & Modele)")
    st.write("Rulează experimente comparative pe același clip video pentru a genera tabelele CSV și graficele din capitolul de evaluare al licenței.")

    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        selected_models = st.multiselect(
            "Modele YOLOv8 de comparat:",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            default=["yolov8n.pt", "yolov8s.pt"]
        )
    with b_col2:
        selected_devices = st.multiselect(
            "Dispozitive Hardware:",
            ["mps", "cpu"],
            default=["mps", "cpu"]
        )
    with b_col3:
        eval_frames = st.slider("Număr cadre evaluate per configurație:", min_value=60, max_value=300, value=120, step=30)

    bench_file = st.file_uploader("Încarcă un clip de referință (sau lasă gol pentru a folosi ultimul meci analizat)", type=["mp4", "avi", "mov"], key="bench_uploader")
    last_sid = st.session_state.get("last_session_id", "")

    if st.button("🚀 Rulează Benchmark Comparativ", type="primary"):
        if not bench_file and not last_sid:
            st.warning("Încarcă un clip video aici sau analizează mai întâi un clip în primul tab.")
        elif not selected_models or not selected_devices:
            st.warning("Selectează cel puțin un model și un dispozitiv.")
        else:
            data_payload = {
                "models": ",".join(selected_models),
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
        st.image(f"{API_URL}{b_out['urls']['perf_chart']}", caption="Comparație FPS și Latență (MPS vs CPU / Modele)")
        st.image(f"{API_URL}{b_out['urls']['acc_chart']}", caption="Comparație Detecție și Stabilitate Tracking")