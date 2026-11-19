import time
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Sports Tactical Intelligence", layout="wide")
st.title("⚽ AI Sports Video Analysis & Tactical Intelligence")

col_in1, col_in2 = st.columns(2)
with col_in1:
    team_a_input = st.text_input("Nume Echipa 1 (opțional)", value="Team A")
with col_in2:
    team_b_input = st.text_input("Nume Echipa 2 (opțional)", value="Team B")

uploaded_file = st.file_uploader("Încarcă un clip video din meci (MP4, max 5 min)", type=["mp4", "avi", "mov"])

if uploaded_file and st.button("Pornește Analiza Tactică", type="primary"):
    with st.spinner("Se încarcă videoclipul și se calibrează culorile echipelor..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
        payload = {"team_a": team_a_input, "team_b": team_b_input}
        res = requests.post(f"{API_URL}/api/analyze", files=files, data=payload)
        session_id = res.json()["session_id"]

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
    t0 = stats["teams"]["0"]
    t1 = stats["teams"]["1"]

    st.divider()

    st.subheader("🎥 Tracking Echipe, Culoare Tricouri & Faze Atac / Apărare")
    st.video(f"{API_URL}{visuals['video_url']}")

    st.divider()

    view_mode = st.selectbox(
        "🔍 Selectează modul de vizualizare a statisticilor tactice:",
        [
            f"Comparație: {t0['name']} vs. {t1['name']}",
            f"Doar {t0['name']}",
            f"Doar {t1['name']}"
        ]
    )

    def render_team_card(team_data, heatmap_url):
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

        st.image(f"{API_URL}{heatmap_url}", caption=f"Heatmap {team_data['name']}")

        st.write("**Top Jucători Activi:**")
        for p in team_data["top_active_players"]:
            st.write(f"• Jucător **#{p['track_id']}** — Mișcare: `{p['movement_score']} px`")

    if view_mode.startswith("Comparație"):
        c_left, c_right = st.columns(2)
        with c_left:
            render_team_card(t0, visuals["heatmap_team_0_url"])
        with c_right:
            render_team_card(t1, visuals["heatmap_team_1_url"])
    elif view_mode == f"Doar {t0['name']}":
        render_team_card(t0, visuals["heatmap_team_0_url"])
    else:
        render_team_card(t1, visuals["heatmap_team_1_url"])

    st.divider()

    st.subheader("📈 Timeline Dinamică Atac / Apărare")
    st.image(f"{API_URL}{visuals['timeline_url']}")

    st.divider()

    st.subheader("🧠 Raport Tactic Comparativ")
    st.info(f"**Strategie Dominantă:** {report.get('dominant_strategy')}")
    for obs in report.get("key_observations", []):
        st.write(f"- {obs}")
    st.write(f"**Recomandare Antrenor:** {report.get('tactical_recommendation')}")

    if visuals["key_frames"]:
        st.subheader("📸 Momente Cheie Extrase Automat")
        kf_cols = st.columns(len(visuals["key_frames"]))
        for idx, kf in enumerate(visuals["key_frames"]):
            with kf_cols[idx]:
                st.image(
                    f"{API_URL}{kf['url']}",
                    caption=f"Sec {kf['timestamp_sec']}s | Atac: {kf['attacking_team']}"
                )