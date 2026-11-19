import os
import json
from PIL import Image
from google import genai
from google.genai import types


def generate_tactical_report(stats: dict, key_frames: list, session_dir: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    report_path = os.path.join(session_dir, "tactical_report.json")

    t0 = stats["teams"]["0"]
    t1 = stats["teams"]["1"]

    prompt = f"""
    Ești un analist tactic de fotbal. Compară cele două echipe pe baza datelor deterministe măsurate și a cadrelor cheie atașate.
    Regulă strictă: Nu inventa cifre sau evenimente care nu apar în datele de mai jos.

    ECHIPA 1 ({t0['name']}):
    - Timp în faza de atac: {t0['attacking_share_pct']}%
    - Distribuție atac: Stânga {t0['attacking_distribution_pct']['left']}%, Centru {t0['attacking_distribution_pct']['center']}%, Dreapta {t0['attacking_distribution_pct']['right']}%
    - Top jucători activi: {t0['top_active_players']}

    ECHIPA 2 ({t1['name']}):
    - Timp în faza de atac: {t1['attacking_share_pct']}%
    - Distribuție atac: Stânga {t1['attacking_distribution_pct']['left']}%, Centru {t1['attacking_distribution_pct']['center']}%, Dreapta {t1['attacking_distribution_pct']['right']}%
    - Top jucători activi: {t1['top_active_players']}

    Returnează un JSON valid cu această structură:
    {{
        "dominant_strategy": "Comparație scurtă între stilul ofensiv al {t0['name']} și {t1['name']}",
        "key_observations": ["Observație 1 ancorată în cifre", "Observație 2", "Observație 3"],
        "visual_context_notes": "Ce se observă în cadrele cheie (liniile de apărare/atac)",
        "tactical_recommendation": "O recomandare concretă"
    }}
    """

    if not api_key:
        fallback = {
            "dominant_strategy": f"{t0['name']} a controlat {t0['attacking_share_pct']}% din fazele de atac, iar {t1['name']} {t1['attacking_share_pct']}%.",
            "key_observations": [
                f"{t0['name']} distribuție atac: Stânga {t0['attacking_distribution_pct']['left']}%, Centru {t0['attacking_distribution_pct']['center']}%, Dreapta {t0['attacking_distribution_pct']['right']}%.",
                f"{t1['name']} distribuție atac: Stânga {t1['attacking_distribution_pct']['left']}%, Centru {t1['attacking_distribution_pct']['center']}%, Dreapta {t1['attacking_distribution_pct']['right']}%.",
                f"Au fost detectate {len(stats['timeline_events'])} momente cheie de tranziție/presiune."
            ],
            "visual_context_notes": "Cheia GEMINI_API_KEY nu este activă încă; raportul este generat din statisticile CV per echipă.",
            "tactical_recommendation": f"Valorificarea flancurilor mai puțin aglomerate în tranzițiile rapide."
        }
        with open(report_path, "w") as f:
            json.dump(fallback, f, indent=2)
        return fallback

    client = genai.Client(api_key=api_key)
    contents = [prompt]
    for kf in key_frames[:2]:
        if os.path.exists(kf["path"]):
            contents.append(Image.open(kf["path"]))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )

    report_data = json.loads(response.text)
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    return report_data