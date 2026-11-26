# AI-Powered Sports Video Analysis & Tactical Intelligence Platform

An end-to-end computer vision and multimodal AI platform that transforms recorded sports footage into multi-object player & ball tracking, 2D tactical radar projections, deterministic event analytics, hardware/model benchmarks, and evidence-grounded tactical reports.

---

## 1. Problema (Problem Statement)

### Ce rezolvă și cu ce ajută
Analiza meciurilor de sport pe baza înregistrărilor video necesită în mod tradițional un efort manual considerabil din partea antrenorilor și analiștilor pentru extragerea statisticilor, urmărirea jucătorilor și identificarea tiparelor tactice.

Această platformă automatizează întregul flux prin procesarea secvențelor video (fișiere locale sau link-uri YouTube), filtrarea cadrelor non-tactice (reluări, prim-planuri), detectarea și urmărirea jucătorilor și a mingii, proiecția 2D a pozițiilor pe teren, extragerea de evenimente-cheie și generarea automată de rapoarte tactice în limbaj natural.

### Obiectivele Proiectului
1. **Pipeline de procesare video multi-sport cap-la-cap:** Construirea unui sistem integrat (*Upload / YouTube $\rightarrow$ Camera Filtering $\rightarrow$ Detection & Tracking $\rightarrow$ 2D Homography & Analytics $\rightarrow$ Multimodal AI Report*) adaptat pentru sporturi de echipă (Fotbal, Handbal, Baschet, Fotbal American).
2. **Clasificare automată a echipelor și calibrare spațială:** Separarea jucătorilor pe echipe folosind clustering nesupervizat (K-Means în spațiul de culoare CIELAB) asistat de citirea automată a tabelei de scor (OCR) și maparea coordonatelor din imagine pe un teren 2D normalizat (radar tactic).
3. **Analitică deterministă și sinteze tactice ancorate în date (Grounded VLM/LLM):** Calcularea mecanică a evenimentelor de joc (atacuri pe flancuri, schimbări de posesie, zone de presiune, heatmaps, timeline) și generarea rapoartelor tactice folosind modele multimodale alimentate exclusiv cu statistici verificate și cadre-cheie.
4. **Benchmarking comparativ și validare Ground-Truth:** Evaluarea cantitativă a modelelor de detecție (YOLOv8, YOLO11, RT-DETR), a algoritmilor de tracking (ByteTrack, BoT-SORT) și a acurateței detecției evenimentelor față de adnotări manuale.

---

## 2. Documentație și Stadiul Actual (Related Work & State of the Art)

Sistemele comerciale de analiză sportivă (ex: Tracab, Hawk-Eye, Veo) depind de camere calibrate dedicate sau echipamente multisenzor costisitoare. Proiectele moderne de cercetare și soluțiile open-source urmăresc obținerea unor metrici similare direct din fluxuri video standard (broadcast TV sau camere tactice), combinând detectoare bazate pe rețele neuronale cu algoritmi rapizi de asociere temporală (MOT).

### Tehnologii și Referințe Cheie
* **[Ultralytics YOLOv8 / YOLO11 & RT-DETR](https://github.com/ultralytics/ultralytics):** Modele de detecție a obiectelor în timp real (arhitecturi CNN și Vision Transformer) utilizate pentru detectarea jucătorilor și a mingii, cu suport pentru accelerare hardware (Apple Silicon MPS / CUDA / CPU).
* **[ByteTrack](https://github.com/ifzhang/ByteTrack) & [BoT-SORT](https://github.com/NirAharon/BoT-SORT):** Algoritmi de urmărire multi-obiect (MOT) care mențin ID-uri persistente pentru jucători în condiții de ocluzie și mișcare a camerei.
* **[Roboflow Supervision](https://github.com/roboflow/supervision):** Librărie pentru gestionarea detecțiilor, tracking și adnotări vizuale pe cadre video.
* **[EasyOCR](https://github.com/JaidedAI/EasyOCR) & [OpenCV](https://opencv.org/):** Utilizate pentru recunoașterea automată a abrevierilor echipelor din tabela de scor (scoreboard), extragerea culorilor dominante în spațiul CIELAB, transformări projective (homografie) și procesare video.
* **Modele Multimodale VLM / LLM (Google Gemini / OpenAI):** Utilizate pentru interpretarea semantică a cadrelor-cheie și redactarea rapoartelor tactice structurate.
* **[FlashScore](https://www.flashscore.ro):** Referință pentru prezentarea structurată a statisticilor de meci și a cronologiei evenimentelor.

### Cu ce este mai bun acest proiect?
Spre deosebire de instrumentele pur vizuale (care afișează doar bounding boxes și hărți de căldură brute) sau de abordările pur generative (susceptibile la halucinații), platforma combină **analitica deterministă** (calibrare colorimetrică a echipelor, filtrare a cadrelor non-tactice, homografie 2D) cu **raționamentul multimodal ancorat în dovezi**. Totodată, include nativ un modul de **benchmarking** și **evaluare Ground-Truth** pentru compararea obiectivă a modelelor CV.

---

## 3. Proiectare (Architecture & Design)

### Arhitectura Sistemului
Aplicația este structurată decuplat într-un frontend interactiv (Streamlit) și un backend asincron (FastAPI) care orchestrează pipeline-ul de Computer Vision, generarea vizualizărilor și modulul de benchmarking:

```text
[ Frontend Dashboard (Streamlit - frontend/app.py) ]
                       │  (REST API / JSON / Video Streams)
                       ▼
[ Backend Orchestrator (FastAPI - backend/main.py) ]
       │                                       │
       ▼                                       ▼
[ Video Analysis Pipeline ]             [ Benchmark & Evaluation Engine ]
  ├── YouTube Downloader (yt-dlp)         ├── Multi-Model / Tracker Runner
  ├── Scoreboard OCR (EasyOCR)            ├── Latency & FPS Profiler
  ├── Camera Angle Filter                 └── Ground-Truth Event Evaluator
  ├── Detection (YOLO / RT-DETR)
  ├── Tracking (ByteTrack / BoT-SORT)   [ Local Session Storage (storage/) ]
  ├── Team Classifier (CIELAB K-Means)    ├── annotated_video.mp4 (Video + 2D Radar)
  ├── 2D Homography & Analytics           ├── heatmaps & timeline.png
  └── Multimodal Report (VLM / LLM)       └── tracking_data.json & final_result.json
```

### Stack Tehnologic
* **Frontend:** Python, Streamlit (configurare analiză, selecție interactivă a colțurilor terenului pentru homografie, player video sincronizat, inversare rapidă a echipelor, explorator de faze la secundă și tab de benchmarking).
* **Backend API:** Python, FastAPI, Uvicorn, BackgroundTasks pentru procesare asincronă cu urmărirea progresului în timp real.
* **Computer Vision & ML:** PyTorch, Ultralytics (`yolov8n`, `yolov8s`, `yolo11n`, `rtdetr-l` și suport pentru modele fine-tuned pe sport), `supervision`, OpenCV, NumPy, scikit-learn (K-Means), EasyOCR.
* **Multimodal AI & Media:** Google Gemini / OpenAI API pentru rapoarte tactice, `yt-dlp` pentru ingestie video din YouTube, Matplotlib/Seaborn pentru grafice analitice și benchmark.

### Soluții Cheie de Design și Optimizare
1. **Filtrarea Cadrelor Non-Tactice & Calibrare Hibridă a Echipelor:** Un clasificator de cameră (`camera_classifier.py`) detectează cadrele de tip *close-up*, reluări sau unghiuri din spatele porții și îngheață radarul tactic pe ultimele poziții valide. Clasificarea jucătorilor combină eșantionarea culorilor din tabela de scor (`scoreboard_reader.py`) cu clustering K-Means în spațiul perceptiv **CIELAB**.
2. **Ancorarea Datelor pentru Raportul AI (Grounded Tactical Insights) & Swap Instant:** LLM-ul primește exclusiv metricile deterministe calculate (posesie estimată, distribuția atacurilor pe flancuri, evenimente cronologice) și cadre-cheie extrase automat. În plus, sistemul permite inversarea instantanee a echipelor (`rerender_swapped_video`) direct din datele salvate în `tracking_data.json`, fără a re-rula detecția neurală.

---

## 4. Implementare (Implementation Details)

### Structura Proiectului
```text
├── backend/
│   ├── main.py                 # Endpoint-uri FastAPI, gestionare sesiuni și job-uri asincrone
│   ├── pipeline.py             # Pipeline-ul CV: detecție, tracking, clasificare echipe, radar 2D
│   ├── camera_classifier.py    # Filtrarea cadrelor tactice vs. reluări / close-up-uri
│   ├── scoreboard_reader.py    # Detecție OCR a echipelor și culorilor din tabela de scor
│   ├── analytics.py            # Generare heatmaps, timeline evenimente și clipuri de 10s
│   ├── llm_service.py          # Generarea raportului tactic multimodal ancorat în statistici
│   ├── benchmark.py            # Evaluare performanță (FPS/latență) și acuratețe Ground-Truth
│   └── youtube_service.py      # Descărcare și decupare segmente video din YouTube
├── frontend/
│   └── app.py                  # Interfața web Streamlit (Analiză Meci, Sesiuni Salvate, Benchmark)
├── storage/                    # Artefacte generate per sesiune (video adnotat, JSON, grafice)
└── requirements.txt            # Dependențele Python ale proiectului
```

### Etapele Pipeline-ului de Analiză
1. **Ingestie & Preprocesare:** Încărcarea unui clip video local sau descărcarea unui interval specific (`start_sec`, `duration_sec`) de pe YouTube.
2. **Calibrare Inițială (Scoreboard & Echipamente):** Extragerea numelor/culorilor din tabela de scor (dacă este vizibilă) și antrenarea clasificatorului de echipe pe primele cadre tactice valide.
3. **Detecție & Multi-Object Tracking:** Rularea modelului selectat (ex. `yolov8n.pt`, `yolov8s.pt`, `rtdetr-l.pt` sau modele fine-tuned pe sport) și asocierea ID-urilor persistente cu ByteTrack / BoT-SORT.
4. **Proiecție 2D (Homografie) & Radar Live:** Maparea pozițiilor jucătorilor și ale mingii pe terenul 2D (folosind fie colțurile calibrate manual de utilizator, fie mapare normalizată) și randarea video-ului compus (cadru adnotat + radar tactic sincronizat).
5. **Analitică Deterministă & Extragere Faze:** Identificarea fazelor de atac (stânga, centru, dreapta), schimbărilor de posesie, generarea hărților de căldură (`heatmap_team_0.png`, `heatmap_team_1.png`), a graficului cronologic (`timeline.png`) și decuparea automată a clipurilor video de 10 secunde pentru fazele-cheie.
6. **Sinteză Tactică VLM/LLM & Export:** Generarea raportului tactic structurat și salvarea întregii sesiuni pentru reîncărcare ulterioară.

---

## 5. Testare și Evaluare (Testing & Evaluation)

Platforma include un modul dedicat de evaluare (`backend/benchmark.py`), accesibil direct din interfața grafică, care măsoară obiectiv compromisul dintre viteză și acuratețe:

### 1. Benchmark de Performanță și Stabilitate (Model $\times$ Tracker $\times$ Hardware)
* **Configurații testate:** Compararea arhitecturilor de detecție (`yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt`, `rtdetr-l.pt`) combinate cu trackere (`bytetrack.yaml`, `botsort.yaml`) pe dispozitive hardware diferite (`mps`, `cpu`, `cuda`).
* **Metrici colectate:** Latența medie per cadru (ms), throughput-ul (FPS), stabilitatea ID-urilor de tracking și rata de detecție a mingii/jucătorilor, exportate automat în `benchmark_results.csv` și grafice comparative (`latency_fps_comparison.png`, `detection_stability_comparison.png`).

### 2. Evaluare Ground-Truth a Evenimentelor Tactice
* **Metodologie:** Compararea evenimentelor detectate automat în `timeline_events` cu un fișier JSON de adnotări manuale (Ground-Truth), aplicând o fereastră de toleranță temporală configurabilă (ex: $\pm 3.0$ secunde).
* **Metrici:** *Precision*, *Recall*, *F1-Score* și eroarea medie temporală (MAE în secunde).

### 3. Evaluare Calitativă a Rapoartelor Tactice
* Verificarea consistenței între cifrele raportate de motorul determinist și afirmațiile generate de modelul VLM/LLM (corectitudine factuală, relevanță tactică și utilitate practică).

---

## 6. Rulare Locală (Quick Start)

1. **Activarea mediului virtual și instalarea dependențelor:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configurarea variabilelor de mediu:**
   Creează un fișier `.env` în rădăcina proiectului cu cheia API pentru raportul tactic (ex: `GEMINI_API_KEY=...` sau `OPENAI_API_KEY=...`).
3. **Pornirea serverului Backend (FastAPI):**
   ```bash
   uvicorn backend.main:app --reload
   ```
4. **Pornirea interfeței Frontend (Streamlit) într-un al doilea terminal:**
   ```bash
   source venv/bin/activate
   streamlit run frontend/app.py --server.address 0.0.0.0
   ```

---

## 7. Concluzii și Direcții Viitoare (Conclusions & Future Work)

### Concluzii
Platforma demonstrează că o arhitectură hibridă — care combină detectoare și trackere moderne de Computer Vision, algoritmi determiniști de calibrare colorimetrică/spațială și modele multimodale ancorate în date — poate livra o analiză tactică completă din înregistrări video standard, fără hardware specializat.

### Direcții Viitoare (Roadmap)
* **Modele Fine-Tuned Specifice pe Sport:** Antrenarea și integrarea unor ponderi dedicate (`football_yolov8.pt`, `basketball_yolov8.pt`, `handball_yolov8.pt`) cu clase native (`player`, `goalkeeper`, `referee`, `ball`) pentru creșterea ratei de detecție a mingii la distanță și eliminarea automată a arbitrilor din calculul posesiei.
* **Homografie Automată (Pitch Keypoint Detection):** Detectarea automată a marcajelor terenului (linii de careu, cerc median, colțuri) folosind un model de tip *pose/keypoints* pentru eliminarea calibrării manuale pe cadre cu cameră mobilă.
* **Interogare Tactică în Limbaj Natural (Natural Language Video Querying):** Căutare semantică peste sesiunile procesate (ex: *"Arată-mi toate atacurile pe flancul drept din ultimele 5 minute"*).