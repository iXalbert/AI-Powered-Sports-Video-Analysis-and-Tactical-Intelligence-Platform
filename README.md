# AI-Powered Sports Video Analysis & Tactical Intelligence Platform

An end-to-end computer vision and multimodal AI application that transforms recorded sports footage into player tracking, visual analytics, event statistics, and automated, evidence-grounded tactical summaries.

---

## 1. Problema (Problem Statement)

### Ce rezolvă și cu ce ajută
Analiza meciurilor de sport pe baza înregistrărilor video necesită în mod tradițional un efort manual considerabil din partea antrenorilor și analiștilor pentru extragerea statisticilor, urmărirea jucătorilor și identificarea tiparelor tactice. 

Această platformă automatizează procesul prin procesarea secvențelor video, detectarea și urmărirea jucătorilor (player tracking), extragerea de evenimente-cheie (key events) și generarea automată de rapoarte tactice în limbaj natural.

### Obiectivele Proiectului (3-5 Obiective)
1. **Pipeline de procesare video cap-la-cap:** Construirea unui sistem integrat (Upload $\rightarrow$ Detectare/Tracking $\rightarrow$ Analitică $\rightarrow$ Raport AI) pentru un sport selectat (ex: fotbal, baschet, handbal).
2. **Generare de analiză vizuală și statistici deterministe:** Calcularea mecanică (fără LLM) a minim 3-4 tipuri de evenimente și generarea de vizualizări specifice (mărci de traiectorie, hărți de căldură / heatmaps, timeline de meci).
3. **Sinteză tactică bazată pe ML/LLM/VLM:** Utilizarea modelelor de viziune și limbaj (Vision-Language & Large Language Models) pentru explicarea contextului vizual și redactarea de observații tactice ancorate strict în date reale.
4. **Validare riguroasă și benchmark dedicat:** Evaluarea cantitativă și calitativă a acurateței detecției, a latenței de procesare și a calității rezumatelor pe un set de date controlat și adnotat manual.

---

## 2. Documentație și Stadiul Actual (Related Work & State of the Art)

Sistemele comerciale de analiză sportivă folosesc de obicei echipamente speciale multisenzor sau rețele de camere calibrate (ex: Tracab, Hawk-Eye, Veo). Proiectele open-source și lucrările recente de cercetare se concentrează pe combinarea detectoarelor bazate pe rețele neuronale de tip YOLO cu algoritmi moderni de urmărire multi-obiect (MOT).

### Soluții existente și referințe
* **[YOLOv8 / YOLOv11 (Ultralytics)](https://github.com/ultralytics/ultralytics):** Standardul curent în detecția obiectelor în timp real, ideal pentru detecția jucătorilor și a mingii.
* **[ByteTrack](https://github.com/ifzhang/ByteTrack):** Un tracker multi-obiect extrem de eficient care asociază detecțiile chiar și în cazul ocluziilor temporare sau al scorurilor scăzute de detecție.
* **[Roboflow Supervision](https://github.com/roboflow/supervision):** Set de instrumente reutilizabile pentru procesare de cadre, desenare de overlays, Zone Analytics și numărare de entități.
* **[OpenCV](https://opencv.org/) & [FFmpeg](https://ffmpeg.org/):** Librării fundamentale pentru decodare video, sampling de cadre și manipulare spațio-temporală.
* **Modele VLM / LLM (ex: OpenAI GPT-4o, Google Gemini, Anthropic Claude):** Utilizate pentru interpretarea semantică a cadrelor extras și structurarea rezumatelor explicative.

### Cu ce este mai bun acest proiect?
Spre deosebire de soluțiile clasice de Computer Vision (care oferă doar metrice brute precum hărți de căldură) sau de abordările pur LLM (care pot fi susceptibile la halucinații), această aplicație combină **analitica deterministă** cu **raționamentul multimodal (VLM/LLM)**. Modelul de limbaj primește ca date de intrare doar statistici verified și cadre-cheie, garantând că observațiile tactice sunt complet ancorate în date concrete.

---

## 3. Proiectare (Architecture & Design)

### Arhitectura Sistemului
Aplicația folosește o arhitectură decuplată pe microservicii sau modules, asigurând procesarea asincronă a fișierelor video de dimensiuni mari.

```
[ Client Web / Dashboard ]
           │
           ▼
[ Backend / Orchestrator API (FastAPI) ]
           │
  ┌────────┴───────────────────────────┐
  ▼                                   ▼
[ Video Processing Pipeline ]   [ Data & Results Storage ]
  ├── Frame Sampler & Preprocessor     ├── Raw / Processed Videos
  ├── Detection (YOLO)                 ├── Detections & Tracks (JSON/DB)
  ├── Tracking (ByteTrack)             └── Reports & Timeline
  ├── Analytics Engine (Heatmaps/Events)
  └── Multimodal AI Layer (VLM / LLM)
```

### Tehnologii Propuse
* **Frontend:** React / Next.js / Tailwind CSS (Dashboard interactiv cu player video, timeline și grafice).
* **Backend:** Python (FastAPI / Celery pentru task-uri asincrone).
* **Computer Vision / AI Core:** PyTorch, OpenCV, `ultralytics` (YOLO), `supervision`, ByteTrack.
* **Multimodal / Reasoning:** API-uri OpenAI/Gemini/Anthropic sau modele open-weights rulate local/cloud.
* **Stocare:** SQLite/PostgreSQL pentru date de sesiune și metadata, sistem de fișiere local/S3 pentru cadre și videoclipuri.

### Soluții de Design și Optimizare (2 Soluții Tehnic-Aplicate)
1. **Soluția de procesare asincronă și eșantionare (Sampling):** Pentru a reduce latența și costurile computaționale, video-ul nu este analizat cadru cu cadru la rata întreagă de FPS pentru LLM; se aplică frame-sampling flexibil (ex: 5-10 FPS) pentru modulul CV, iar VLM-ul este apelat exclusiv pe cadre-cheie declanșate de evenimente.
2. **Soluția ancorării datelor (Grounded Tactical Insights):** Prevenirea halucinațiilor LLM prin forțarea generării de output-uri JSON structurate bazate strict pe datele matematice extrapolate (frecvență atacuri, poziții medii, faze de posesie).

---

## 4. Implementare (Implementation Details)

Pipeline-ul principal de analiză este structurat în 9 etape consecutive:

1. **Upload & Session Setup:** Încărcarea fișierului video ($10-15$ minute) și crearea unei sesiuni de analiză.
2. **Decoding & Frame Sampling:** Decodarea prin OpenCV/FFmpeg la o rată configurabilă de cadre.
3. **Player Detection:** Rularea detecției cu un model pre-antrenat (ex: YOLOv8) direct pe cadrele eșantionate.
4. **Multi-Object Tracking:** Asocierea detecțiilor între cadre utilizând ByteTrack și atribuirea de ID-uri unice persistente pentru fiecare jucător.
5. **Spatial & Trajectory Aggregation:** Calcularea coordonatelor spațiale, generarea hărților de căldură (heatmaps) și definirea zonelor de teren.
6. **Rule-Based Event Inference:** Aplicarea unei logici deterministe pentru identificarea a 3-4 evenimente specifice sportului (ex: șuturi, faze de atac pe benzi, schimbări de posesie).
7. **Key-Clip & Timestamp Extraction:** Extragerea automată a cadrelor și secvențelor reprezentative din preajma momentelor importante.
8. **VLM/LLM Interpretation:** Trimiterea evidențelor selectate (date structurate + cadre reprezentative) către modelul AI pentru analiză contextuală.
9. **Dashboard Rendering:** Renders-ul final în interfața grafică (video cu overlay de urmărire, timeline interactiv, componente grafice și raportul tactic generat).

---

## 5. Testare și Evaluare (Testing & Performance)

Evaluarea aplicației se bazează pe experimente controlate folosind un set redus de date adnotat manual (ground-truth benchmark în format JSON/CSV).

### Strategie de Testare
* **Unit Testing:** Teste unitare pentru modulele deterministe de calcul al statisticilor, parsarea metadatelor video, algoritmul de agregare a pozițiilor și generarea structurilor de date JSON.
* **System / End-to-End Testing:** Verificarea pipeline-ului complet de la încărcarea fișierului până la popularea dashboard-ului.

### Evaluări și Metrici (3 Evaluări Directe)
1. **Acuratețea detecției și a evenimentelor (CV Accuracy):** Compararea numărului și a marjei temporale (timestamp tolerance) pentru evenimentele detectate automat față de adnotările manuale din setul de test.
2. **Performanță și Latență (Latency & Throughput):** Măsurarea timpului total de procesare ($E2E$ latency) și a cadrelor procesate pe secundă (FPS) pentru fișiere cu rezoluții și durate diferite (ex: 720p vs 1080p).
3. **Calitatea rezumatelor tactice (Human Rubric Evaluation):** Evaluarea rezumatului generat de VLM/LLM de către un evaluator uman pe baza unei rubrici cu 3 criterii: *Corectitudine factuală*, *Relevanță tactică* și *Utilitate*.

---

## 6. Concluzii (Conclusions)

### Legate de Aplicație
Sistemul dezvoltat demonstrează că se poate construi o platformă funcțională de inteligență tactică sportivă fără a fi necesară antrenarea de la zero a modelelor complexe de viziune. Integrarea modelelor pre-antrenate de detecție/tracking cu puterea de sintetizare a LLM-urilor moderne oferă un instrument util și accesibil pentru analiza meciurilor.

### Ce am învățat
* Integrarea și orchestrarea modelelor moderne de Computer Vision (YOLO, ByteTrack) într-un pipeline continuu.
* Arhitecturarea aplicațiilor asistate de LLM/VLM în care modelele generative sunt folosite strict pe post de strat de raționament explicativ, prevenind halucinațiile prin grounding pe date matematice brute.
* Metodologia de evaluare obiectivă a sistemelor AI hibride prin metrici de latență, acuratețe temporală și evaluare calitativă structurată.

### Ce ne așteptăm / Direcții viitoare (Future Work)
* **Ball Tracking & Pitch Calibration:** Extinderea detecției la minge și maparea coordonatelor din imagine pe un model 2D normalizat al terenului.
* **Natural Language Querying:** Interfață de căutare în limbaj natural (ex: *"Arată-mi atacurile create pe partea dreaptă în repriza a doua"*).
* **Automatic Highlight Reel Generation:** Asamblarea automată a unui clip video restrâns cu cele mai importante momente tactice din meci.