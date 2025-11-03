
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import csv, io, base64
import pandas as pd
import matplotlib.pyplot as plt

# Ensure non-interactive backend for matplotlib
import matplotlib
matplotlib.use("Agg")

app = FastAPI(
    title="Professor Aurelius Actions API",
    description="Backend för Jana/Analys-aktionsendpoints: kalender, flashcards, tentastatistik, simulera tenta, exportera bevis, veckorapport, kunskapsluckor, quiz-rättare.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FILES_DIR = "/tmp/aurelius_files"
os.makedirs(FILES_DIR, exist_ok=True)

def public_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/files/{filename}"

@app.get("/files/{filename}")
def get_file(filename: str):
    path = os.path.join(FILES_DIR, filename)
    if not os.path.exists(path):
        return {"error": "file_not_found"}
    return FileResponse(path)

# ---- 1) schedule_study ----
class Studiepass(BaseModel):
    ämne: str
    datum: str
    starttid: str
    sluttid: str
    beskrivning: Optional[str] = None

class ScheduleRequest(BaseModel):
    pass_: List[Studiepass] = Field(alias="pass")

@app.post("/schedule_study")
def schedule_study(payload: ScheduleRequest):
    # Demo: we don't integrate real calendar here; we simulate creation IDs.
    created_ids = [f"evt_{i+1}" for i, _ in enumerate(payload.pass_)]
    return {"message": "Studiepass registrerade (demo). Koppla till Google Calendar om du vill göra detta på riktigt.", "created_ids": created_ids}

# ---- 2) generate_flashcards ----
class FlashReq(BaseModel):
    topic: str
    par: List[List[str]]  # [front, back]

@app.post("/generate_flashcards")
def generate_flashcards(payload: FlashReq, request: Request):
    filename = f"{payload.topic.replace(' ', '_')}_flashcards.csv"
    path = os.path.join(FILES_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Framsida", "Baksida"])
        for pair in payload.par:
            if len(pair) >= 2:
                writer.writerow([pair[0], pair[1]])
    return {"message": f"Skapade {len(payload.par)} flashcards", "download_url": public_url(request, filename)}

# ---- 3) tenta_stats ----
class TentaEntry(BaseModel):
    år: int
    datum: Optional[str] = None
    kategori: str
    poäng: Optional[float] = None

class TentaStatsReq(BaseModel):
    entries: List[TentaEntry]

@app.post("/tenta_stats")
def tenta_stats(payload: TentaStatsReq, request: Request):
    df = pd.DataFrame([e.model_dump() for e in payload.entries])
    if df.empty:
        return {"sammanfattning": "Inga data", "per_kategori": {}, "trendbild_url": None}
    per_cat = df["kategori"].value_counts().to_dict()
    # Simple bar chart
    plt.figure()
    pd.Series(per_cat).plot(kind="bar")
    plt.title("Frekvens av uppgiftstyper")
    plt.xlabel("Kategori")
    plt.ylabel("Antal")
    plt.tight_layout()
    fname = "tentastatistik.png"
    fpath = os.path.join(FILES_DIR, fname)
    plt.savefig(fpath)
    plt.close()
    summary = f"{int(df.shape[0])} uppgifter; {len(per_cat)} kategorier. Vanligast: {max(per_cat, key=per_cat.get)}."
    return {"sammanfattning": summary, "per_kategori": per_cat, "trendbild_url": public_url(request, fname)}

# ---- 4) generate_exam ----
class Balans(BaseModel):
    gränsvärden: int = 2
    derivata_integral: int = 2
    graf: int = 1
    bevis: int = 1
    koncept: int = 2

class ExamReq(BaseModel):
    år: Optional[int] = None
    balans: Optional[Balans] = None

TEMPLATES = {
    "gränsvärden": "Bestäm gränsvärdet (utan L'Hôpital): ...",
    "derivata_integral": "Beräkna (a) derivata, (b) primitiv/integral: ...",
    "graf": "Rita grafen till f(x)=..., ange asymptoter, extrempunkter och konvexitet.",
    "bevis": "Formulera och bevisa satsen: ... (t.ex. Medelvärdessatsen eller Analysens huvudsats).",
    "koncept": "Sant/Falskt: påståenden om kontinuitet, derivata, primitivitet. Motivera kort."
}

@app.post("/generate_exam")
def generate_exam(payload: ExamReq):
    b = payload.balans or Balans()
    pieces = []
    def add(cat, n):
        for _ in range(n):
            pieces.append(TEMPLATES[cat])
    add("gränsvärden", b.gränsvärden)
    add("derivata_integral", b.derivata_integral)
    add("graf", b.graf)
    add("bevis", b.bevis)
    add("koncept", b.koncept)
    return {"titel": "Simulerad tenta i Jana-stil", "uppgifter": pieces, "pdf_url": None}

# ---- 5) export_proof ----
class ExportProofReq(BaseModel):
    titel: str
    latex: str

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

@app.post("/export_proof")
def export_proof(payload: ExportProofReq, request: Request):
    fname = payload.titel.replace(" ", "_") + ".pdf"
    path = os.path.join(FILES_DIR, fname)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path)
    story = [Paragraph(payload.titel, styles["Title"]), Spacer(1, 12),
             Paragraph("LaTeX-källa (kompilera i Overleaf om du vill ha TeX-typsatt matematik):", styles["BodyText"]),
             Spacer(1, 6),
             Paragraph(payload.latex.replace("\n", "<br/>"), styles["Code"])]
    doc.build(story)
    return {"message": "PDF skapad (visar LaTeX-källan).", "pdf_url": public_url(request, fname)}

# ---- 6) weekly_report ----
class WeeklyReq(BaseModel):
    period: str
    aktiviteter: Optional[List[str]] = None
    mål: Optional[List[str]] = None

@app.post("/weekly_report")
def weekly_report(payload: WeeklyReq):
    aktiviteter = payload.aktiviteter or []
    mål = payload.mål or []
    sammanfattning = f"Vecka {payload.period}: {len(aktiviteter)} aktiviteter loggade."
    rekommendationer = [
        "Fortsätt med active recall på bevis.",
        "Lägg in ett grafritningspass under tidspress.",
        "Repetera konvergenskriterierna."
    ]
    return {"sammanfattning": sammanfattning, "rekommendationer": rekommendationer, "pdf_url": None}

# ---- 7) gap_analysis ----
class GapReq(BaseModel):
    textlogg: str
    nyckelord: Optional[List[str]] = None

@app.post("/gap_analysis")
def gap_analysis(payload: GapReq):
    text = payload.textlogg.lower()
    signs = ["vet inte", "osäker", "förstår inte", "förklara igen"]
    hits = {k: text.count(k) for k in signs}
    problems = [k for k, v in hits.items() if v > 0]
    suggestions = []
    if "gränsvärde" in text or "gränsvärden" in text:
        suggestions.append("Träna epsilon–delta och standardgränsvärden utan L'Hôpital.")
    if "integral" in text:
        suggestions.append("Gör 3 uppgifter med partiell integration och substitution.")
    return {"problemområden": problems, "förslag": suggestions}

# ---- 8) grade_quiz ----
class QuizReq(BaseModel):
    frågor: Optional[List[str]] = None
    rätt_svar: List[str]
    dina_svar: List[str]

@app.post("/grade_quiz")
def grade_quiz(payload: QuizReq):
    n = min(len(payload.rätt_svar), len(payload.dina_svar))
    rätt = 0
    felindex = []
    for i in range(n):
        if str(payload.dina_svar[i]).strip() == str(payload.rätt_svar[i]).strip():
            rätt += 1
        else:
            felindex.append(i)
    fel = n - rätt
    procent = (rätt / n) * 100 if n else 0.0
    return {"antal_rätt": rätt, "antal_fel": fel, "procent": round(procent, 2), "felindex": felindex}

# Health check
@app.get("/")
def root():
    return {"status": "ok", "service": "Professor Aurelius Actions API"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
