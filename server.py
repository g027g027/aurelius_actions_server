# server.py
import os
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

app = FastAPI(
    title="Professor Aurelius Actions API",
    description=(
        "Backend för Jana/Analys-aktionsendpoints: "
        "kalender, flashcards, tentastatistik, simulera tenta, exportera bevis, "
        "veckorapport, kunskapsluckor och quiz-rättare."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PUBLIC_BASE_URL = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
FILES_DIR = "/tmp/aurelius_files"
os.makedirs(FILES_DIR, exist_ok=True)

def public_url(request: Request, filename: str) -> str:
    base = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    return f"{base}/files/{filename}"

@app.get("/")
def root():
    return {"status": "ok", "service": "Professor Aurelius Actions API"}

@app.get("/files/{filename}")
def get_file(filename: str):
    path = os.path.join(FILES_DIR, filename)
    if not os.path.exists(path):
        return {"error": "file_not_found"}
    return FileResponse(path)

# ---------- 1) schedule_study ----------
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
    created_ids = [f"evt_{i+1}" for i, _ in enumerate(payload.pass_)]
    return {"message": "Studiepass registrerade (demo).", "created_ids": created_ids}

# ---------- 2) generate_flashcards ----------
import csv
class FlashReq(BaseModel):
    topic: str
    par: List[List[str]]

@app.post("/generate_flashcards")
def generate_flashcards(payload: FlashReq, request: Request):
    safe_topic = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in payload.topic)
    filename = f"{safe_topic}_flashcards.csv"
    path = os.path.join(FILES_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Framsida", "Baksida"])
        for pair in payload.par:
            if len(pair) >= 2:
                w.writerow([pair[0], pair[1]])
    return {"message": f"Skapade {len(payload.par)} flashcards", "download_url": public_url(request, filename)}

# ---------- 3) tenta_stats (lazy imports) ----------
class TentaEntry(BaseModel):
    år: int
    datum: Optional[str] = None
    kategori: str
    poäng: Optional[float] = None

class TentaStatsReq(BaseModel):
    entries: List[TentaEntry]

@app.post("/tenta_stats")
def tenta_stats(payload: TentaStatsReq, request: Request):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame([e.model_dump() for e in payload.entries])
    if df.empty:
        return {"sammanfattning": "Inga data", "per_kategori": {}, "trendbild_url": None}

    per_cat = df["kategori"].value_counts().to_dict()

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

    common = max(per_cat, key=per_cat.get)
    summary = f"{int(df.shape[0])} uppgifter; {len(per_cat)} kategorier. Vanligast: {common}."
    return {"sammanfattning": summary, "per_kategori": per_cat, "trendbild_url": public_url(request, fname)}

# ---------- 4) generate_exam ----------
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
    "gränsvärden": "Bestäm gränsvärdet (utan L'Hôpital): …",
    "derivata_integral": "Beräkna (a) derivata, (b) primitiv/integral: …",
    "graf": "Rita grafen till f(x)=…; ange asymptoter, extrempunkter och konvexitet.",
    "bevis": "Formulera och bevisa en vald sats (t.ex. Medelvärdessatsen eller Analysens huvudsats).",
    "koncept": "Sant/Falskt med motivering om kontinuitet/deriverbarhet/primitivitet.",
}

@app.post("/generate_exam")
def generate_exam(payload: ExamReq):
    b = payload.balans or Balans()
    pieces: List[str] = []
    def add(cat: str, n: int):
        for _ in range(max(0, n)):
            pieces.append(TEMPLATES[cat])
    add("gränsvärden", b.gränsvärden)
    add("derivata_integral", b.derivata_integral)
    add("graf", b.graf)
    add("bevis", b.bevis)
    add("koncept", b.koncept)
    return {"titel": "Simulerad tenta i Jana-stil", "uppgifter": pieces, "pdf_url": None}

# ---------- 5) export_proof (lazy import reportlab) ----------
class ExportProofReq(BaseModel):
    titel: str
    latex: str

@app.post("/export_proof")
def export_proof(payload: ExportProofReq, request: Request):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    safe_title = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in payload.titel)
    fname = f"{safe_title}.pdf"
    path = os.path.join(FILES_DIR, fname)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path)
    story = [
        Paragraph(payload.titel, styles["Title"]),
        Spacer(1, 12),
        Paragraph("LaTeX-källa (kompilera i Overleaf för TeX-typografi):", styles["BodyText"]),
        Spacer(1, 6),
        Paragraph(payload.latex.replace("\n", "<br/>"), styles["Code"]),
    ]
    doc.build(story)

    return {"message": "PDF skapad (innehåller LaTeX-källan som text).", "pdf_url": public_url(request, fname)}

# ---------- 6) weekly_report ----------
class WeeklyReq(BaseModel):
    period: str
    aktiviteter: Optional[List[str]] = None
    mål: Optional[List[str]] = None

@app.post("/weekly_report")
def weekly_report(payload: WeeklyReq):
    aktiviteter = payload.aktiviteter or []
    sammanfattning = f"Vecka {payload.period}: {len(aktiviteter)} aktiviteter loggade."
    rekommendationer = [
        "Fortsätt med active recall på bevis.",
        "Lägg in ett grafritningspass under tidspress.",
        "Repetera konvergenskriterierna.",
    ]
    return {"sammanfattning": sammanfattning, "rekommendationer": rekommendationer, "pdf_url": None}

# ---------- 7) gap_analysis ----------
class GapReq(BaseModel):
    textlogg: str
    nyckelord: Optional[List[str]] = None

@app.post("/gap_analysis")
def gap_analysis(payload: GapReq):
    text = payload.textlogg.lower()
    signs = ["vet inte", "osäker", "förstår inte", "förklara igen"]
    hits = {k: text.count(k) for k in signs}
    problems = [k for k, v in hits.items() if v > 0]
    suggestions: List[str] = []
    if "gränsvärde" in text or "gränsvärden" in text:
        suggestions.append("Träna epsilon–delta och standardgränsvärden utan L'Hôpital.")
    if "integral" in text:
        suggestions.append("Gör 3 uppgifter med partiell integration och substitution.")
    return {"problemområden": problems, "förslag": suggestions}

# ---------- 8) grade_quiz ----------
class QuizReq(BaseModel):
    frågor: Optional[List[str]] = None
    rätt_svar: List[str]
    dina_svar: List[str]

@app.post("/grade_quiz")
def grade_quiz(payload: QuizReq):
    n = min(len(payload.rätt_svar), len(payload.dina_svar))
    rätt = 0
    felindex: List[int] = []
    for i in range(n):
        if str(payload.dina_svar[i]).strip() == str(payload.rätt_svar[i]).strip():
            rätt += 1
        else:
            felindex.append(i)
    fel = n - rätt
    procent = (rätt / n) * 100 if n else 0.0
    return {"antal_rätt": rätt, "antal_fel": fel, "procent": round(procent, 2), "felindex": felindex}

# --- GitHub repo analyzer (recursive) ---
from fastapi import Body
import requests, re
from urllib.parse import urlparse
from typing import Tuple, List, Dict

GITHUB_TIMEOUT = 20
MAX_FILES = 400  # säkerhetstak så vi inte drar igenom enorma repos

def _gh_headers():
    # Lägg ev. GITHUB_TOKEN i Render → Settings → Environment för högre rate limit
    token = os.environ.get("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"token {token}"
    return h

def _parse_repo_url(repo_url: str) -> Tuple[str, str, str, str]:
    """
    Returnerar (org, repo, ref, path) givet en GitHub URL:
    - https://github.com/org/repo
    - https://github.com/org/repo/tree/main/sub/dir
    """
    if "github.com" not in repo_url:
        raise ValueError("Not a GitHub URL")
    parts = repo_url.split("github.com/")[-1].split("/")
    org, repo = parts[0], parts[1]
    ref, path = "main", ""
    if "/tree/" in repo_url:
        after = repo_url.split("/tree/")[1]
        bits = after.split("/", 1)
        ref = bits[0]
        path = bits[1] if len(bits) > 1 else ""
    return org, repo, ref, path

def _contents_api(org: str, repo: str, path: str = "", ref: str = "main") -> str:
    base = f"https://api.github.com/repos/{org}/{repo}/contents"
    return f"{base}/{path}" if path else base

def _list_recursive(org: str, repo: str, ref: str, start_path: str = "") -> List[Dict]:
    """
    Går rekursivt via GitHub Contents API och returnerar filobjekt.
    Respekterar MAX_FILES.
    """
    stack = [start_path]
    files = []
    headers = _gh_headers()
    seen = set()

    while stack and len(files) < MAX_FILES:
        path = stack.pop()
        api = _contents_api(org, repo, path, ref=ref)
        r = requests.get(api, params={"ref": ref}, headers=headers, timeout=GITHUB_TIMEOUT)
        if r.status_code != 200:
            continue
        items = r.json()
        if isinstance(items, dict) and items.get("type") == "file":
            items = [items]
        for it in items:
            t = it.get("type")
            if t == "dir":
                p = it.get("path")
                if p and p not in seen:
                    seen.add(p)
                    stack.append(p)
            elif t == "file":
                files.append(it)
                if len(files) >= MAX_FILES:
                    break
    return files

def _analyze_text_md(text: str) -> Dict:
    heads = re.findall(r"^#+\s+(.+)$", text, flags=re.M)[:10]
    code_blocks = len(re.findall(r"```", text))
    return {"headings": heads, "code_fences": code_blocks}

def _analyze_python(code: str) -> Dict:
    funcs = len(re.findall(r"^\s*def\s+\w+\(", code, flags=re.M))
    classes = len(re.findall(r"^\s*class\s+\w+\(", code, flags=re.M))
    imports = re.findall(r"^\s*(?:from\s+\w+(?:\.\w+)*\s+import|import\s+\w+(?:\.\w+)*)", code, flags=re.M)[:12]
    return {"functions": funcs, "classes": classes, "imports": imports}

@app.post("/analyze_github_repo")
def analyze_github_repo(payload: dict = Body(...)):
    """
    Input: {"repo_url":"https://github.com/org/repo[/tree/<ref>/<path>]",
            "extensions":[".py",".md",".pdf",".ipynb"] (optional),
            "max_files": 400 (optional)}
    Output: {"status":"ok","count":N,"files":[...]}
    """
    repo_url = payload.get("repo_url", "").strip()
    if not repo_url:
        return {"status": "error", "message": "Missing repo_url"}
    try:
        org, repo, ref, path = _parse_repo_url(repo_url)
    except Exception:
        return {"status": "error", "message": "Invalid GitHub URL"}

    exts = tuple([e.lower() for e in payload.get("extensions", [".py",".md",".pdf",".ipynb"])])
    max_files = int(payload.get("max_files", MAX_FILES))

    # Lista rekursivt
    files = _list_recursive(org, repo, ref, start_path=path)
    # Filtrera på extension
    files = [f for f in files if f.get("type")=="file" and f.get("name","").lower().endswith(exts)]
    files = files[:max_files]

    summary = []
    headers = _gh_headers()

    for f in files:
        name = f.get("name","")
        size = f.get("size")
        dl = f.get("download_url")
        filetype = name.split(".")[-1].lower()
        row = {"name": name, "path": f.get("path"), "size": size, "type": filetype, "download_url": dl}

        # Lättviktsanalys: bara .py och .md hämtas i text (PDF/IPYNB lämnas som länkar)
        try:
            if dl and filetype in ("py","md"):
                r = requests.get(dl, headers=headers, timeout=GITHUB_TIMEOUT)
                if r.status_code == 200:
                    text = r.text
                    if filetype == "py":
                        row.update(_analyze_python(text))
                    elif filetype == "md":
                        row.update(_analyze_text_md(text))
        except Exception:
            pass

        summary.append(row)

    return {
        "status": "ok",
        "count": len(summary),
        "ref": ref,
        "repo": f"{org}/{repo}",
        "root": path or "",
        "files": summary
    }


# ---------- OpenAPI servers patch ----------
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    if PUBLIC_BASE_URL:
        openapi_schema["servers"] = [{"url": PUBLIC_BASE_URL}]
    else:
        openapi_schema["servers"] = [{"url": "/"}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
