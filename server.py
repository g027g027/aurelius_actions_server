import os, io, csv, re, json, zipfile, datetime, threading, time
from typing import Tuple, List, Dict, Optional
import requests
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from pypdf import PdfReader
# =========================
# FastAPI app (incl. servers for GPT import)
# =========================
app = FastAPI(
    title="Professor Aurelius DSA Actions API",
    version="1.0.0",
    servers=[{"url": "https://aurelius-actions-server.onrender.com"}]
)

# =========================
# Paths & static
# =========================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(ROOT_DIR, "files")
os.makedirs(FILES_DIR, exist_ok=True)

# Serve static output files
app.mount("/files", StaticFiles(directory=FILES_DIR), name="files")

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

TOPIC_KEYWORDS = {
    "sorting": ["sort", "quicksort", "mergesort", "heapsort", "insertion", "selection"],
    "trees": ["tree", "bst", "avl", "red-black", "splay"],
    "heaps": ["heap", "priority queue"],
    "hashing": ["hash", "hashtable", "hash table"],
    "graphs": ["graph", "bfs", "dfs", "dijkstra", "mst", "kruskal", "prim", "topological"],
    "complexity": ["big-o", "o(", "omega(", "theta(", "time complexity", "space complexity"],
    "dp": ["dynamic programming", "recurrence", "memoization", "tabulation"],
    "greedy": ["greedy", "exchange argument"],
}

QUESTION_PATTERNS = re.compile(r"\b(question|problem|uppgift)\b", re.IGNORECASE)

def _extract_pdf_info(local_path: str) -> dict:
    info = {"pages": None, "questions_count": None, "topics": [], "text_extracted": False}
    try:
        reader = PdfReader(local_path)
        info["pages"] = len(reader.pages)
        # Försök enkel textextraktion (kan misslyckas för scannade pdf:er)
        text_chunks = []
        for i, page in enumerate(reader.pages[:20]):  # cap:a första 20 sidor
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                text_chunks.append(txt)
        full = "\n".join(text_chunks)
        if full.strip():
            info["text_extracted"] = True
            info["questions_count"] = len(QUESTION_PATTERNS.findall(full))
            # Topic heuristics (case-insensitiv)
            low = full.lower()
            topics = []
            for topic, keys in TOPIC_KEYWORDS.items():
                if any(k.lower() in low for k in keys):
                    topics.append(topic)
            info["topics"] = sorted(set(topics))
        else:
            info["questions_count"] = None
    except Exception:
        pass
    return info


def _base_url(request: Optional[Request] = None) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    if request is not None:
        return str(request.base_url).rstrip("/")
    return ""

# =========================
# Pydantic input models (for clear OpenAPI)
# =========================
class GenericRepoInput(BaseModel):
    repo_url: str = Field(..., example="https://github.com/ChalmersGU-data-structure-courses/dsabook")
    extensions: Optional[List[str]] = Field(default=[".py", ".md", ".pdf", ".ipynb"])
    max_files: Optional[int] = 400

class StudyPlanInput(BaseModel):
    repo_url: str
    weeks: int = 6
    hours_per_week: int = 8

class ExamRepoInput(BaseModel):
    repo_url: str = Field(..., example="https://github.com/ChalmersGU-data-structure-courses/past-exams")
    max_exams: int = 20

class LabFetchInput(BaseModel):
    lab_number: int = Field(..., ge=1)
    year: int = Field(default=datetime.datetime.now().year)
    term: str = Field(default="lp2", description="lp1/lp2")

class AutoLabCheckInput(BaseModel):
    year: int = Field(default=datetime.datetime.now().year)
    term: str = "lp2"
    max_labs: int = 4

class AnalyzeLabInput(BaseModel):
    lab_name: str = Field(..., example="lab-2")

class ScheduleLabInput(BaseModel):
    year: int = Field(default=datetime.datetime.now().year)
    term: str = "lp2"
    max_labs: int = 4
    interval_days: int = 14

class CourseMaterialInput(BaseModel):
    course_name: str = "Data Structures and Algorithms"

class StudyplanGenInput(BaseModel):
    weeks: int = 6
    hours_per_week: int = 10
    focus: List[str] = ["exam_topics", "labs_info", "modules"]

class ObsidianNoteInput(BaseModel):
    title: str
    body_md: str

class DiscordChannelsInput(BaseModel):
    channel_ids: Optional[List[str]] = []

class DiscordFetchInput(BaseModel):
    max_per_channel: int = 100

class DiscordScheduleInput(BaseModel):
    time: str = "08:00"
    max_per_channel: int = 100

# NEW: import files via URL (Option B)
class ImportFilesInput(BaseModel):
    urls: List[str] = Field(..., description="List of absolute URLs to fetch and store under /files")

# =========================
# Helpers (GitHub, parsing) — UPDATED: Trees API + Raw + Retries
# =========================
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "Aurelius-DSA-Actions/1.0 (+https://render.com)"
DEFAULT_TIMEOUT = (5, 25)           # (connect, read)
MAX_FILES_DEFAULT = 300
MAX_BYTES_PER_FILE = 200_000        # 200 kB per file when fetching text

# One shared session with retries/backoff
SESSION = requests.Session()
_retry = Retry(
    total=5,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD"]
)
SESSION.mount("https://", HTTPAdapter(max_retries=_retry))
SESSION.headers.update({"User-Agent": USER_AGENT})

def _gh_headers():
    token = os.environ.get("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    if token:
        h["Authorization"] = f"token {token}"
    return h

def _gh_get(url: str, params: dict | None = None):
    r = SESSION.get(url, headers=_gh_headers(), params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r

def _parse_repo_url(repo_url: str) -> Tuple[str, str, str, str]:
    """Return (org, repo, ref, path) for https://github.com/org/repo[/tree/<ref>/<path>]"""
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

def _list_via_tree(org: str, repo: str, ref: str) -> List[Dict]:
    """List the entire repository tree in a single call via Git Trees API."""
    url = f"https://api.github.com/repos/{org}/{repo}/git/trees/{ref}"
    r = _gh_get(url, params={"recursive": "1"})
    data = r.json()
    return data.get("tree", [])  # entries: {'path','type'('blob'|'tree'),'sha',...}

def _raw_url(org: str, repo: str, ref: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{org}/{repo}/{ref}/{path}"

def _blob_url(org: str, repo: str, ref: str, path: str) -> str:
    return f"https://github.com/{org}/{repo}/blob/{ref}/{path}"

def _safe_fetch_text(url: str, max_bytes: int = MAX_BYTES_PER_FILE) -> str:
    """Fetch text with a byte cap to avoid huge files/timeouts."""
    with SESSION.get(url, stream=True, timeout=DEFAULT_TIMEOUT) as r:
        r.raise_for_status()
        chunks, total = [], 0
        for chunk in r.iter_content(chunk_size=8192):
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")

# --- analyzers (unchanged) ---
def _analyze_text_md(text: str) -> Dict:
    heads = re.findall(r"^#+\s+(.+)$", text, flags=re.M)[:10]
    code_blocks = len(re.findall(r"```", text))
    return {"headings": heads, "code_fences": code_blocks}

def _analyze_python(code: str) -> Dict:
    funcs = len(re.findall(r"^\s*def\s+\w+\(", code, flags=re.M))
    classes = len(re.findall(r"^\s*class\s+\w+\(", code, flags=re.M))
    imports = re.findall(r"^\s*(?:from\s+[\w\.]+\s+import|import\s+[\w\.]+)", code, flags=re.M)[:12]
    return {"functions": funcs, "classes": classes, "imports": imports}

# =========================
# Root
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "Professor Aurelius DSA Actions API"}

# =========================
# Repo / GitHub
# =========================
@app.post("/analyze_github_repo")
def analyze_github_repo(payload: GenericRepoInput):
    repo_url = payload.repo_url.strip()
    try:
        org, repo, ref, subpath = _parse_repo_url(repo_url)
    except Exception:
        return {"status": "error", "message": "Invalid GitHub URL"}

    exts = tuple([e.lower() for e in (payload.extensions or [".py",".md",".pdf",".ipynb"])])
    max_files = int(payload.max_files or MAX_FILES_DEFAULT)

    tree = _list_via_tree(org, repo, ref)
    def in_sub(p: str) -> bool:
        return (not subpath) or p == subpath or p.startswith(subpath.rstrip("/") + "/")
    blobs = [t for t in tree if t.get("type")=="blob" and t["path"].lower().endswith(exts) and in_sub(t["path"])]
    blobs = blobs[:max_files]

    out = []
    for t in blobs:
        path = t["path"]; name = os.path.basename(path); kind = name.split(".")[-1].lower()
        raw = _raw_url(org, repo, ref, path)
        row = {"name": name, "path": path, "type": kind, "download_url": raw, "web_url": _blob_url(org, repo, ref, path)}
        if kind in ("py","md"):
            try:
                text = _safe_fetch_text(raw)
                row.update(_analyze_python(text) if kind=="py" else _analyze_text_md(text))
            except Exception:
                pass
        out.append(row)

    return {"status":"ok","repo":f"{org}/{repo}","ref":ref,"root":subpath or "", "count":len(out), "files": out}

@app.post("/summarize_repo_for_studyplan")
def summarize_repo_for_studyplan(payload: StudyPlanInput):
    repo_url = payload.repo_url.strip()
    weeks = int(payload.weeks or 6)
    hours = int(payload.hours_per_week or 8)
    try:
        org, repo, ref, subpath = _parse_repo_url(repo_url)
    except Exception:
        return {"status":"error","message":"Invalid GitHub URL"}

    tree = _list_via_tree(org, repo, ref)
    def in_sub(p: str) -> bool:
        return (not subpath) or p == subpath or p.startswith(subpath.rstrip("/") + "/")
    cand = [t for t in tree if t.get("type")=="blob" and in_sub(t["path"]) and t["path"].lower().endswith((".md",".py"))][:MAX_FILES_DEFAULT]

    topics = []
    for t in cand:
        path = t["path"]; name = os.path.basename(path).lower()
        raw = _raw_url(org, repo, ref, path)
        try:
            text = _safe_fetch_text(raw)
        except Exception:
            continue
        if name.endswith(".md"):
            info = _analyze_text_md(text)
            w = min(3, max(1, len(info.get("headings",[]))//3))
            topics.append((path, "md", w, {"headings": info.get("headings",[])[:8]}))
        else:
            info = _analyze_python(text)
            w = min(3, max(1, (info.get("functions",0)+info.get("classes",0))//3))
            topics.append((path, "py", max(1,w), {"functions": info.get("functions",0), "classes": info.get("classes",0)}))

    if not topics:
        return {"status":"ok","repo":f"{org}/{repo}","ref":ref,"weeks":weeks,"hours_per_week":hours,"plan":[],"note":"No MD/PY topics found."}

    topics.sort(key=lambda t: (t[1]!="md", -t[2], t[0]))
    buckets = [[] for _ in range(weeks)]
    for i, t in enumerate(topics):
        buckets[i % weeks].append(t)

    plan = []
    for w in range(weeks):
        items = []
        N = max(1, len(buckets[w]))
        for path_item, kind, weight, meta in buckets[w]:
            entry = {
                "path": path_item,
                "type": "markdown" if kind=="md" else "python",
                "suggested_hours": max(1, weight * (hours // N)),
                "meta": meta
            }
            if kind == "md" and meta.get("headings"):
                entry["flashcard_suggestions"] = meta["headings"]
            elif kind == "py":
                fc = meta.get("functions",0)+meta.get("classes",0)
                entry["practice_suggestions"] = [f"Skriv tester för {fc} funktioner/klasser"]
            items.append(entry)
        plan.append({"week": w+1, "total_hours": hours, "items": items})

    return {"status":"ok","repo":f"{org}/{repo}","ref":ref,"weeks":weeks,"hours_per_week":hours,"plan":plan}

@app.post("/list_past_exams")
def list_past_exams(payload: GenericRepoInput):
    repo_url = payload.repo_url.strip()
    try:
        org, repo, ref, subpath = _parse_repo_url(repo_url)
    except Exception:
        return {"status":"error","message":"Invalid GitHub URL"}

    exam_pats = {"exam.pdf","midterm.pdf","final.pdf"}
    sol_pats  = {"solutions.pdf","solution.pdf","answers.pdf","key.pdf"}

    tree = _list_via_tree(org, repo, ref)
    def in_sub(p: str) -> bool:
        return (not subpath) or p == subpath or p.startswith(subpath.rstrip("/") + "/")
    files = [t for t in tree if t.get("type")=="blob" and in_sub(t["path"]) and t["path"].lower().endswith(".pdf")]

    by_dir = {}
    for t in files:
        path = t["path"]
        folder = path.rsplit("/",1)[0] if "/" in path else ""
        name = os.path.basename(path).lower()
        if name in exam_pats:
            by_dir.setdefault(folder, {})["exam_url"] = _raw_url(org, repo, ref, path)
            by_dir[folder]["exam_web"] = _blob_url(org, repo, ref, path)
        if name in sol_pats:
            by_dir.setdefault(folder, {})["solution_url"] = _raw_url(org, repo, ref, path)
            by_dir[folder]["solution_web"] = _blob_url(org, repo, ref, path)

    rows = [{"folder":k, **v} for k,v in sorted(by_dir.items())]
    return {"status":"ok","count":len(rows),"items":rows}

@app.post("/create_flashcards_from_repo")
def create_flashcards_from_repo(request: Request, payload: GenericRepoInput):
    repo_url = payload.repo_url.strip()
    topic = "repo_flashcards"
    back_tmpl = "Förklara kort och ge ett exempel."
    max_files = int(payload.max_files or 200)

    try:
        org, repo, ref, subpath = _parse_repo_url(repo_url)
    except Exception:
        return {"status":"error","message":"Invalid GitHub URL"}

    tree = _list_via_tree(org, repo, ref)
    def in_sub(p: str) -> bool:
        return (not subpath) or p == subpath or p.startswith(subpath.rstrip("/") + "/")
    md_blobs = [t for t in tree if t.get("type")=="blob" and in_sub(t["path"]) and t["path"].lower().endswith(".md")][:max_files]

    pairs = []
    for t in md_blobs:
        raw = _raw_url(org, repo, ref, t["path"])
        try:
            text = _safe_fetch_text(raw)
            info = _analyze_text_md(text)
            for h in info.get("headings", []):
                pairs.append([h, back_tmpl])
        except Exception:
            continue

    safe_topic = "".join(c if c.isalnum() or c in ("_","-") else "_" for c in topic)
    fname = f"{safe_topic}_flashcards.csv"
    fpath = os.path.join(FILES_DIR, fname)
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Framsida","Baksida"])
        for a,b in pairs:
            w.writerow([a,b])

    base = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    return {"status":"ok","count":len(pairs),"download_url":f"{base}/files/{fname}"}


# =========================
# Labbar (Chalmers GitLab)
# =========================
@app.post("/fetch_lab_repo")
def fetch_lab_repo(request: Request, payload: LabFetchInput):
    """
    Hämtar specifik labb (ZIP) och extraherar lokalt.
    """
    lab_number = int(payload.lab_number)
    year = int(payload.year)
    term = payload.term.lower()

    base_url = f"https://git.chalmers.se/courses/data-structures/{term}/{year}/lab-{lab_number}"
    zip_url = f"{base_url}/-/archive/main/lab-{lab_number}-main.zip"

    labs_dir = os.path.join(FILES_DIR, "labs")
    os.makedirs(labs_dir, exist_ok=True)
    zip_path = os.path.join(labs_dir, f"lab-{lab_number}.zip")
    extract_path = os.path.join(labs_dir, f"lab-{lab_number}")

    try:
        r = requests.get(zip_url, timeout=25)
        if r.status_code != 200:
            return {"status": "error", "message": f"Lab {lab_number} not available yet (HTTP {r.status_code})"}
        with open(zip_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            os.makedirs(extract_path, exist_ok=True)
            z.extractall(extract_path)
        return {
            "status": "ok",
            "lab_number": lab_number,
            "lab_url": base_url,
            "saved_zip": f"{_base_url(request)}/files/labs/lab-{lab_number}.zip",
            "saved_dir": f"/files/labs/lab-{lab_number}/"
        }
    except Exception as e:
        return {"status": "error", "message": f"Download failed: {e}"}

@app.post("/auto_check_labs")
def auto_check_labs(payload: AutoLabCheckInput):
    """
    Kolla lab-1..lab-N, ladda ner nya om de finns och saknas lokalt.
    """
    year = int(payload.year)
    term = payload.term.lower()
    max_labs = int(payload.max_labs)

    labs_dir = os.path.join(FILES_DIR, "labs")
    os.makedirs(labs_dir, exist_ok=True)

    downloaded, already, not_available = [], [], []
    for n in range(1, max_labs + 1):
        lab_name = f"lab-{n}"
        base_url = f"https://git.chalmers.se/courses/data-structures/{term}/{year}/{lab_name}"
        zip_url = f"{base_url}/-/archive/main/{lab_name}-main.zip"

        local_zip = os.path.join(labs_dir, f"{lab_name}.zip")
        if os.path.exists(local_zip):
            already.append(lab_name)
            continue

        try:
            r = requests.head(zip_url, timeout=10)
            if r.status_code != 200:
                not_available.append(lab_name)
                continue
            r = requests.get(zip_url, timeout=25)
            if r.status_code == 200:
                with open(local_zip, "wb") as f:
                    f.write(r.content)
                extract_path = os.path.join(labs_dir, lab_name)
                os.makedirs(extract_path, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    z.extractall(extract_path)
                downloaded.append(lab_name)
            else:
                not_available.append(lab_name)
        except Exception:
            not_available.append(lab_name)

    return {"status": "ok", "downloaded": downloaded, "already_have": already, "not_available_yet": not_available}

@app.post("/analyze_lab_files")
def analyze_lab_files(payload: AnalyzeLabInput):
    """
    Läs README.md + .py i /files/labs/<lab-N>/ och skapa <lab-N>_summary.json
    """
    lab_name = payload.lab_name
    lab_path = os.path.join(FILES_DIR, "labs", lab_name)
    if not os.path.isdir(lab_path):
        return {"status": "error", "message": f"{lab_name} not found locally"}

    summary = {"lab": lab_name, "readme_summary": {}, "code_summary": []}
    readme_path = os.path.join(lab_path, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            text = f.read()
        info = _analyze_text_md(text)
        summary["readme_summary"] = {"headings": info.get("headings", []), "code_blocks": info.get("code_fences", 0)}

    for fname in os.listdir(lab_path):
        if fname.endswith(".py"):
            with open(os.path.join(lab_path, fname), "r", encoding="utf-8") as f:
                info = _analyze_python(f.read())
            summary["code_summary"].append({"file": fname, **info})

    outpath = os.path.join(FILES_DIR, "labs", f"{lab_name}_summary.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return {"status": "ok", "summary_file": f"/files/labs/{lab_name}_summary.json"}

@app.post("/schedule_lab_check")
def schedule_lab_check(payload: ScheduleLabInput):
    """
    Starta bakgrundsjobb som kör auto_check_labs var 'interval_days' och analyserar nya labbar.
    """
    year = int(payload.year)
    term = payload.term.lower()
    max_labs = int(payload.max_labs)
    interval_days = int(payload.interval_days)

    def job():
        while True:
            result = auto_check_labs(AutoLabCheckInput(year=year, term=term, max_labs=max_labs))
            for lab in result.get("downloaded", []):
                analyze_lab_files(AnalyzeLabInput(lab_name=lab))
            time.sleep(interval_days * 24 * 3600)

    threading.Thread(target=job, daemon=True).start()
    return {"status": "ok", "message": f"Lab check scheduled every {interval_days} days."}

# =========================
# Tentor
# =========================
@app.post("/analyze_exams_repo")
def analyze_exams_repo(payload: ExamRepoInput):
    """
    Hämtar pdf-tentor från past-exams-repo, sparar lokalt och bygger exam_summary.json
    Input: {"repo_url":"https://github.com/.../past-exams","max_exams":40}
    """
    repo_url = payload.repo_url.strip()
    max_exams = int(payload.max_exams or 40)
    if not repo_url:
        return {"status":"error","message":"Missing repo_url"}
    try:
        org, repo, ref, subpath = _parse_repo_url(repo_url)
    except Exception:
        return {"status":"error","message":"Invalid GitHub URL"}

    tree = _list_via_tree(org, repo, ref)

    def in_sub(p: str) -> bool:
        return (not subpath) or p.startswith(subpath.rstrip("/") + "/") or p == subpath

    pdf_nodes = [t for t in tree if t.get("type")=="blob" and in_sub(t["path"]) and t["path"].lower().endswith(".pdf")]
    # Gruppér per mapp (en mapp = en tenta)
    by_dir: Dict[str, Dict[str, str]] = {}
    for t in pdf_nodes:
        path = t["path"]
        folder = path.rsplit("/",1)[0] if "/" in path else ""
        name = os.path.basename(path).lower()
        raw = _raw_url(org, repo, ref, path)
        web = _blob_url(org, repo, ref, path)
        by_dir.setdefault(folder, {})
        if name in ("exam.pdf", "exam 2025-10-29.pdf") or name.startswith("exam"):
            by_dir[folder]["exam_url"] = raw
            by_dir[folder]["exam_web"] = web
            by_dir[folder]["exam_name"] = os.path.basename(path)
        if name in ("solutions.pdf", "solutions 2025-10-29.pdf", "solution.pdf", "answers.pdf", "key.pdf") or name.startswith("solution"):
            by_dir[folder]["solution_url"] = raw
            by_dir[folder]["solution_web"] = web
            by_dir[folder]["solution_name"] = os.path.basename(path)

    # Hämta lokalt + analysera
    exams_dir = os.path.join(FILES_DIR, "exams")
    os.makedirs(exams_dir, exist_ok=True)

    summary_items = []
    topic_freq: Dict[str,int] = {}
    q_counts: List[int] = []

    # Begränsa antal mappar
    folders = sorted(by_dir.keys())[:max_exams]
    for folder in folders:
        record = {"folder": folder}
        rec = by_dir[folder]

        # EXAM
        exam_local = None
        if "exam_url" in rec:
            exam_name = rec.get("exam_name", os.path.basename(rec["exam_url"]))
            exam_local = os.path.join(exams_dir, f"{folder.replace('/','_')}__{exam_name}")
            if not os.path.exists(exam_local):
                try:
                    with SESSION.get(rec["exam_url"], timeout=DEFAULT_TIMEOUT, stream=True) as r:
                        r.raise_for_status()
                        with open(exam_local, "wb") as fp:
                            for chunk in r.iter_content(8192):
                                if chunk: fp.write(chunk)
                except Exception:
                    exam_local = None
            record["exam_url"] = rec["exam_url"]
            record["exam_web"] = rec.get("exam_web")

        # SOLUTION
        if "solution_url" in rec:
            sol_name = rec.get("solution_name", os.path.basename(rec["solution_url"]))
            sol_local = os.path.join(exams_dir, f"{folder.replace('/','_')}__{sol_name}")
            if not os.path.exists(sol_local):
                try:
                    with SESSION.get(rec["solution_url"], timeout=DEFAULT_TIMEOUT, stream=True) as r:
                        r.raise_for_status()
                        with open(sol_local, "wb") as fp:
                            for chunk in r.iter_content(8192):
                                if chunk: fp.write(chunk)
                except Exception:
                    pass
            record["solution_url"] = rec["solution_url"]
            record["solution_web"] = rec.get("solution_web")

        # ANALYS på exam
        if exam_local and os.path.exists(exam_local):
            info = _extract_pdf_info(exam_local)
            record.update(info)
            if info.get("questions_count") is not None:
                q_counts.append(info["questions_count"])
            for tpc in info.get("topics", []):
                topic_freq[tpc] = topic_freq.get(tpc, 0) + 1

        summary_items.append(record)

    # Aggregat
    agg = {
        "exam_count": len(summary_items),
        "topic_frequency": dict(sorted(topic_freq.items(), key=lambda x: (-x[1], x[0]))),
        "questions_per_exam": {
            "values": q_counts,
            "mean": (sum(q_counts)/len(q_counts)) if q_counts else None
        }
    }

    out = {"status":"ok", "repo": f"{org}/{repo}", "ref": ref, "items": summary_items, "aggregate": agg}

    outpath = os.path.join(exams_dir, "exam_summary.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return {"status":"ok","count":len(summary_items),"summary_file":"/files/exams/exam_summary.json"}


# =========================
# Kursmaterial (HTML/CSV) -> summary + studieplan
# =========================
@app.post("/analyze_course_material")
def analyze_course_material(payload: CourseMaterialInput):
    """
    Läs .html/.csv i /files och skapa course_summary.json
    """
    course_name = payload.course_name
    html_files = [f for f in os.listdir(FILES_DIR) if f.endswith(".html")]
    csv_files = [f for f in os.listdir(FILES_DIR) if f.endswith(".csv")]
    summary = {"course": course_name, "modules": [], "exam_topics": [], "labs_info": [], "literature": [], "schedule": []}

    for fname in html_files:
        path = os.path.join(FILES_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        title = soup.title.string if soup.title else fname.replace(".html", "")
        headers = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
        links = [a["href"] for a in soup.find_all("a", href=True)]

        if "Exam" in title or "examination" in title or "Written examination" in title:
            summary["exam_topics"].append({"file": fname, "headers": headers, "paragraphs": paragraphs})
        elif "Lab" in title or "lab" in title:
            summary["labs_info"].append({"file": fname, "headers": headers, "paragraphs": paragraphs})
        elif "Literature" in title or "reading" in title:
            summary["literature"].append({"file": fname, "paragraphs": paragraphs, "links": links})
        else:
            summary["modules"].append({"file": fname, "headers": headers, "paragraphs": paragraphs})

    for fname in csv_files:
        path = os.path.join(FILES_DIR, fname)
        with open(path, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                summary["schedule"].append(row)

    outpath = os.path.join(FILES_DIR, "course_summary.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {"status": "ok", "summary_file": "/files/course_summary.json",
            "topics_found": sum(len(m.get("headers", [])) for m in summary["modules"])}

@app.post("/generate_studyplan_from_summary")
def generate_studyplan_from_summary(payload: StudyplanGenInput):
    """
    Skapa studieplan från course_summary.json
    """
    weeks = int(payload.weeks)
    hours_per_week = int(payload.hours_per_week)
    focus = payload.focus or ["exam_topics", "labs_info"]

    summary_path = os.path.join(FILES_DIR, "course_summary.json")
    if not os.path.exists(summary_path):
        return {"status": "error", "message": "course_summary.json not found. Run /analyze_course_material first."}

    with open(summary_path, "r", encoding="utf-8") as f:
        course_data = json.load(f)

    topics = []
    if "exam_topics" in focus:
        for t in course_data.get("exam_topics", []):
            topics.extend(t.get("headers", []))
    if "modules" in focus:
        for m in course_data.get("modules", []):
            topics.extend(m.get("headers", []))
    if "labs_info" in focus:
        for l in course_data.get("labs_info", []):
            topics.extend(l.get("headers", []))

    total_topics = len(topics)
    per_week = max(1, total_topics // weeks) if total_topics else 1
    chunks = [topics[i:i + per_week] for i in range(0, total_topics, per_week)] or [[]]

    schedule = course_data.get("schedule", [])
    labs = course_data.get("labs_info", [])

    plan = []
    for i in range(weeks):
        week_topics = chunks[i] if i < len(chunks) else []
        week_labs = []
        for l in labs:
            if len(week_labs) < 1:
                week_labs.append(l.get("headers", ["Labb"])[0] if l.get("headers") else "Labb")
        week_schedule = [row for row in schedule if str(i + 1) in row.get("Week", "")]
        plan.append({"week": i + 1, "topics": week_topics, "labs": week_labs, "hours": hours_per_week, "schedule": week_schedule})

    csv_path = os.path.join(FILES_DIR, "studyplan.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Week", "Hours", "Topics", "Labs"])
        for w in plan:
            writer.writerow([w["week"], w["hours"], "; ".join(w["topics"]), "; ".join(w["labs"])])

    return {"status": "ok", "plan_file": "/files/studyplan.csv", "summary": plan}

# =========================
# NEW: Import course files via URL (Option B)
# =========================
@app.post("/import_course_files")
def import_course_files(payload: ImportFilesInput, request: Request):
    """
    Hämta en lista av filer via URL och spara dem i /files.
    Stöder bl.a. .html, .csv, .md, .pdf (andra filer skrivs också, men använd med omdöme).
    Input: {"urls":["https://.../file1.html","https://.../schedule.csv", ...]}
    """
    allowed = {".html", ".csv", ".md", ".pdf"}
    saved, errors = [], []

    for u in payload.urls:
        try:
            r = requests.get(u, timeout=30)
            if r.status_code != 200:
                errors.append({"url": u, "error": f"HTTP {r.status_code}"})
                continue

            name = (u.split("/")[-1] or "file").split("?")[0].split("#")[0]
            safe = "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " ")).strip().replace(" ", "_")
            if not safe:
                safe = "file"

            ext = "." + safe.split(".")[-1].lower() if "." in safe else ""
            # (valfri strikt kontroll) if allowed and ext and ext not in allowed: continue

            fpath = os.path.join(FILES_DIR, safe)
            with open(fpath, "wb") as f:
                f.write(r.content)

            saved.append(f"{_base_url(request)}/files/{safe}")
        except Exception as e:
            errors.append({"url": u, "error": str(e)})

    return {"status": "ok", "saved": saved, "errors": errors}

# =========================
# Discord daily digest (optional)
# =========================
DISCORD_STATE_DIR = os.path.join(FILES_DIR, "discord")
os.makedirs(DISCORD_STATE_DIR, exist_ok=True)
DISCORD_STATE_FILE = os.path.join(DISCORD_STATE_DIR, "state.json")

def _discord_headers():
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN is not set")
    return {"Authorization": f"Bot {token}"}

def _load_state():
    if os.path.exists(DISCORD_STATE_FILE):
        with open(DISCORD_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"channels": [], "last_since": {}}

def _save_state(st):
    with open(DISCORD_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)

def _discord_fetch_channel(channel_id: str, since_id: Optional[str] = None, limit: int = 100):
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    params = {"limit": min(limit, 100)}
    if since_id:
        params["after"] = since_id
    r = requests.get(url, headers=_discord_headers(), params=params, timeout=20)
    if r.status_code != 200:
        return {"channel": channel_id, "status": "error", "http": r.status_code, "messages": []}
    msgs = r.json()
    msgs = sorted(msgs, key=lambda m: m.get("id"))
    return {"channel": channel_id, "status": "ok", "messages": msgs}

def _render_digest_md(day_label: str, collected: List[Dict]):
    lines = [f"# Discord-digest {day_label}", ""]
    for block in collected:
        ch = block["channel"]
        msgs = block["messages"]
        lines.append(f"## Kanal {ch}")
        if not msgs:
            lines.append("_Inga nya meddelanden._")
        for m in msgs:
            ts = m.get("timestamp", "")[:19].replace("T", " ")
            author = (m.get("author") or {}).get("username", "unknown")
            content = (m.get("content") or "").replace("\r", "").strip() or "[Bifogning/Embed]"
            lines.append(f"- **{ts} – {author}:** {content}")
        lines.append("")
    return "\n".join(lines)

def _discord_fetch_latest_core(max_per_channel: int) -> Dict:
    """Core logic for fetching + saving digest; callable from endpoint or scheduler."""
    st = _load_state()
    channels = st.get("channels", [])
    if not channels:
        env_ids = os.environ.get("DISCORD_CHANNEL_IDS", "")
        channels = [c.strip() for c in env_ids.split(",") if c.strip()]
        st["channels"] = channels
    if not channels:
        return {"status": "error", "message": "No channels configured."}

    collected = []
    for ch in channels:
        since_id = (st.get("last_since") or {}).get(ch)
        res = _discord_fetch_channel(ch, since_id=since_id, limit=max_per_channel)
        if res["status"] == "ok" and res["messages"]:
            last_id = res["messages"][-1]["id"]
            st.setdefault("last_since", {})[ch] = last_id
        collected.append(res)
    _save_state(st)

    today = datetime.datetime.now(datetime.timezone.utc).astimezone().date().isoformat()
    md = _render_digest_md(today, collected)
    md_name = f"digest-{today}.md"
    md_path = os.path.join(DISCORD_STATE_DIR, md_name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    json_name = f"digest-{today}.json"
    json_path = os.path.join(DISCORD_STATE_DIR, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(collected, f, indent=2, ensure_ascii=False)

    return {
        "status": "ok",
        "digest_md": f"/files/discord/{md_name}",
        "digest_json": f"/files/discord/{json_name}",
        "channels": channels
    }

@app.post("/discord_set_channels")
def discord_set_channels(payload: DiscordChannelsInput):
    """Sätt kanaler som ska läsas."""
    st = _load_state()
    chan = payload.channel_ids or []
    if not isinstance(chan, list) or not chan:
        env_ids = os.environ.get("DISCORD_CHANNEL_IDS", "")
        chan = [c.strip() for c in env_ids.split(",") if c.strip()]
    st["channels"] = chan
    _save_state(st)
    return {"status": "ok", "channels": st["channels"]}

@app.post("/discord_fetch_latest")
def discord_fetch_latest(request: Request, payload: DiscordFetchInput):
    """
    Hämta nya meddelanden från konfigurerade kanaler och spara dags-digest.
    """
    res = _discord_fetch_latest_core(payload.max_per_channel)
    base = _base_url(request)
    if res.get("status") == "ok":
        res["digest_md"] = f"{base}{res['digest_md']}"
        res["digest_json"] = f"{base}{res['digest_json']}"
    return res

def _seconds_until_next(time_str: str, tz_name: str = "Europe/Stockholm"):
    from zoneinfo import ZoneInfo
    hour, minute = map(int, time_str.split(":"))
    now = datetime.datetime.now(ZoneInfo(tz_name))
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run += datetime.timedelta(days=1)
    return (next_run - now).total_seconds()

@app.post("/discord_schedule_daily")
def discord_schedule_daily(payload: DiscordScheduleInput):
    """Schemalägg daglig digest."""
    at = payload.time
    max_per = int(payload.max_per_channel)

    def worker():
        while True:
            wait = _seconds_until_next(at, "Europe/Stockholm")
            time.sleep(wait)
            try:
                _discord_fetch_latest_core(max_per)
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True).start()
    return {"status": "ok", "message": f"Daily Discord digest scheduled at {at} Europe/Stockholm"}

# =========================
# Run
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
