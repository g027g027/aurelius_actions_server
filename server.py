import os, io, csv, re, json, zipfile, datetime, threading, time
from typing import Tuple, List, Dict
import requests
from fastapi import FastAPI, Body, Request
from fastapi.staticfiles import StaticFiles
from bs4 import BeautifulSoup

# ---------------------------
# App & env
# ---------------------------
app = FastAPI(title="Professor Aurelius Actions API")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(ROOT_DIR, "files")
os.makedirs(FILES_DIR, exist_ok=True)

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "")  # t.ex. https://<din-service>.onrender.com

# Exponera statiska filer (CSV/PNG/PDF/JSON/ZIP) via /files/...
app.mount("/files", StaticFiles(directory=FILES_DIR), name="files")

# ---------------------------
# Helpers
# ---------------------------
def _base_url(request: Request) -> str:
    """Returnera publik bas-URL (env om satt, annars från inkommande request)."""
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL.rstrip("/")
    return str(request.base_url).rstrip("/")

def _gh_headers():
    token = os.environ.get("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"token {token}"
    return h

def _parse_repo_url(repo_url: str) -> Tuple[str, str, str, str]:
    """Returnerar (org, repo, ref, path) för https://github.com/org/repo[/tree/<ref>/<path>]"""
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

def _contents_api(org: str, repo: str) -> str:
    return f"https://api.github.com/repos/{org}/{repo}/contents"

def _list_recursive(org: str, repo: str, ref: str, start_path: str = "", max_files: int = 400) -> List[Dict]:
    """Rekursivt via GitHub Contents API; respekterar max_files."""
    stack = [start_path]
    files = []
    headers = _gh_headers()
    seen = set()
    while stack and len(files) < max_files:
        path = stack.pop()
        api = _contents_api(org, repo)
        url = f"{api}/{path}" if path else api
        r = requests.get(url, params={"ref": ref}, headers=headers, timeout=20)
        if r.status_code != 200:
            continue
        items = r.json()
        if isinstance(items, dict) and items.get("type") == "file":
            items = [items]
        for it in items:
            if it.get("type") == "dir":
                p = it.get("path")
                if p and p not in seen:
                    seen.add(p)
                    stack.append(p)
            elif it.get("type") == "file":
                files.append(it)
                if len(files) >= max_files:
                    break
    return files

def _analyze_text_md(text: str) -> Dict:
    heads = re.findall(r"^#+\s+(.+)$", text, flags=re.M)[:10]
    code_blocks = len(re.findall(r"```", text))
    return {"headings": heads, "code_fences": code_blocks}

def _analyze_python(code: str) -> Dict:
    funcs = len(re.findall(r"^\s*def\s+\w+\(", code, flags=re.M))
    classes = len(re.findall(r"^\s*class\s+\w+\(", code, flags=re.M))
    imports = re.findall(r"^\s*(?:from\s+[\w\.]+\s+import|import\s+[\w\.]+)", code, flags=re.M)[:12]
    return {"functions": funcs, "classes": classes, "imports": imports}

# ---------------------------
# Root
# ---------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "Professor Aurelius Actions API"}

# ---------------------------
# GitHub / Repo-analys
# ---------------------------
@app.post("/analyze_github_repo")
def analyze_github_repo(payload: dict = Body(...)):
    """
    Input: {"repo_url":"https://github.com/org/repo[/tree/<ref>/<path>]",
            "extensions":[".py",".md",".pdf",".ipynb"], "max_files":400}
    """
    repo_url = payload.get("repo_url", "").strip()
    if not repo_url:
        return {"status": "error", "message": "Provide a valid GitHub URL."}
    try:
        org, repo, ref, path = _parse_repo_url(repo_url)
    except Exception:
        return {"status": "error", "message": "Invalid GitHub URL"}

    exts = tuple([e.lower() for e in payload.get("extensions", [".py",".md",".pdf",".ipynb"])])
    max_files = int(payload.get("max_files", 400))

    items = _list_recursive(org, repo, ref, start_path=path, max_files=max_files)
    files = [it for it in items if it.get("type")=="file" and it["name"].lower().endswith(exts)]

    summary = []
    headers = _gh_headers()
    for f in files:
        name = f["name"]
        url = f.get("download_url")
        kind = name.split(".")[-1].lower()
        row = {"name": name, "path": f.get("path"), "size": f.get("size"), "type": kind, "download_url": url}
        if url and kind in ("py", "md"):
            try:
                text = requests.get(url, headers=headers, timeout=20).text
                if kind == "py":
                    row.update(_analyze_python(text))
                elif kind == "md":
                    row.update(_analyze_text_md(text))
            except Exception:
                pass
        summary.append(row)

    return {"status": "ok", "count": len(summary), "repo": f"{org}/{repo}", "ref": ref, "root": path or "", "files": summary}

@app.post("/summarize_repo_for_studyplan")
def summarize_repo_for_studyplan(payload: dict = Body(...)):
    """
    Input: {"repo_url":"...","weeks":6,"hours_per_week":8}
    Output: fördelning av MD/PY över veckor + rekommendationer
    """
    repo_url = payload.get("repo_url","").strip()
    weeks = int(payload.get("weeks", 6))
    hours = int(payload.get("hours_per_week", 8))
    if not repo_url:
        return {"status":"error","message":"Missing repo_url"}
    try:
        org, repo, ref, path = _parse_repo_url(repo_url)
    except Exception:
        return {"status":"error","message":"Invalid GitHub URL"}

    items = _list_recursive(org, repo, ref, start_path=path)
    files = [f for f in items if f.get("type")=="file" and f.get("name","").lower().endswith((".md",".py"))][:400]

    topics = []
    headers = _gh_headers()
    for f in files:
        dl = f.get("download_url")
        name = f["name"].lower()
        if not dl: 
            continue
        try:
            text = requests.get(dl, headers=headers, timeout=20).text
        except Exception:
            continue
        if name.endswith(".md"):
            info = _analyze_text_md(text)
            w = min(3, max(1, len(info.get("headings",[]))//3))
            topics.append((f["path"], "md", w, {"headings": info.get("headings",[])[:8]}))
        else:
            info = _analyze_python(text)
            w = min(3, max(1, (info.get("functions",0)+info.get("classes",0))//3))
            topics.append((f["path"], "py", max(1,w), {"functions": info.get("functions",0), "classes": info.get("classes",0)}))

    if not topics:
        return {"status":"ok","weeks":weeks,"hours_per_week":hours,"plan":[],"note":"No MD/PY topics found."}

    topics.sort(key=lambda t: (t[1]!="md", -t[2], t[0]))  # md före py; tungt först
    buckets = [[] for _ in range(weeks)]
    for i, t in enumerate(topics):
        buckets[i % weeks].append(t)

    plan = []
    for w in range(weeks):
        entries = []
        for path_item, kind, weight, meta in buckets[w]:
            entry = {
                "path": path_item,
                "type": "markdown" if kind=="md" else "python",
                "suggested_hours": max(1, weight * (hours // max(1, len(buckets[w])))),
                "meta": meta
            }
            if kind == "md" and meta.get("headings"):
                entry["flashcard_suggestions"] = meta["headings"]
            elif kind == "py":
                fc = meta.get("functions",0)+meta.get("classes",0)
                entry["practice_suggestions"] = [f"Skapa tester för {fc} funktioner/klasser"]
            entries.append(entry)

        plan.append({
            "week": w+1,
            "total_hours": hours,
            "items": entries,
            "focus": "Konsolidera teori (MD) och implementera/öva kod (PY).",
            "quiz_ideas": ["Begreppsquiz", "Komplexitet/Edge-cases"]
        })

    return {"status":"ok","repo": f"{org}/{repo}", "ref": ref, "weeks": weeks, "hours_per_week": hours, "plan": plan}

@app.post("/list_past_exams")
def list_past_exams(payload: dict = Body(...)):
    """
    Input: {"repo_url":".../past-exams[/tree/<ref>/<path>]",
            "exam_patterns":["exam.pdf","midterm.pdf","final.pdf"],
            "solution_patterns":["solutions.pdf","solution.pdf","answers.pdf","key.pdf"]}
    """
    repo_url = payload.get("repo_url","").strip()
    exam_pats = set([p.lower() for p in payload.get("exam_patterns", ["exam.pdf"])])
    sol_pats = set([p.lower() for p in payload.get("solution_patterns", ["solutions.pdf","solution.pdf","answers.pdf","key.pdf"])])
    if not repo_url:
        return {"status":"error","message":"Missing repo_url"}
    try:
        org, repo, ref, path = _parse_repo_url(repo_url)
    except Exception:
        return {"status":"error","message":"Invalid GitHub URL"}

    items = _list_recursive(org, repo, ref, start_path=path)
    by_dir = {}
    for f in items:
        if f.get("type") != "file":
            continue
        p = f.get("path","")
        folder = p.rsplit("/",1)[0] if "/" in p else ""
        name = f.get("name","").lower()
        if name in exam_pats:
            by_dir.setdefault(folder, {})["exam_url"] = f.get("download_url")
        if name in sol_pats:
            by_dir.setdefault(folder, {})["solution_url"] = f.get("download_url")

    rows = [{"folder":k, **v} for k,v in sorted(by_dir.items())]
    return {"status":"ok","count":len(rows),"items":rows}

@app.post("/create_flashcards_from_repo")
def create_flashcards_from_repo(request: Request, payload: dict = Body(...)):
    """
    Input: {"repo_url":"...","topic":"DSA-boken","back_template":"Definiera ..."}
    Output: {"download_url":"/files/<topic>_flashcards.csv","count":N}
    """
    repo_url = payload.get("repo_url","").strip()
    topic = payload.get("topic","repo_flashcards")
    back_tmpl = payload.get("back_template","Förklara kort och ge ett exempel.")
    if not repo_url:
        return {"status":"error","message":"Missing repo_url"}

    try:
        org, repo, ref, path = _parse_repo_url(repo_url)
    except Exception:
        return {"status":"error","message":"Invalid GitHub URL"}

    items = _list_recursive(org, repo, ref, start_path=path)
    md_files = [f for f in items if f.get("type")=="file" and f.get("name","").lower().endswith(".md")][:200]

    headers = _gh_headers()
    pairs = []
    for f in md_files:
        dl = f.get("download_url")
        if not dl:
            continue
        try:
            text = requests.get(dl, headers=headers, timeout=20).text
            info = _analyze_text_md(text)
            for h in info.get("headings", []):
                pairs.append([h, back_tmpl])
        except Exception:
            continue

    safe_topic = "".join(c if c.isalnum() or c in ("_","-") else "_" for c in topic)
    fname = f"{safe_topic}_flashcards.csv"
    fpath = os.path.join(FILES_DIR, fname)
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Framsida","Baksida"])
        for a,b in pairs:
            w.writerow([a,b])

    download_url = f"{_base_url(request)}/files/{fname}"
    return {"status":"ok","count":len(pairs),"download_url":download_url}

# ---------------------------
# Labbar (Chalmers GitLab)
# ---------------------------
@app.post("/fetch_lab_repo")
def fetch_lab_repo(request: Request, payload: dict = Body(...)):
    """
    Hämtar en specifik labb (ZIP) och extraherar lokalt.
    Input: {"lab_number":1,"year":2025,"term":"lp2"}
    """
    lab_number = int(payload.get("lab_number", 0))
    year = int(payload.get("year", datetime.datetime.now().year))
    term = payload.get("term", "lp2").lower()
    if not lab_number:
        return {"status": "error", "message": "Missing lab_number"}

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
def auto_check_labs(payload: dict = Body(...)):
    """
    Kollar lab-1..lab-N, laddar ner nya om de finns och inte redan finns lokalt.
    Input: {"year":2025,"term":"lp2","max_labs":4}
    """
    year = int(payload.get("year", datetime.datetime.now().year))
    term = payload.get("term", "lp2").lower()
    max_labs = int(payload.get("max_labs", 4))

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
def analyze_lab_files(payload: dict = Body(...)):
    """
    Läser README.md + .py i /files/labs/<lab-N>/ och skapar <lab-N>_summary.json
    Input: {"lab_name":"lab-2"}
    """
    lab_name = payload.get("lab_name")
    if not lab_name:
        return {"status":"error","message":"Missing lab_name"}

    lab_path = os.path.join(FILES_DIR, "labs", lab_name)
    if not os.path.isdir(lab_path):
        return {"status":"error","message":f"{lab_name} not found locally"}

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
    return {"status":"ok","summary_file":f"/files/labs/{lab_name}_summary.json"}

@app.post("/schedule_lab_check")
def schedule_lab_check(payload: dict = Body(...)):
    """
    Startar bakgrundsjobb som kör auto_check_labs var 'interval_days' och analyserar nya labbar.
    Input: {"year":2025,"term":"lp2","max_labs":4,"interval_days":14}
    """
    year = int(payload.get("year", datetime.datetime.now().year))
    term = payload.get("term", "lp2").lower()
    max_labs = int(payload.get("max_labs", 4))
    interval_days = int(payload.get("interval_days", 14))

    def job():
        while True:
            result = auto_check_labs({"year":year, "term":term, "max_labs":max_labs})
            for lab in result.get("downloaded", []):
                analyze_lab_files({"lab_name": lab})
            time.sleep(interval_days * 24 * 3600)

    threading.Thread(target=job, daemon=True).start()
    return {"status":"ok","message":f"Lab check scheduled every {interval_days} days."}

# ---------------------------
# Tentor
# ---------------------------
@app.post("/analyze_exams_repo")
def analyze_exams_repo(payload: dict = Body(...)):
    """
    Hämtar pdf-tentor från past-exams-repo och bygger exam_summary.json
    Input: {"repo_url":"https://github.com/.../past-exams","max_exams":20}
    """
    repo_url = payload.get("repo_url","").strip()
    max_exams = int(payload.get("max_exams", 20))
    if not repo_url:
        return {"status":"error","message":"Missing repo_url"}
    try:
        org, repo, ref, path = _parse_repo_url(repo_url)
    except Exception:
        return {"status":"error","message":"Invalid repo_url"}

    items = _list_recursive(org, repo, ref, start_path=path)
    pdfs = [f for f in items if f.get("type")=="file" and f.get("name","").lower().endswith(".pdf")][:max_exams]

    exams_dir = os.path.join(FILES_DIR, "exams")
    os.makedirs(exams_dir, exist_ok=True)

    headers = _gh_headers()
    summary = []
    for f in pdfs:
        name, url = f["name"], f["download_url"]
        local_path = os.path.join(exams_dir, name)
        if not os.path.exists(local_path) and url:
            try:
                r = requests.get(url, headers=headers, timeout=25)
                if r.status_code == 200:
                    with open(local_path, "wb") as fp:
                        fp.write(r.content)
            except Exception:
                continue
        summary.append({"name": name, "path": f.get("path"), "size": f.get("size"), "url": url})

    outpath = os.path.join(exams_dir, "exam_summary.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return {"status":"ok","count":len(summary),"summary_file":"/files/exams/exam_summary.json"}

# ---------------------------
# Kursmaterial (HTML/CSV) → summary + studieplan
# ---------------------------
@app.post("/analyze_course_material")
def analyze_course_material(payload: dict = Body(...)):
    """
    Läser alla .html/.csv i /files och skapar course_summary.json
    Input: {"course_name":"Data Structures and Algorithms"}
    """
    course_name = payload.get("course_name", "Data Structures and Algorithms")
    html_files = [f for f in os.listdir(FILES_DIR) if f.endswith(".html")]
    csv_files = [f for f in os.listdir(FILES_DIR) if f.endswith(".csv")]
    summary = {"course": course_name, "modules": [], "exam_topics": [], "labs_info": [], "literature": [], "schedule": []}

    for fname in html_files:
        path = os.path.join(FILES_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        title = soup.title.string if soup.title else fname.replace(".html", "")
        headers = [h.get_text(strip=True) for h in soup.find_all(["h1","h2","h3"])]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True))>40]
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

    return {"status":"ok","summary_file":"/files/course_summary.json",
            "topics_found": sum(len(m.get("headers",[])) for m in summary["modules"])}

@app.post("/generate_studyplan_from_summary")
def generate_studyplan_from_summary(payload: dict = Body(...)):
    """
    Skapar studieplan från course_summary.json
    Input: {"weeks":6,"hours_per_week":10,"focus":["exam_topics","labs_info","modules"]}
    """
    weeks = int(payload.get("weeks", 6))
    hours_per_week = int(payload.get("hours_per_week", 10))
    focus = payload.get("focus", ["exam_topics", "labs_info"])

    summary_path = os.path.join(FILES_DIR, "course_summary.json")
    if not os.path.exists(summary_path):
        return {"status":"error","message":"course_summary.json not found. Run /analyze_course_material first."}

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
    chunks = [topics[i:i+per_week] for i in range(0, total_topics, per_week)] or [[]]

    schedule = course_data.get("schedule", [])
    labs = course_data.get("labs_info", [])

    plan = []
    for i in range(weeks):
        week_topics = chunks[i] if i < len(chunks) else []
        week_labs = []
        for l in labs:
            if len(week_labs) < 1:
                week_labs.append(l.get("headers", ["Labb"])[0] if l.get("headers") else "Labb")
        week_schedule = [row for row in schedule if str(i+1) in row.get("Week","")]
        plan.append({"week": i+1, "topics": week_topics, "labs": week_labs, "hours": hours_per_week, "schedule": week_schedule})

    csv_path = os.path.join(FILES_DIR, "studyplan.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Week", "Hours", "Topics", "Labs"])
        for w in plan:
            writer.writerow([w["week"], w["hours"], "; ".join(w["topics"]), "; ".join(w["labs"])])

    return {"status":"ok","plan_file":"/files/studyplan.csv","summary":plan}
