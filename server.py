import os, io, csv, re, json, zipfile, datetime, threading, time
from typing import Tuple, List, Dict, Optional
import requests
from fastapi import FastAPI, Body, Request
from fastapi.staticfiles import StaticFiles
from bs4 import BeautifulSoup

# =========================
# App & paths
# =========================
app = FastAPI(title="Professor Aurelius DSA Actions API")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(ROOT_DIR, "files")
os.makedirs(FILES_DIR, exist_ok=True)

# Exponera statiska filer via /files/...
app.mount("/files", StaticFiles(directory=FILES_DIR), name="files")

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

def _base_url(request: Optional[Request]=None) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    if request is not None:
        return str(request.base_url).rstrip("/")
    return ""

# =========================
# Helpers (GitHub, parsing)
# =========================
def _gh_headers():
    token = os.environ.get("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"token {token}"
    return h

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
def analyze_github_repo(payload: dict = Body(...)):
    """
    Input: {"repo_url":"https://github.com/org/repo[/tree/<ref>/<path>]",
            "extensions":[".py",".md",".pdf",".ipynb"], "max_files":400}
    """
    repo_url = payload.get("repo_url", "").strip()
    if not repo_url:
        return {"status": "error", "message": "Provide repo_url"}
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
    Output: plan (fördelning MD/PY)
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

    topics.sort(key=lambda t: (t[1]!="md", -t[2], t[0]))  # md först; tyngst först
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
            "focus": "Konsolidera teori (MD) och öva implementation (PY).",
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

# =========================
# Labbar (Chalmers GitLab)
# =========================
@app.post("/fetch_lab_repo")
def fetch_lab_repo(request: Request, payload: dict = Body(...)):
    """
    Hämtar specifik labb (ZIP) och extraherar lokalt.
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
    Kolla lab-1..lab-N, ladda ner nya om de finns och saknas lokalt.
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
    Läs README.md + .py i /files/labs/<lab-N>/ och skapa <lab-N>_summary.json
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
    Starta bakgrundsjobb som kör auto_check_labs var 'interval_days' och analyserar nya labbar.
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

# =========================
# Tentor
# =========================
@app.post("/analyze_exams_repo")
def analyze_exams_repo(payload: dict = Body(...)):
    """
    Hämta pdf-tentor från past-exams-repo och bygg exam_summary.json
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

# =========================
# Kursmaterial (HTML/CSV) -> summary + studieplan
# =========================
@app.post("/analyze_course_material")
def analyze_course_material(payload: dict = Body(...)):
    """
    Läs .html/.csv i /files och skapa course_summary.json
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
    Skapa studieplan från course_summary.json
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

# =========================
# Obsidian-note export
# =========================
@app.post("/create_obsidian_note")
def create_obsidian_note(request: Request, payload: dict = Body(...)):
    """
    Spara en Obsidian-kompatibel .md-fil
    Input: {
      "title":"Binary Search – korrekthet",
      "body_md":"<Hela markdown-innehållet (inkl YAML-frontmatter om du vill)>"
    }
    Output: {"status":"ok","note_url":"..."}
    """
    title = (payload.get("title") or "note").strip()
    body = payload.get("body_md") or ""
    safe = "".join(c for c in title if c.isalnum() or c in (" ","-","_")).strip().replace(" ", "_")
    notes_dir = os.path.join(FILES_DIR, "obsidian_notes")
    os.makedirs(notes_dir, exist_ok=True)
    fname = f"{safe}.md"
    fpath = os.path.join(notes_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(body)
    return {"status":"ok","note_url": f"{_base_url(request)}/files/obsidian_notes/{fname}"}

# =========================
# Discord daily digest (valfritt)
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

def _render_digest_md(day_label: str, collected: list[dict]):
    lines = [f"# Discord-digest {day_label}", ""]
    for block in collected:
        ch = block["channel"]
        msgs = block["messages"]
        lines.append(f"## Kanal {ch}")
        if not msgs:
            lines.append("_Inga nya meddelanden._")
        for m in msgs:
            ts = m.get("timestamp","")[:19].replace("T"," ")
            author = (m.get("author") or {}).get("username","unknown")
            content = (m.get("content") or "").replace("\r","").strip() or "[Bifogning/Embed]"
            lines.append(f"- **{ts} – {author}:** {content}")
        lines.append("")
    return "\n".join(lines)

@app.post("/discord_set_channels")
def discord_set_channels(payload: dict = Body(...)):
    """Input: {"channel_ids": ["123","456"]} eller sätt env DISCORD_CHANNEL_IDS."""
    st = _load_state()
    chan = payload.get("channel_ids") or []
    if not isinstance(chan, list) or not chan:
        env_ids = os.environ.get("DISCORD_CHANNEL_IDS", "")
        chan = [c.strip() for c in env_ids.split(",") if c.strip()]
    st["channels"] = chan
    _save_state(st)
    return {"status": "ok", "channels": st["channels"]}

@app.post("/discord_fetch_latest")
def discord_fetch_latest(request: Request, payload: dict = Body(...)):
    """
    Hämta nya meddelanden från konfigurerade kanaler och spara dags-digest.
    Input: {"max_per_channel": 100}
    """
    st = _load_state()
    channels = st.get("channels", [])
    if not channels:
        env_ids = os.environ.get("DISCORD_CHANNEL_IDS", "")
        channels = [c.strip() for c in env_ids.split(",") if c.strip()]
        st["channels"] = channels
    if not channels:
        return {"status":"error","message":"No channels configured. Call /discord_set_channels first."}

    max_per = int(payload.get("max_per_channel", 100))
    collected = []
    for ch in channels:
        since_id = (st.get("last_since") or {}).get(ch)
        res = _discord_fetch_channel(ch, since_id=since_id, limit=max_per)
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

    base = _base_url(request)
    return {
        "status":"ok",
        "digest_md": f"{base}/files/discord/{md_name}",
        "digest_json": f"{base}/files/discord/{json_name}",
        "channels": channels
    }

def _seconds_until_next(time_str: str, tz_name: str = "Europe/Stockholm"):
    from zoneinfo import ZoneInfo
    hour, minute = map(int, time_str.split(":"))
    now = datetime.datetime.now(ZoneInfo(tz_name))
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run += datetime.timedelta(days=1)
    return (next_run - now).total_seconds()

@app.post("/discord_schedule_daily")
def discord_schedule_daily(payload: dict = Body(...)):
    """Input: {"time":"08:00","max_per_channel":100} – schemalägg daglig digest."""
    at = payload.get("time", "08:00")
    max_per = int(payload.get("max_per_channel", 100))

    def worker():
        while True:
            wait = _seconds_until_next(at, "Europe/Stockholm")
            time.sleep(wait)
            try:
                discord_fetch_latest({"max_per_channel": max_per})
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True).start()
    return {"status":"ok","message":f"Daily Discord digest scheduled at {at} Europe/Stockholm"}

# =========================
# Run (Render uses: python server.py)
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
