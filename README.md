# Professor Aurelius DSA Actions API

🚀 **Render-ready FastAPI backend** for the *Data Structures & Algorithms GPT*.  
Implements endpoints for:
- 🔍 Analyzing GitHub repos (`dsabook`, `past-exams`)
- 📘 Parsing course HTML & CSV files
- 🧪 Fetching and auto-checking Chalmers GitLab labs
- 🧩 Generating study plans & flashcards
- 🗒️ Creating Obsidian-compatible notes
- 💬 Fetching Discord announcements (optional)

---

## 🌐 Deploy on Render

### 1. Environment variables

| Key | Example value | Purpose |
|-----|----------------|----------|
| `PYTHON_VERSION` | `3.11.9` | Ensures consistent runtime |
| `PUBLIC_BASE_URL` | `https://aurelius-actions-server.onrender.com` | Used for absolute URLs in responses |
| `GITHUB_TOKEN` *(optional)* | `<your_personal_token>` | Higher GitHub API rate limits |
| `DISCORD_BOT_TOKEN` *(optional)* | `<your_discord_bot_token>` | Enables /discord_ actions |
| `DISCORD_CHANNEL_IDS` *(optional)* | `123,456,789` | Preconfigured Discord channels |

**Build command**
```bash
pip install -r requirements.txt
