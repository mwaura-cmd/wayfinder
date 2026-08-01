# Wayfinder — Render Deployment Guide

## Prerequisites
- A [GitHub](https://github.com) account (Render deploys from GitHub)
- A [Render](https://render.com) account (free, no credit card required)
- Your project pushed to a GitHub repo

---

## Step 1 — Push to GitHub

```powershell
cd c:\Users\denis\wayfinder

git init
git add .
git commit -m "Initial commit — Wayfinder Research Agent"

# Create a new repo on GitHub (https://github.com/new), then:
git remote add origin https://github.com/YOUR_USERNAME/wayfinder.git
git branch -M main
git push -u origin main
```

> **Confirm your .env is gitignored before pushing:**
> `git status` should NOT show `.env` in the file list.

---

## Step 2 — Generate your WAYFINDER_API_KEY

This is the password that protects your Tavily + Gemini quota from public use.
Run this once and save the output somewhere safe:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Example output: `xK9mP2vQr7nL4hD8wJ3cF6tY1sA5bE0`

---

## Step 3 — Create the Render service

1. Go to → **https://render.com/dashboard**
2. Click **New +** → **Web Service**
3. Connect your GitHub account → select the **wayfinder** repo
4. Render will auto-detect `render.yaml` — click **Apply**

If it doesn't auto-detect, configure manually:
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Plan**: Free

---

## Step 4 — Set environment variables

In the Render dashboard → your service → **Environment** tab:

| Variable | Value |
|----------|-------|
| `OPENROUTER_API_KEY` | Your OpenRouter key from [openrouter.ai/keys](https://openrouter.ai/keys) |
| `OPENROUTER_MODEL` | `openrouter/free` (optional, defaults to auto-routing free tier) |
| `TAVILY_API_KEY` | Your Tavily key from [app.tavily.com](https://app.tavily.com) |
| `WAYFINDER_API_KEY` | The random key you generated in Step 2 |
| `MAX_MEMORY_SESSIONS` | `200` (optional, this is the default) |

Click **Save Changes** → Render will redeploy automatically.

---

## Step 5 — Open your deployed app

Render gives you a URL like:
```
https://wayfinder-xxxx.onrender.com
```

1. Open it in your browser
2. You'll see the **Access key required** modal
3. Enter your `WAYFINDER_API_KEY` → click **Save & Continue**
4. The key is stored in your browser's `localStorage` — you won't need to enter it again

---

## Important: Free tier limitations on Render

| Limitation | Detail |
|-----------|--------|
| **Cold starts** | Free services spin down after 15 min of inactivity. First request after idle takes ~30–60s to wake up |
| **Ephemeral disk** | The SQLite memory DB (`wayfinder_memory.db`) resets on every redeploy or restart. The agent works fully — just without cross-session memory after restarts |
| **750 free hours/month** | Enough for ~1 service running continuously |

### Keeping Tavily on the free plan
The `WAYFINDER_API_KEY` gate ensures only you can trigger searches.
1,000 Tavily credits/month = ~200 research tasks (each uses ~5 searches).
To preserve credits: keep `TAVILY_MAX_RESULTS = 5` and `MAX_TURNS = 8` in `config.py`.

---

## Updating the deployment

Any `git push` to your `main` branch triggers an automatic redeploy on Render:

```powershell
git add .
git commit -m "Your change description"
git push
```

---

## Local dev after deployment setup

Your local `.env` still works for local development. The server auto-detects
whether it's running locally (reads `.env`) or on Render (reads env vars):

```powershell
# Kill old server if running
# Then:
python main.py
# → http://localhost:8000  (no auth modal locally if WAYFINDER_API_KEY not in .env)
```
