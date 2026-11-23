# ==============================================================================
# Sakhi — Startup & Management Script (PowerShell / Windows)
# ==============================================================================
# Automates:
# 1. Environment validation (.env check)
# 2. Port cleanup (8000 and 5173)
# 3. Starting the FastAPI backend
# 4. Waiting for backend readiness (/health)
# 5. Starting the React Vite frontend
# ==============================================================================

$ErrorActionPreference = "Continue"

Write-Host "`n⚖️  Starting Sakhi Legal Companion (Windows)..." -ForegroundColor Cyan

# ── Step 1: Check .env file ───────────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host "⚠️  No .env file found. Creating from .env.example..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "👉 Please edit .env with your GROQ_API_KEY and OPENROUTER_API_KEY before chatting!`n" -ForegroundColor Yellow
    } else {
        Write-Host "❌ .env file missing! Please create one with GROQ_API_KEY and OPENROUTER_API_KEY." -ForegroundColor Red
    }
}

# ── Step 2: Clean stale ports ────────────────────────────────────────────────
function Kill-PortProcess($port) {
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conns) {
        $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($p in $pids) {
            if ($p -gt 0) {
                Write-Host "Killing stale process on port $port (PID: $p)..." -ForegroundColor Yellow
                Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

Kill-PortProcess 8000
Kill-PortProcess 5173

# ── Step 3: Start Backend ─────────────────────────────────────────────────────
Write-Host "🚀 Launching Sakhi FastAPI Backend on port 8000..." -ForegroundColor Green
$venvPython = ".\.venv\Scripts\python.exe"
$pythonCmd = if (Test-Path $venvPython) { $venvPython } else { "python" }

$backendJob = Start-Process -FilePath $pythonCmd -ArgumentList "-m", "src.api.main" -PassThru -NoNewWindow

# ── Step 4: Wait for Health ───────────────────────────────────────────────────
Write-Host "⏳ Waiting for RAG Engine and ChromaDB to initialize..." -NoNewline -ForegroundColor Gray
$healthy = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    Write-Host "." -NoNewline -ForegroundColor Gray
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($resp -and ($resp.status -eq "healthy" -or $resp.status -eq "initializing")) {
            $healthy = $true
            break
        }
    } catch {}
}
Write-Host ""

if ($healthy) {
    Write-Host "✅ Backend is UP and READY!" -ForegroundColor Green
} else {
    Write-Host "⚠️ Backend is still initializing or encountered an issue. See logs above." -ForegroundColor Yellow
}

# ── Step 5: Start Frontend ────────────────────────────────────────────────────
Write-Host "🎨 Starting Vite React Frontend on http://localhost:5173..." -ForegroundColor Green
cd sakhi-ui
npm run dev

# Cleanup when frontend exits
if ($backendJob -and -not $backendJob.HasExited) {
    Write-Host "🛑 Stopping Sakhi Backend..." -ForegroundColor Gray
    Stop-Process -Id $backendJob.Id -Force -ErrorAction SilentlyContinue
}
Write-Host "Peace out! 🙏✨" -ForegroundColor Cyan
