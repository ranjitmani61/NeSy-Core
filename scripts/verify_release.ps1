<# 
    scripts/verify_release.ps1
    ==========================
    One-command release verification for NeSy-Core.
    Run from repo root:  .\scripts\verify_release.ps1
    
    Exit code 0 = ALL PASS.  Non-zero = FAIL (with details).
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Push-Location $root

$failed = 0
$total  = 0

function Run-Check {
    param([string]$Label, [scriptblock]$Block)
    $script:total++
    Write-Host ""
    Write-Host "[$script:total] $Label" -ForegroundColor Cyan
    Write-Host ("-" * 60)
    try {
        & $Block
        if ($LASTEXITCODE -ne 0) { throw "Non-zero exit code: $LASTEXITCODE" }
        Write-Host "    PASS" -ForegroundColor Green
    } catch {
        Write-Host "    FAIL: $_" -ForegroundColor Red
        $script:failed++
    }
}

# ── 1. Full test suite ────────────────────────────────────────────
Run-Check "Full test suite (pytest -q, 0 failures, 0 warnings)" {
    $output = & .venv\Scripts\python -m pytest -q --tb=short 2>&1 | Out-String
    Write-Host $output
    if ($output -match "failed") { throw "Test failures detected" }
    if ($output -match "\d+ warnings?") {
        # Extract warning count
        if ($output -match "(\d+) warnings?") {
            $warnCount = [int]$Matches[1]
            if ($warnCount -gt 0) { throw "$warnCount warning(s) detected" }
        }
    }
}

# ── 2. Coverage: nesy/api/nesy_model.py ≥ 100% ───────────────────
Run-Check "Coverage: nesy/api/nesy_model.py = 100%" {
    $output = & .venv\Scripts\python -m pytest tests/unit/test_api_hardening.py `
        --cov=nesy.api.nesy_model --cov-report=term-missing -q --tb=no 2>&1 | Out-String
    Write-Host $output
    if ($output -notmatch "nesy_model\.py\s+\d+\s+0\s+100%") {
        throw "nesy_model.py coverage < 100%"
    }
}

# ── 3. Coverage: nesy/core/validators.py ≥ 100% ──────────────────
Run-Check "Coverage: nesy/core/validators.py = 100%" {
    $output = & .venv\Scripts\python -m pytest tests/unit/test_api_hardening.py `
        --cov=nesy.core.validators --cov-report=term-missing -q --tb=no 2>&1 | Out-String
    Write-Host $output
    if ($output -notmatch "validators\.py\s+\d+\s+0\s+100%") {
        throw "validators.py coverage < 100%"
    }
}

# ── 4. Example: basic_reasoning.py ────────────────────────────────
Run-Check "Example: basic_reasoning.py" {
    & .venv\Scripts\python examples/basic_reasoning.py 2>&1 | Out-String | Write-Host
}

# ── 5. Example: medical_diagnosis.py ──────────────────────────────
Run-Check "Example: medical_diagnosis.py" {
    & .venv\Scripts\python examples/medical_diagnosis.py 2>&1 | Out-String | Write-Host
}

# ── 6. Example: continual_learning.py ─────────────────────────────
Run-Check "Example: continual_learning.py" {
    & .venv\Scripts\python examples/continual_learning.py 2>&1 | Out-String | Write-Host
}

# ── 7. Example: edge_deployment.py ────────────────────────────────
Run-Check "Example: edge_deployment.py" {
    & .venv\Scripts\python examples/edge_deployment.py 2>&1 | Out-String | Write-Host
}

# ── SUMMARY ───────────────────────────────────────────────────────
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor White
if ($failed -eq 0) {
    Write-Host "  RELEASE VERIFICATION: ALL $total CHECKS PASSED" -ForegroundColor Green
} else {
    Write-Host "  RELEASE VERIFICATION: $failed / $total CHECKS FAILED" -ForegroundColor Red
}
Write-Host ("=" * 60) -ForegroundColor White

Pop-Location
exit $failed
