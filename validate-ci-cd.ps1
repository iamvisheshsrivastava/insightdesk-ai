# PowerShell script for CI/CD validation on Windows
# validate-ci-cd.ps1

Write-Host "🚀 InsightDesk AI - CI/CD Pipeline Validation" -ForegroundColor Blue
Write-Host "=============================================" -ForegroundColor Blue

# Function to check if command exists
function Test-Command {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to run command and capture output
function Invoke-SafeCommand {
    param($Command, $Description)
    Write-Host "⏳ $Description..." -ForegroundColor Yellow
    try {
        $result = Invoke-Expression $Command 2>&1
        Write-Host "✅ $Description completed" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ $Description failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

$success = $true

# Check Python
Write-Host "`n🐍 Checking Python environment..." -ForegroundColor Cyan
if (Test-Command "python") {
    $pythonVersion = python --version
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    
    # Check Python version (should be 3.10+)
    $version = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ([float]$version -ge 3.10) {
        Write-Host "✅ Python version is compatible" -ForegroundColor Green
    } else {
        Write-Host "❌ Python version should be 3.10+" -ForegroundColor Red
        $success = $false
    }
} else {
    Write-Host "❌ Python not found" -ForegroundColor Red
    $success = $false
}

# Check pip
if (Test-Command "pip") {
    Write-Host "✅ pip found" -ForegroundColor Green
} else {
    Write-Host "❌ pip not found" -ForegroundColor Red
    $success = $false
}

# Check required files
Write-Host "`n📋 Checking configuration files..." -ForegroundColor Cyan
$requiredFiles = @(
    ".github\workflows\ci-cd.yml",
    "pyproject.toml",
    ".flake8",
    ".gitignore",
    "Dockerfile",
    "requirements.txt",
    "Makefile"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✅ Found: $file" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing: $file" -ForegroundColor Red
        $success = $false
    }
}

# Check development tools
Write-Host "`n🛠️  Checking development tools..." -ForegroundColor Cyan
$tools = @("black", "isort", "flake8", "pytest")

foreach ($tool in $tools) {
    if (Test-Command $tool) {
        Write-Host "✅ $tool is available" -ForegroundColor Green
    } else {
        Write-Host "⚠️  $tool not found (install with: pip install $tool)" -ForegroundColor Yellow
    }
}

# Check Docker (optional)
Write-Host "`n🐳 Checking Docker..." -ForegroundColor Cyan
if (Test-Command "docker") {
    $dockerVersion = docker --version
    Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
} else {
    Write-Host "⚠️  Docker not found (optional for local development)" -ForegroundColor Yellow
}

# Try to run Python validation script
Write-Host "`n🔍 Running detailed validation..." -ForegroundColor Cyan
if (Test-Path "scripts\validate_ci_cd.py") {
    try {
        $pythonResult = python scripts\validate_ci_cd.py 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Detailed validation passed" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Some detailed validation issues found" -ForegroundColor Yellow
            Write-Host $pythonResult
        }
    }
    catch {
        Write-Host "⚠️  Could not run detailed validation: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Detailed validation script not found" -ForegroundColor Yellow
}

# Quick format check
Write-Host "`n🎨 Quick format check..." -ForegroundColor Cyan
if (Test-Command "black") {
    $formatResult = Invoke-SafeCommand "python -m black --check --diff ." "Code format check"
    if (-not $formatResult) {
        Write-Host "💡 Run 'black .' to fix formatting" -ForegroundColor Blue
    }
}

# Summary
Write-Host "`n📊 Validation Summary" -ForegroundColor Blue
Write-Host "===================" -ForegroundColor Blue

if ($success) {
    Write-Host "🎉 Basic validation passed! CI/CD pipeline should work correctly." -ForegroundColor Green
    Write-Host "💡 Next steps:" -ForegroundColor Blue
    Write-Host "  1. Install missing dev tools: pip install black isort flake8 pytest bandit safety" -ForegroundColor White
    Write-Host "  2. Format code: black . && isort ." -ForegroundColor White
    Write-Host "  3. Run tests: pytest" -ForegroundColor White
    Write-Host "  4. Push to GitHub to trigger CI/CD pipeline" -ForegroundColor White
} else {
    Write-Host "❌ Some basic requirements are missing." -ForegroundColor Red
    Write-Host "💡 Please fix the issues above before proceeding." -ForegroundColor Blue
}

Write-Host "`n🔗 Useful commands:" -ForegroundColor Blue
Write-Host "  make help          # See all available commands" -ForegroundColor White
Write-Host "  make dev-setup     # Setup development environment" -ForegroundColor White
Write-Host "  make pre-commit    # Run pre-commit checks" -ForegroundColor White
Write-Host "  make build         # Full build pipeline" -ForegroundColor White

# Pause to see results
Read-Host "`nPress Enter to exit"