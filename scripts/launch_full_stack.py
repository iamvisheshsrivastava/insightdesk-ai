#!/usr/bin/env python3
"""
Full Stack Launch Script
========================

This script launches both the FastAPI backend and Streamlit dashboard
for the InsightDesk AI system.
"""

import subprocess
import sys
import time
import requests
import webbrowser
from pathlib import Path
import signal
import os


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def log(message: str, color: str = Colors.BLUE):
    """Log a colored message"""
    print(f"{color}{message}{Colors.END}")


def check_port_available(port: int) -> bool:
    """Check if a port is available"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return False  # Port is in use
    except:
        return True  # Port is available


def wait_for_service(url: str, timeout: int = 30) -> bool:
    """Wait for a service to become available"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    
    return False


def launch_services():
    """Launch FastAPI backend and Streamlit dashboard"""
    
    log("🚀 InsightDesk AI - Full Stack Launcher", Colors.BOLD)
    log("=" * 50)
    
    processes = []
    
    try:
        # Check if ports are available
        log("🔍 Checking port availability...")
        
        if not check_port_available(8000):
            log("⚠️  Port 8000 is already in use (FastAPI may already be running)", Colors.YELLOW)
            fastapi_running = True
        else:
            fastapi_running = False
        
        if not check_port_available(8501):
            log("⚠️  Port 8501 is already in use (Streamlit may already be running)", Colors.YELLOW)
            streamlit_running = True
        else:
            streamlit_running = False
        
        # Launch FastAPI backend
        if not fastapi_running:
            log("🔧 Starting FastAPI backend server...", Colors.BLUE)
            
            # Change to src directory and start FastAPI
            src_path = Path("src")
            if src_path.exists():
                fastapi_process = subprocess.Popen([
                    sys.executable, "-m", "uvicorn", "api.main:app", 
                    "--reload", "--host", "0.0.0.0", "--port", "8000"
                ], cwd=src_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                processes.append(("FastAPI", fastapi_process))
                
                # Wait for FastAPI to start
                log("⏳ Waiting for FastAPI to start...")
                if wait_for_service("http://localhost:8000/health", timeout=30):
                    log("✅ FastAPI backend is ready!", Colors.GREEN)
                else:
                    log("❌ FastAPI failed to start within timeout", Colors.RED)
                    return
            else:
                log("❌ 'src' directory not found. Please run from project root.", Colors.RED)
                return
        else:
            log("✅ FastAPI backend already running", Colors.GREEN)
        
        # Launch Streamlit dashboard
        if not streamlit_running:
            log("🎨 Starting Streamlit dashboard...", Colors.BLUE)
            
            streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.port", "8501",
                "--server.headless", "false"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(("Streamlit", streamlit_process))
            
            # Wait for Streamlit to start
            log("⏳ Waiting for Streamlit to start...")
            time.sleep(3)  # Streamlit needs a moment to initialize
            
            if wait_for_service("http://localhost:8501", timeout=20):
                log("✅ Streamlit dashboard is ready!", Colors.GREEN)
            else:
                log("❌ Streamlit failed to start within timeout", Colors.RED)
        else:
            log("✅ Streamlit dashboard already running", Colors.GREEN)
        
        # Success message
        log("\n🎉 InsightDesk AI is ready!", Colors.GREEN)
        log("=" * 50)
        log("📱 Dashboard: http://localhost:8501", Colors.BLUE)
        log("🔧 API Docs:  http://localhost:8000/docs", Colors.BLUE)
        log("❤️  Health:   http://localhost:8000/health", Colors.BLUE)
        
        # Open browser
        try:
            log("\n🌐 Opening browser...")
            webbrowser.open("http://localhost:8501")
        except:
            log("⚠️  Could not open browser automatically", Colors.YELLOW)
        
        # Wait for user input to shut down
        log(f"\n{Colors.YELLOW}Press Ctrl+C to stop all services...{Colors.END}")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            for name, process in processes:
                if process.poll() is not None:
                    log(f"⚠️  {name} process has stopped", Colors.YELLOW)
    
    except KeyboardInterrupt:
        log("\n🛑 Shutting down services...", Colors.YELLOW)
        
        # Terminate all processes
        for name, process in processes:
            try:
                log(f"🔻 Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
                log(f"✅ {name} stopped", Colors.GREEN)
            except subprocess.TimeoutExpired:
                log(f"⚠️  Force killing {name}...", Colors.YELLOW)
                process.kill()
            except:
                pass
        
        log("👋 All services stopped. Goodbye!", Colors.BLUE)
    
    except Exception as e:
        log(f"❌ Error: {str(e)}", Colors.RED)
        
        # Clean up processes
        for name, process in processes:
            try:
                process.terminate()
            except:
                pass


def main():
    """Main entry point"""
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        log("❌ app.py not found. Please run this script from the project root directory.", Colors.RED)
        log("💡 Expected structure:", Colors.BLUE)
        log("   📁 project-root/")
        log("   ├── app.py")
        log("   ├── src/")
        log("   │   └── api/")
        log("   │       └── main.py")
        log("   └── scripts/")
        return
    
    # Check if required files exist
    required_files = ["app.py", "src/api/main.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        log(f"❌ Missing required files: {missing_files}", Colors.RED)
        log("💡 Please ensure the project structure is correct", Colors.BLUE)
        return
    
    # Launch services
    launch_services()


if __name__ == "__main__":
    main()