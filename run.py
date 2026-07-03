import os
import sys
import subprocess
import time

def run():
    print("🚀 Starting Aura Agent Chatbot Web Application...")
    
    # 1. Determine local virtual environment python interpreter path
    cwd = os.getcwd()
    venv_python = os.path.join(cwd, "venv", "bin", "python")
    
    # Windows support
    if os.name == "nt":
        venv_python = os.path.join(cwd, "venv", "Scripts", "python.exe")
        
    if not os.path.exists(venv_python):
        print(f"⚠️ Virtual environment python not found at {venv_python}.")
        print("Falling back to system python...")
        venv_python = sys.executable

    # 2. Launch FastAPI Backend
    print("🤖 Starting Backend Server (Uvicorn running FastAPI)...")
    backend_process = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        cwd=cwd
    )
    
    # Give uvicorn a brief second to claim the port
    time.sleep(1.5)

    # 3. Launch React Frontend
    print("🎨 Starting Frontend Dev Server (Vite running React)...")
    frontend_cwd = os.path.join(cwd, "Aura agent")
    
    # Use shell=True for windows command prompt execution compatibility
    use_shell = os.name == "nt"
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_cwd,
        shell=use_shell
    )

    # 4. Monitor processes and keep main thread alive
    try:
        while True:
            # Check if any process terminated unexpectedly
            if backend_process.poll() is not None:
                print("❌ Backend server stopped unexpectedly.")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend dev server stopped unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers safely...")
    finally:
        backend_process.terminate()
        frontend_process.terminate()
        print("Goodbye!")

if __name__ == "__main__":
    run()
