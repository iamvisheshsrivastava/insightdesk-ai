# scripts/demo_workflow.py

"""
Demo script that demonstrates the complete InsightDesk AI workflow:
1. Data preparation
2. Model training 
3. API testing
4. MLflow experiment tracking

This script provides a complete end-to-end demonstration.
"""

import subprocess
import sys
import time
import requests
import json
from pathlib import Path


def run_command(command, description, check=True):
    """Run a command with description."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(command) if isinstance(command, list) else command}")
    
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=check, capture_output=True, text=True)
        
        if result.stdout:
            print("✅ Output:", result.stdout.strip())
        if result.stderr and check:
            print("⚠️ Warnings:", result.stderr.strip())
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stdout:
            print("Output:", e.stdout.strip())
        if e.stderr:
            print("Error:", e.stderr.strip())
        return False


def check_file_exists(file_path, description):
    """Check if a file exists."""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ Missing {description}: {file_path}")
        return False


def test_api_endpoint(url, data=None, description="API test"):
    """Test an API endpoint."""
    try:
        if data:
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        
        print(f"🌐 {description}")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Success: {json.dumps(result, indent=2)[:200]}...")
            return True
        else:
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False


def main():
    """Run the complete demonstration workflow."""
    print("🚀 InsightDesk AI - Complete Workflow Demonstration")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Check prerequisites
    print("\n📋 Step 1: Checking Prerequisites")
    
    required_files = [
        ("support_tickets.zip", "Dataset"),
        ("src/api/main.py", "API module"),
        ("src/models/xgboost_classifier.py", "XGBoost classifier"),
        ("src/models/tensorflow_classifier.py", "TensorFlow classifier")
    ]
    
    all_files_exist = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_files_exist = False
    
    if not all_files_exist:
        print("❌ Missing required files. Please ensure all files are in place.")
        return
    
    # Step 2: Data preparation
    print("\n📊 Step 2: Data Preparation")
    
    if not Path("data/support_tickets.json").exists():
        success = run_command(
            [sys.executable, "scripts/unzip_and_load.py"],
            "Extracting and loading support tickets data"
        )
        if not success:
            print("❌ Data preparation failed")
            return
    else:
        print("✅ Data already prepared")
    
    # Step 3: Feature building
    print("\n⚙️ Step 3: Feature Engineering")
    
    if not Path("data/features.joblib").exists():
        success = run_command(
            [sys.executable, "scripts/build_features.py"],
            "Building feature matrix"
        )
        if not success:
            print("❌ Feature building failed")
            return
    else:
        print("✅ Features already built")
    
    # Step 4: Setup MLflow (optional)
    print("\n📊 Step 4: MLflow Setup")
    run_command(
        [sys.executable, "scripts/setup_mlflow.py"],
        "Setting up MLflow experiment tracking",
        check=False  # Don't fail if MLflow setup has issues
    )
    
    # Step 5: Model training
    print("\n🏗️ Step 5: Model Training")
    
    # Check if models already exist
    xgb_exists = Path("models/xgb_category.pkl").exists()
    tf_exists = Path("models/tf_category_model.h5").exists()
    
    if not xgb_exists or not tf_exists:
        print("Training models (this may take several minutes)...")
        
        # Train XGBoost if missing
        if not xgb_exists:
            success = run_command(
                [sys.executable, "scripts/train_xgboost.py"],
                "Training XGBoost model"
            )
            if not success:
                print("⚠️ XGBoost training failed, continuing...")
        
        # Train TensorFlow if missing (this can take a while)
        if not tf_exists:
            print("⚠️ TensorFlow training can take 10-30 minutes...")
            success = run_command(
                [sys.executable, "scripts/train_tensorflow.py"],
                "Training TensorFlow model"
            )
            if not success:
                print("⚠️ TensorFlow training failed, continuing...")
    else:
        print("✅ Models already trained")
    
    # Step 6: Model testing
    print("\n🧪 Step 6: Model Testing")
    run_command(
        [sys.executable, "scripts/test_models.py"],
        "Testing trained models",
        check=False
    )
    
    # Step 7: API testing
    print("\n🌐 Step 7: API Testing")
    
    print("Starting API server for testing...")
    
    # Start API server in background
    import subprocess
    api_process = None
    try:
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "src.api.main:app", 
            "--host", "127.0.0.1", "--port", "8000"
        ])
        
        # Wait for server to start
        print("Waiting for API server to start...")
        time.sleep(10)
        
        # Test endpoints
        base_url = "http://127.0.0.1:8000"
        
        # Health check
        test_api_endpoint(f"{base_url}/health", description="Health check")
        
        # Models info
        test_api_endpoint(f"{base_url}/models/info", description="Models info")
        
        # Category prediction
        sample_ticket = {
            "ticket_id": "DEMO-001",
            "subject": "Cannot login to application",
            "description": "User is unable to authenticate with correct credentials",
            "product": "web_application",
            "priority": "high"
        }
        
        test_api_endpoint(
            f"{base_url}/predict/category",
            sample_ticket,
            "Category prediction"
        )
        
        # Priority prediction (placeholder)
        test_api_endpoint(
            f"{base_url}/predict/priority",
            sample_ticket,
            "Priority prediction"
        )
        
    finally:
        if api_process:
            api_process.terminate()
            api_process.wait()
            print("🛑 API server stopped")
    
    # Step 8: Results summary
    print("\n📋 Step 8: Results Summary")
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 Workflow Demonstration Complete!")
    print("=" * 40)
    print(f"⏱️ Total time: {total_time:.1f} seconds")
    
    # Check what was created
    print("\n📁 Generated Files:")
    
    files_to_check = [
        ("data/support_tickets.json", "Dataset"),
        ("data/features.joblib", "Feature matrix"),
        ("models/xgb_category.pkl", "XGBoost model"),
        ("models/tf_category_model.h5", "TensorFlow model"),
        ("models/label_encoder_category.pkl", "Label encoder"),
        ("plots/", "Visualization plots (directory)"),
        ("mlruns/", "MLflow experiments (directory)")
    ]
    
    for file_path, description in files_to_check:
        path = Path(file_path)
        if path.exists():
            if path.is_dir():
                file_count = len(list(path.glob("*")))
                print(f"   ✅ {description}: {file_path} ({file_count} items)")
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {description}: {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {description}: {file_path} (missing)")
    
    print("\n🚀 Next Steps:")
    print("   1. Start the API: uvicorn src.api.main:app --reload")
    print("   2. View API docs: http://localhost:8000/docs")
    print("   3. View MLflow: mlflow ui --backend-store-uri file://./mlruns")
    print("   4. Run tests: pytest tests/test_api.py -v")
    print("   5. Generate comparison report: python scripts/train_and_compare_models.py")


if __name__ == "__main__":
    main()