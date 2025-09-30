# scripts/setup_mlflow.py

"""
Setup script for MLflow experiment tracking.
Run this to initialize MLflow tracking for the InsightDesk AI project.
"""

import subprocess
import sys
from pathlib import Path


def install_mlflow():
    """Install MLflow if not already installed."""
    try:
        import mlflow
        print("✅ MLflow already installed")
        return True
    except ImportError:
        print("📦 Installing MLflow...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow"])
            print("✅ MLflow installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install MLflow")
            return False


def setup_mlflow_tracking():
    """Setup MLflow tracking directory and server."""
    # Create MLflow directory
    mlflow_dir = Path("mlruns")
    mlflow_dir.mkdir(exist_ok=True)
    
    print("✅ MLflow tracking directory created")
    print(f"📁 MLflow data will be stored in: {mlflow_dir.absolute()}")
    
    # Create experiments
    try:
        import mlflow
        
        # Set tracking URI to local directory
        mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
        
        # Create experiments
        experiments = [
            "ticket_classification",
            "model_comparison", 
            "hyperparameter_tuning"
        ]
        
        for exp_name in experiments:
            try:
                exp_id = mlflow.create_experiment(exp_name)
                print(f"✅ Created experiment: {exp_name} (ID: {exp_id})")
            except Exception:
                print(f"📁 Experiment already exists: {exp_name}")
        
        print("\n🚀 MLflow setup complete!")
        print("\nTo start MLflow UI, run:")
        print(f"   mlflow ui --backend-store-uri file://{mlflow_dir.absolute()}")
        print("\nThen open: http://localhost:5000")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up MLflow: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up MLflow for InsightDesk AI")
    print("=" * 40)
    
    # Install MLflow
    if not install_mlflow():
        return
    
    # Setup tracking
    if not setup_mlflow_tracking():
        return
    
    print("\n✅ MLflow setup successful!")
    print("\nNext steps:")
    print("1. Run model training scripts to log experiments")
    print("2. Start MLflow UI to view results")
    print("3. Compare model performance visually")


if __name__ == "__main__":
    main()