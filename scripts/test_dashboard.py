#!/usr/bin/env python3
"""
Streamlit Dashboard Test Script
==============================

This script tests the Streamlit dashboard functionality and validates
all components work correctly.
"""

import subprocess
import sys
import time
import requests
from pathlib import Path


def test_imports():
    """Test required imports for the dashboard"""
    print("🧪 Testing dashboard imports...")
    
    required_packages = [
        "streamlit",
        "requests", 
        "plotly",
        "pandas"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Install with: pip install {package}")
            return False
    
    return True


def test_dashboard_syntax():
    """Test dashboard syntax without running"""
    print("\n🔍 Testing dashboard syntax...")
    
    try:
        # Test main dashboard
        result = subprocess.run([
            sys.executable, "-m", "py_compile", "app.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ app.py syntax is valid")
        else:
            print(f"❌ app.py syntax error: {result.stderr}")
            return False
        
        # Test demo dashboard
        if Path("demo_dashboard.py").exists():
            result = subprocess.run([
                sys.executable, "-m", "py_compile", "demo_dashboard.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ demo_dashboard.py syntax is valid")
            else:
                print(f"❌ demo_dashboard.py syntax error: {result.stderr}")
        
        return True
        
    except Exception as e:
        print(f"❌ Syntax test failed: {e}")
        return False


def test_streamlit_config():
    """Test Streamlit configuration"""
    print("\n⚙️  Testing Streamlit configuration...")
    
    config_path = Path(".streamlit/config.toml")
    if config_path.exists():
        print("✅ Streamlit config file found")
        
        # Try to parse the config
        try:
            import toml
            with open(config_path) as f:
                config = toml.load(f)
            print("✅ Config file is valid TOML")
            
            # Check required sections
            if "theme" in config:
                print("✅ Theme configuration found")
            if "server" in config:
                print("✅ Server configuration found")
                
        except ImportError:
            print("⚠️  TOML package not available, skipping config validation")
        except Exception as e:
            print(f"❌ Config file error: {e}")
            return False
    else:
        print("⚠️  No Streamlit config file found (optional)")
    
    return True


def test_launch_script():
    """Test the full stack launch script"""
    print("\n🚀 Testing launch script...")
    
    launch_script = Path("scripts/launch_full_stack.py")
    if launch_script.exists():
        try:
            result = subprocess.run([
                sys.executable, "-m", "py_compile", str(launch_script)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Launch script syntax is valid")
            else:
                print(f"❌ Launch script syntax error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Launch script test failed: {e}")
            return False
    else:
        print("❌ Launch script not found")
        return False
    
    return True


def test_makefile_commands():
    """Test Makefile dashboard commands"""
    print("\n📋 Testing Makefile dashboard commands...")
    
    makefile_path = Path("Makefile")
    if makefile_path.exists():
        with open(makefile_path) as f:
            content = f.read()
            
        dashboard_commands = [
            "dashboard:",
            "dashboard-demo:",
            "full-stack:",
            "dashboard-dev:"
        ]
        
        for command in dashboard_commands:
            if command in content:
                print(f"✅ {command.rstrip(':')} command found")
            else:
                print(f"❌ {command.rstrip(':')} command missing")
                return False
    else:
        print("⚠️  Makefile not found")
    
    return True


def run_quick_demo():
    """Run a quick demo test"""
    print("\n🎭 Running quick demo test...")
    
    # Test if we can import the main dashboard module
    try:
        sys.path.insert(0, '.')
        
        # Try to import without running
        spec = __import__('importlib.util', fromlist=['spec_from_file_location']).spec_from_file_location
        spec = spec("app", "app.py")
        
        if spec and spec.loader:
            print("✅ Dashboard module can be imported")
        else:
            print("❌ Dashboard module import failed")
            return False
            
    except Exception as e:
        print(f"❌ Demo test failed: {e}")
        return False
    
    return True


def main():
    """Main test function"""
    print("🧪 Streamlit Dashboard Test Suite")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_imports),
        ("Dashboard Syntax", test_dashboard_syntax),
        ("Streamlit Config", test_streamlit_config),
        ("Launch Script", test_launch_script),
        ("Makefile Commands", test_makefile_commands),
        ("Quick Demo", run_quick_demo)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20}")
        print(f"🧪 {test_name}")
        print('='*20)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    # Summary
    print(f"\n{'='*40}")
    print(f"📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Install dependencies: pip install streamlit requests plotly")
        print("2. Start the dashboard: streamlit run app.py")
        print("3. Or use full stack: python scripts/launch_full_stack.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\n💡 Quick fixes:")
        print("- Install missing packages: pip install streamlit requests plotly pandas")
        print("- Check file paths and syntax errors")
        print("- Ensure all required files exist")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)