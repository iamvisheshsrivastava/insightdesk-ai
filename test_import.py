#!/usr/bin/env python3
import sys
try:
    from src.api.main import app
    print("✅ Import successful")
    print(f"✅ App loaded: {app}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
