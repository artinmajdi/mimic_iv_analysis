import sys
import os

# Print Python path
print("Python Path:")
for path in sys.path:
    print(f"  - {path}")

print("\nCurrent directory:", os.getcwd())

# Try different import approaches
print("\nTrying imports:")
try:
    import mimic_iv_analysis
    print("✅ import mimic_iv_analysis worked")
except ImportError as e:
    print(f"❌ import mimic_iv_analysis failed: {e}")

try:
    from mimic_iv_analysis import data
    print("✅ from mimic_iv_analysis import data worked")
except ImportError as e:
    print(f"❌ from mimic_iv_analysis import data failed: {e}")

try:
    import src
    print("✅ import src worked")
except ImportError as e:
    print(f"❌ import src failed: {e}")

try:
    from src import data
    print("✅ from src import data worked")
except ImportError as e:
    print(f"❌ from src import data failed: {e}")
