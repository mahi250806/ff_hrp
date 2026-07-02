import sys
from pathlib import Path

# Make `ffhrp` importable from src/ without an editable install
sys.path.insert(0, str(Path(__file__).parent / "src"))
