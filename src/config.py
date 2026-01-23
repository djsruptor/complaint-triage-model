from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / 'model'
MODEL_FILE = MODEL_DIR / 'pipeline.joblib'

DEFAULT_THRESHOLD = 0.5

API_NAME = 'Complaint Priority Scorer'
API_VERSION = '1.0.0'