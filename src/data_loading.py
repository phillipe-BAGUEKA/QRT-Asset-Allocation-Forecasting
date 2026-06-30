import sys
from pathlib import Path
import pandas as pd

ROOT = Path.cwd().parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

X_TRAIN_FILE = ROOT / "data" / "raw" / "X_train.csv"
X_TEST_FILE = ROOT / "data" / "raw" / "X_test.csv"
Y_TRAIN_FILE = ROOT / "data" / "raw" / "y_train.csv"
SAMPLE_SUBMISSION_FILE = ROOT / "data" / "raw" / "sample_submission.csv"


# fonction pour charger les features d'entrainement
def load_X_train() -> pd.DataFrame:
    return pd.read_csv(X_TRAIN_FILE)

# fonction pour charger les features de test
def load_X_test() -> pd.DataFrame:
    return pd.read_csv(X_TEST_FILE)

# fonction pour charger le label d'entrainement
def load_y_train() -> pd.DataFrame:
    return pd.read_csv(Y_TRAIN_FILE)


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_SUBMISSION_FILE)

