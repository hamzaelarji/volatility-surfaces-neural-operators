from datetime import datetime
from pathlib import Path
import logging
import sys

import papermill as pm


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "executed_notebooks"
LOG_FILE = BASE_DIR / "notebooks_execution.log"

# L'ordre d'exécution est important.
NOTEBOOKS = [
    "04_neural_operator_v2.ipynb",
    "05_downstream_economics.ipynb",
]


OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")

for notebook_name in NOTEBOOKS:
    input_path = BASE_DIR / notebook_name
    output_path = OUTPUT_DIR / (
        f"{input_path.stem}_{execution_id}_executed.ipynb"
    )

    if not input_path.exists():
        logging.error("Notebook introuvable : %s", input_path)
        sys.exit(1)

    logging.info("Début : %s", notebook_name)

    try:
        pm.execute_notebook(
            input_path=str(input_path),
            output_path=str(output_path),
            cwd=str(BASE_DIR),
            kernel_name="python3",
            log_output=True,
        )
    except Exception:
        logging.exception("Échec du notebook : %s", notebook_name)
        sys.exit(1)

    logging.info("Terminé : %s", notebook_name)

logging.info("Tous les notebooks ont été exécutés avec succès.")