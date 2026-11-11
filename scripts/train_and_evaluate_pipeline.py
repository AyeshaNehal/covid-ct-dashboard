"""
Sequential training and evaluation orchestrator.
Trains models in exact sequence:
  1. resnet18
  2. mobilenet_v3_large
  3. efficientnet_b0

Settings:
  - Max epochs: 12
  - Early stopping patience: 3
  - Batch size: 64
  - Full training split: 13,735 images
  - Full evaluation split: Train (13,735), Val (2,950), Test (3,000)

After each model finishes, immediately runs full-split evaluation and updates
Power BI CSVs so results are usable incrementally.
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train import train_model
from src.evaluate import run_full_evaluation
from src.utils import setup_logger

logger = setup_logger("orchestrator")

ORDER = ["resnet18", "mobilenet_v3_large", "efficientnet_b0"]
EPOCHS = 12
PATIENCE = 3
BATCH_SIZE = 64

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    models_to_run = [target] if target else ORDER

    for model_name in models_to_run:
        logger.info(f"===========================================================")
        logger.info(f"  STARTING PIPELINE FOR: {model_name.upper()}")
        logger.info(f"  Configuration: epochs={EPOCHS}, patience={PATIENCE}, batch_size={BATCH_SIZE}")
        logger.info(f"===========================================================")

        # 1. Train model
        t0 = time.time()
        ckpt_path, hist_df, train_time_min = train_model(
            model_name=model_name,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            patience=PATIENCE,
        )
        t_train = round((time.time() - t0) / 60.0, 2)
        logger.info(f"Completed training for {model_name} in {t_train} minutes.")

        # 2. Immediately evaluate across full splits (train, val, test)
        logger.info(f"Running full evaluation for {model_name} across all 19,685 slices...")
        t_eval_start = time.time()
        run_full_evaluation(model_names=[model_name], batch_size=BATCH_SIZE)
        t_eval = round((time.time() - t_eval_start) / 60.0, 2)
        logger.info(f"Completed full evaluation for {model_name} in {t_eval} minutes.")
        logger.info(f"Results for {model_name} are now written and ready in powerbi/data/!")
