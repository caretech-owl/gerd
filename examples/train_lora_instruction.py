"""Example script to train a LoRA model on a simple instruction dataset.

The generated data set is very simple and should only illustrate the intended
usage of the InstructTrainingData and InstructTrainingSample classes.
"""

import logging
import time
from pathlib import Path

from gerd.config import PROJECT_DIR
from gerd.models.model import ChatMessage
from gerd.training.instruct import (
    InstructTrainingData,
    InstructTrainingSample,
    train_lora,
)

_LOGGER = logging.getLogger(__name__)

logging.basicConfig(level=logging.WARNING)
logging.getLogger("gerd").setLevel(logging.DEBUG)

if Path().cwd() != PROJECT_DIR:
    msg = "This example must be run from the project root."
    raise AssertionError(msg)

data_glob = "tmp/*.json"

data = InstructTrainingData()
for i in range(1000):
    data.samples.append(
        InstructTrainingSample(
            messages=[
                ChatMessage(
                    role="user",
                    content=f"Bitte erweitere die Zahl {i}.",
                ),
                ChatMessage(
                    role="assistant",
                    content=(
                        "Nichts leichter als das! "
                        f"Hier hast du die {i} drei mal hintereinander: {i}{i}{i}."
                    ),
                ),
            ]
        )
    )

Path("tmp").mkdir(exist_ok=True)
with open("tmp/test.json", "w") as f:
    f.write(data.model_dump_json(indent=2))

trainer = train_lora("lora_instruct_example")
try:
    while trainer.thread.is_alive():
        time.sleep(0.5)
except KeyboardInterrupt:
    trainer.interrupt()
    _LOGGER.info(
        "Interrupting, please wait... "
        "*(Run will stop after the current training step completes.)*"
    )
    trainer.thread.join()

if not trainer.tracked.did_save:
    trainer.save()

if trainer.tracked.interrupted:
    _LOGGER.info("Interrupted. Incomplete LoRA saved to %s.", trainer.config.output_dir)
else:
    _LOGGER.info(
        "Done! LoRA saved to %s.\n\nBefore testing your new LoRA, "
        "make sure to first reload the model, as it is currently dirty from training.",
        trainer.config.output_dir,
    )
