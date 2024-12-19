import logging
import time
from pathlib import Path

from gerd.config import PROJECT_DIR
from gerd.training.unstructured import train_lora

_LOGGER = logging.getLogger(__name__)

logging.basicConfig(level=logging.WARNING)
logging.getLogger("gerd").setLevel(logging.DEBUG)

if Path().cwd() != PROJECT_DIR:
    msg = "This example must be run from the project root."
    raise AssertionError(msg)

trainer = train_lora("lora_example")
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
