import logging

import fire

from team_red.config import CONFIG
from team_red.frontends import document_generation, document_questioning


class App:
    def gen(self) -> None:
        document_generation()

    def qa(self) -> None:
        document_questioning()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    fire.Fire(App)
