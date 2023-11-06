import logging

import fire

from team_red.config import CONFIG
from team_red.document_generation import document_generation
from team_red.document_questioning import document_questioning


class App:
    def gen(self):
        document_generation()

    def qa(self):
        document_questioning()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    fire.Fire(App)
