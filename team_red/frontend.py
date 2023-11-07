import logging

import fire

from team_red.config import CONFIG
from team_red.frontends import gen_frontend, qa_frontend


class App:
    def gen(self) -> None:
        gen_frontend()

    def qa(self) -> None:
        qa_frontend()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    fire.Fire(App)
