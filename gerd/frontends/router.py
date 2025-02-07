"""A gradio frontend to start and stop the GERD services.

Since most hosts that use a frontend will not have enough memory to run multiple
services at the same time this router is used to start and stop the services as needed.
"""

import logging
import os
import re
import socket
import subprocess
import sys
from enum import Enum, auto
from time import sleep
from typing import Optional

import gradio as gr

_LOGGER = logging.getLogger(__name__)

demo = gr.Blocks(title="GERD")


class AppState(Enum):
    """The state of the service."""

    STOPPED = auto()
    """All services is stopped."""
    GENERATE_STARTING = auto()
    """The generation service is starting."""
    GENERATE_STARTED = auto()
    """The generation service is started."""
    QA_STARTING = auto()
    """The QA service is starting."""
    QA_STARTED = auto()
    """The QA service is started."""
    SIMPLE_STARTING = auto()
    """The simple generation service is starting."""
    SIMPLE_STARTED = auto()
    """The simple generation service is started."""
    INSTRUCT_STARTING = auto()
    """The instruct service is starting."""
    INSTRUCT_STARTED = auto()
    """The instruct service is started."""
    TRAINING_STARTING = auto()
    """The training service is starting."""
    TRAINING_STARTED = auto()
    """The training service is started."""


GRADIO_SERVER_PORT = os.environ.get("GRADIO_SERVER_PORT", "12121")
"""The port the gradio server is running on."""
GRADIO_ROUTER_PORT = os.environ.get("GRADIO_ROUTER_PORT", "7860")
"""The port the router is running on."""


class AppController:
    """The controller for the app."""

    _instance: Optional["AppController"] = None
    state = AppState.STOPPED

    @classmethod
    def instance(cls) -> "AppController":
        """Get the instance of the controller."""
        if cls._instance is None:
            cls._instance = AppController()
        return cls._instance

    def __init__(self) -> None:
        """The controlller is initialized in the stopped state."""
        self.process: subprocess.Popen | None = None
        self.state = AppState.STOPPED

    def stop(self) -> str:
        """Stop the service when it is running.

        Returns:
            The name of the current app state.
        """
        if self.state != AppState.STOPPED and self.process:
            gr.Info(f"Trying to stop process... {self.process.pid}")
            self.process.terminate()
            ret = self.process.wait(5)
            if ret is None:
                gr.Info("Process did not stop in time, killing it")
                self.process.kill()
            gr.Info("Stopped")
        if self.check_port(int(GRADIO_SERVER_PORT)):
            gr.Info("Stopping service")
            prt = int(GRADIO_SERVER_PORT)
            res = subprocess.check_output(["lsof", "-i", f":{prt}"])  # noqa: S603, S607
            m = re.search(r"[Py]ython\s+(\d+)", res.decode(encoding="utf-8"))
            if m:
                subprocess.check_call(["kill", m.group(1)])  # noqa: S603, S607
                gr.Warning(f"Killed service on port {prt}")
            else:
                msg = "Service could not be stopped"
                raise gr.Error(msg)
        self.state = AppState.STOPPED
        return self.state.name

    def start(self, frontend: str) -> None:
        """Start the service with the given frontend.

        Parameters:
            frontend: The frontend service name to start.
        """
        if not re.match("^[a-zA-Z0-9_]+$", frontend):
            msg = "Invalid frontend name"
            raise gr.Error(msg)
        self.stop()
        cmd = [sys.executable, "-m", f"gerd.frontends.{frontend}"]
        self.process = subprocess.Popen(  # noqa: S603
            cmd,
            env=os.environ | {"GRADIO_SERVER_PORT": GRADIO_SERVER_PORT},
        )

    @staticmethod
    def check_port(port: int) -> bool:
        """Check if the service port is open.

        Parameters:
            port: The port to check

        Returns:
            True if the port us open, False otherwise.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", port)) == 0
        sock.close()
        return result

    def start_gen(self) -> str:
        """Start the generation service.

        Returns:
            The name of the current app state.
        """
        self.start("gen_frontend")
        self.state = AppState.GENERATE_STARTING
        return self.state.name

    def start_simple(self) -> str:
        """Start the simple generation service.

        Returns:
            The name of the current app state
        """
        self.start("generate")
        self.state = AppState.SIMPLE_STARTING
        return self.state.name

    def start_instruct(self) -> str:
        """Start the instruct service.

        Returns:
            The name of the current app state
        """
        self.start("instruct")
        self.state = AppState.INSTRUCT_STARTING
        return self.state.name

    def start_qa(self) -> str:
        """Start the QA service.

        Returns:
            The name of the current app state
        """
        self.start("qa_frontend")
        self.state = AppState.QA_STARTING
        return self.state.name

    def start_training(self) -> str:
        """Start the training service.

        Returns:
            The name of the current app state
        """
        self.start("training")
        self.state = AppState.TRAINING_STARTING
        return self.state.name


def check_state() -> str:
    """Checks the app state and waits for the service to start.

    Returns:
        The name of the current app state.
    """
    app = AppController.instance()
    cnt = 0
    while not app.check_port(int(GRADIO_SERVER_PORT)):
        _LOGGER.info("Waiting for service to start")
        sleep(1)
        cnt += 1
        if cnt > 30:
            app.state = AppState.STOPPED
            msg = "Service did not start in time"
            raise Exception(msg)
    app.state = AppState(app.state.value + 1)
    gr.Success(f"Service started on port {GRADIO_SERVER_PORT}")
    return app.state.name


with demo:
    gr.Markdown("# GERD - Router")
    app = AppController.instance()

    with gr.Row(equal_height=True):
        state_txt = gr.Textbox(app.state.name, label="State")
        service_link = gr.Button(
            value=f"Open Service",
            interactive=False,
        )
        base_url_js = "${window.location.protocol}//${window.location.hostname}"
        service_link.click(
            None,
            js=f"() => {{ window.open(`{base_url_js}:{GRADIO_SERVER_PORT}`); }}",
        )
    gr.Markdown("## Start Service")
    with gr.Row(height="10rem", equal_height=True):
        gr.Button("Generate").click(lambda: app.start_simple(), outputs=state_txt).then(
            check_state, outputs=state_txt
        ).then(
            lambda: gr.update(interactive=app.state != AppState.STOPPED),
            outputs=service_link,
        )
        gr.Button("Instruct ").click(
            lambda: app.start_instruct(), outputs=state_txt
        ).then(check_state, outputs=state_txt).then(
            lambda: gr.update(interactive=app.state != AppState.STOPPED),
            outputs=service_link,
        )
        gr.Button("Generate discharge letter").click(
            app.start_gen, outputs=state_txt
        ).then(check_state, outputs=state_txt).then(
            lambda: gr.update(interactive=app.state != AppState.STOPPED),
            outputs=service_link,
        )
        gr.Button("Document QA").click(lambda: app.start_qa(), outputs=state_txt).then(
            lambda: gr.update(interactive=app.state != AppState.STOPPED),
            outputs=service_link,
        ).then(check_state, outputs=state_txt).then(
            lambda: gr.update(interactive=app.state != AppState.STOPPED),
            outputs=service_link,
        )
        gr.Button("LoRA Training").click(
            lambda: app.start_training(), outputs=state_txt
        ).then(
            lambda: gr.update(interactive=app.state != AppState.STOPPED),
            outputs=service_link,
        ).then(check_state, outputs=state_txt).then(
            lambda: gr.update(interactive=app.state != AppState.STOPPED),
            outputs=service_link,
        )
    gr.Button("Stop").click(lambda: app.stop(), outputs=state_txt).then(
        lambda: gr.update(interactive=app.state != AppState.STOPPED),
        outputs=service_link,
    )
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("gerd").setLevel("DEBUG")
    demo.launch(server_port=int(GRADIO_ROUTER_PORT))
