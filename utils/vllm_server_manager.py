"""
Context manager for vLLM OpenAI-compatible server lifecycle (Phase 2).

Starts the server as a subprocess, waits for the /health endpoint,
and ensures clean shutdown on exit — even if the calling code raises.
"""

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional


class VLLMServerManager:
    """
    Manages a vLLM server subprocess for Phase 2 (agentic RAG).

    Usage::

        with VLLMServerManager(model_id="zai-org/GLM-4.7-Flash", ...) as srv:
            # srv.url  →  "http://localhost:8000/v1"
            ...
        # server is automatically stopped
    """

    def __init__(
        self,
        model_id: str,
        port: int = 8000,
        api_key: str = "ai4all",
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.92,
        max_model_len: int = 131072,
        dtype: str = "auto",
        host: str = "0.0.0.0",
        trust_remote_code: bool = True,
        enable_prefix_caching: bool = True,
        extra_args: Optional[List[str]] = None,
        health_timeout: int = 600,
        health_poll_interval: int = 10,
        served_model_name: Optional[str] = None,
        server_log_dir: Optional[str] = "tmp",
    ):
        self.model_id = model_id
        self.served_model_name = served_model_name or model_id
        self.port = port
        self.api_key = api_key
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.host = host
        self.trust_remote_code = trust_remote_code
        self.enable_prefix_caching = enable_prefix_caching
        self.extra_args = extra_args or []
        self.health_timeout = health_timeout
        self.health_poll_interval = health_poll_interval
        self.server_log_dir = server_log_dir

        self._process: Optional[subprocess.Popen] = None
        self._log_files: list = []  # open file handles for server logs
        self.url = f"http://localhost:{port}/v1"

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_command(self) -> list:
        """Build the ``python -m vllm.entrypoints.openai.api_server`` cmd."""
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_id,
            "--served-model-name",
            self.served_model_name,
            "--port",
            str(self.port),
            "--api-key",
            self.api_key,
            "--host",
            self.host,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            "--dtype",
            self.dtype,
        ]
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        cmd.append("--disable-custom-all-reduce")

        cmd.extend(self.extra_args)
        return cmd

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the vLLM server subprocess."""
        cmd = self._build_command()
        print(f"Starting vLLM server: {self.model_id} on port {self.port}")
        print(f"  Command: {' '.join(cmd[:8])} ...")

        # Redirect server stdout/stderr to log files instead of PIPE.
        # Using PIPE without consuming the output causes the OS pipe
        # buffer (~64KB) to fill up, which blocks the server and makes
        # all subsequent API requests fail with "Connection error".
        stdout_dest = subprocess.DEVNULL
        stderr_dest = subprocess.DEVNULL
        if self.server_log_dir:
            from pathlib import Path
            log_dir = Path(self.server_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            f_out = open(log_dir / "vllm_server.out", "w")
            f_err = open(log_dir / "vllm_server.err", "w")
            self._log_files = [f_out, f_err]
            stdout_dest = f_out
            stderr_dest = f_err
            print(f"  Server logs: {log_dir / 'vllm_server.out'}")
            print(f"               {log_dir / 'vllm_server.err'}")

        self._process = subprocess.Popen(
            cmd,
            stdout=stdout_dest,
            stderr=stderr_dest,
            # Create a new process group so we can kill the entire tree
            preexec_fn=os.setsid,
        )
        print(f"  PID: {self._process.pid}")

    def wait_for_health(self) -> None:
        """Block until the ``/health`` endpoint returns HTTP 200."""
        health_url = f"http://localhost:{self.port}/health"
        print(f"Waiting for vLLM server health "
              f"(timeout={self.health_timeout}s) ...")

        start = time.time()
        while time.time() - start < self.health_timeout:
            # Check if the process died
            if self._process.poll() is not None:
                stderr_tail = ""
                if self.server_log_dir:
                    from pathlib import Path
                    err_path = Path(self.server_log_dir) / "vllm_server.err"
                    if err_path.exists():
                        stderr_tail = err_path.read_text(
                            errors="replace")[-2000:]
                raise RuntimeError(f"vLLM server exited unexpectedly "
                                   f"(code {self._process.returncode}).\n"
                                   f"stderr (last 2000 chars):\n{stderr_tail}")

            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        elapsed = time.time() - start
                        print(f">>> vLLM server healthy! "
                              f"(took {elapsed:.0f}s) <<<")
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass

            elapsed = int(time.time() - start)
            print(f"  [{elapsed}s/{self.health_timeout}s] Not ready yet...")
            time.sleep(self.health_poll_interval)

        # Timed out
        self.stop()
        raise TimeoutError(
            f"vLLM server not healthy after {self.health_timeout}s")

    def stop(self) -> None:
        """Stop the vLLM server subprocess (kills entire process group)."""
        if self._process is None:
            return

        if self._process.poll() is None:
            print(f"Stopping vLLM server (PID {self._process.pid}) ...")
            try:
                pgid = os.getpgid(self._process.pid)
                os.killpg(pgid, signal.SIGTERM)
                self._process.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                    self._process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            print("vLLM server stopped.")

        # Close log file handles
        for fh in self._log_files:
            try:
                fh.close()
            except Exception:
                pass
        self._log_files = []

        self._process = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        self.wait_for_health()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
