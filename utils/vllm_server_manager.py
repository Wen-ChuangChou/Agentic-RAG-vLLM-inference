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

        self._process: Optional[subprocess.Popen] = None
        self.url = f"http://localhost:{port}/v1"

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_command(self) -> list:
        """Build the ``python -m vllm.entrypoints.openai.api_server`` cmd."""
        cmd = [
            sys.executable,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_id,
            "--served-model-name", self.served_model_name,
            "--port", str(self.port),
            "--api-key", self.api_key,
            "--host", self.host,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--dtype", self.dtype,
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

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Create a new process group so we can kill the entire tree
            preexec_fn=os.setsid,
        )
        print(f"  PID: {self._process.pid}")

    def wait_for_health(self) -> None:
        """Block until the ``/health`` endpoint returns HTTP 200."""
        health_url = f"http://localhost:{self.port}/health"
        print(
            f"Waiting for vLLM server health "
            f"(timeout={self.health_timeout}s) ..."
        )

        start = time.time()
        while time.time() - start < self.health_timeout:
            # Check if the process died
            if self._process.poll() is not None:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode(
                        errors="replace"
                    )
                raise RuntimeError(
                    f"vLLM server exited unexpectedly "
                    f"(code {self._process.returncode}).\n"
                    f"stderr (last 2000 chars):\n{stderr[-2000:]}"
                )

            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        elapsed = time.time() - start
                        print(
                            f">>> vLLM server healthy! "
                            f"(took {elapsed:.0f}s) <<<"
                        )
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass

            elapsed = int(time.time() - start)
            print(
                f"  [{elapsed}s/{self.health_timeout}s] Not ready yet..."
            )
            time.sleep(self.health_poll_interval)

        # Timed out
        self.stop()
        raise TimeoutError(
            f"vLLM server not healthy after {self.health_timeout}s"
        )

    def stop(self) -> None:
        """Stop the vLLM server subprocess (kills entire process group).

        Uses SIGTERM first, then SIGKILL as fallback.  After termination
        we briefly pause to give the CUDA driver time to reclaim GPU
        memory from the defunct worker processes.
        """
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

        self._process = None

        # Give the CUDA driver time to reclaim GPU memory from the killed
        # worker processes.  Without this, the next phase may see stale
        # GPU allocations and fail with OOM.
        print("Waiting 15s for GPU memory reclaim after server shutdown ...")
        time.sleep(15)

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
