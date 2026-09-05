# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/envs.py

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    TRAINER_RINGBUFFER_WARNING_INTERVAL: int = 60
    TRAINER_NCCL_SO_PATH: str | None = None
    LD_LIBRARY_PATH: str | None = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: str | None = None
    TRAINER_CACHE_ROOT: str = os.path.expanduser("~/.cache/trainer")
    TRAINER_CONFIG_ROOT: str = os.path.expanduser("~/.config/trainer")
    TRAINER_CONFIGURE_LOGGING: int = 1
    TRAINER_LOGGING_LEVEL: str = "INFO"
    TRAINER_LOGGING_PREFIX: str = ""
    TRAINER_LOGGING_CONFIG_PATH: str | None = None
    TRAINER_TRACE_FUNCTION: int = 0
    TRAINER_WORKER_MULTIPROC_METHOD: str = "fork"
    TRAINER_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    CMAKE_BUILD_TYPE: str | None = None
    VERBOSE: bool = False
    TRAINER_SERVER_DEV_MODE: bool = False
    TRAINER_STAGE_LOGGING: bool = False


def get_default_cache_root() -> str:
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root() -> str:
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Installation Time Env Vars ==================
    # Target device of Trainer, supporting [cuda (by default),
    # rocm, neuron, cpu, openvino]
    "TRAINER_TARGET_DEVICE": lambda: os.getenv("TRAINER_TARGET_DEVICE", "cuda"),
    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),
    # If set, trainer will use precompiled binaries (*.so)
    "TRAINER_USE_PRECOMPILED": lambda: bool(os.environ.get("TRAINER_USE_PRECOMPILED"))
    or bool(os.environ.get("TRAINER_PRECOMPILED_WHEEL_LOCATION")),
    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # If set, trainer will print verbose logs during installation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # Root directory for TRAINER configuration files
    # Defaults to `~/.config/trainer` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how trainer finds its configuration files
    # during runtime, but also affects how trainer installs its configuration
    # files during **installation**.
    "TRAINER_CONFIG_ROOT": lambda: os.path.expanduser(
        os.getenv(
            "TRAINER_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "trainer"),
        )
    ),
    # ================== Runtime Env Vars ==================
    # Root directory for TRAINER cache files
    # Defaults to `~/.cache/trainer` unless `XDG_CACHE_HOME` is set
    "TRAINER_CACHE_ROOT": lambda: os.path.expanduser(
        os.getenv(
            "TRAINER_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "trainer"),
        )
    ),
    # Interval in seconds to log a warning message when the ring buffer is full
    "TRAINER_RINGBUFFER_WARNING_INTERVAL": lambda: int(
        os.environ.get("TRAINER_RINGBUFFER_WARNING_INTERVAL", "60")
    ),
    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "TRAINER_NCCL_SO_PATH": lambda: os.environ.get("TRAINER_NCCL_SO_PATH", None),
    # when `TRAINER_NCCL_SO_PATH` is not set, trainer will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH": lambda: os.environ.get("LD_LIBRARY_PATH", None),
    # Internal flag to enable Dynamo fullgraph capture
    "TRAINER_TEST_DYNAMO_FULLGRAPH_CAPTURE": lambda: bool(
        os.environ.get("TRAINER_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1") != "0"
    ),
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES": lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),
    # timeout for each iteration in the engine
    "TRAINER_ENGINE_ITERATION_TIMEOUT_S": lambda: int(
        os.environ.get("TRAINER_ENGINE_ITERATION_TIMEOUT_S", "60")
    ),
    # Logging configuration
    # If set to 0, trainer will not configure logging
    # If set to 1, trainer will configure logging using the default configuration
    #    or the configuration file specified by TRAINER_LOGGING_CONFIG_PATH
    "TRAINER_CONFIGURE_LOGGING": lambda: int(os.getenv("TRAINER_CONFIGURE_LOGGING", "1")),
    "TRAINER_LOGGING_CONFIG_PATH": lambda: os.getenv("TRAINER_LOGGING_CONFIG_PATH"),
    # this is used for configuring the default logging level
    "TRAINER_LOGGING_LEVEL": lambda: os.getenv("TRAINER_LOGGING_LEVEL", "INFO"),
    # if set, TRAINER_LOGGING_PREFIX will be prepended to all log messages
    "TRAINER_LOGGING_PREFIX": lambda: os.getenv("TRAINER_LOGGING_PREFIX", ""),
    # Trace function calls
    # If set to 1, trainer will trace function calls
    # Useful for debugging
    "TRAINER_TRACE_FUNCTION": lambda: int(os.getenv("TRAINER_TRACE_FUNCTION", "0")),
    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "TRAINER_WORKER_MULTIPROC_METHOD": lambda: os.getenv("TRAINER_WORKER_MULTIPROC_METHOD", "fork"),
    # Enables torch profiler if set. Path to the directory where torch profiler
    # traces are saved. Note that it must be an absolute path.
    "TRAINER_TORCH_PROFILER_DIR": lambda: (
        None
        if os.getenv("TRAINER_TORCH_PROFILER_DIR", None) is None
        else os.path.expanduser(os.getenv("TRAINER_TORCH_PROFILER_DIR", "."))
    ),
    # If set, trainer will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "TRAINER_SERVER_DEV_MODE": lambda: bool(int(os.getenv("TRAINER_SERVER_DEV_MODE", "0"))),
    # If set, trainer will enable stage logging, which will print the time
    # taken for each stage
    "TRAINER_STAGE_LOGGING": lambda: bool(int(os.getenv("TRAINER_STAGE_LOGGING", "0"))),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
