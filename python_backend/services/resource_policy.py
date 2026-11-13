"""Admission policy for streamed CSV/TXT and bounded whole-file XLSX.

Streaming ceilings are configuration limits, not evidence of tested capacity.
The detector runtime and the artifact's privacy checks still govern release.
Live memory and disk measurements can lower admission below those ceilings.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SCHEMA_VERSION = "capacity-v2"
MiB = 1024 * 1024
GiB = 1024 * MiB

STREAMING = "streaming"
INCREMENTAL = "incremental_page_row_tile"
WHOLE_FILE = "whole_file"
DISABLED = "disabled"
UNKNOWN = "unknown"

TESTED = "tested"
INFERRED = "inferred"
UNVERIFIED = "unverified"

PROFILE_ENV_VAR = "BIOBLOCK_RESOURCE_PROFILE"
PROFILES = ("auto", "laptop", "workstation", "gpu")
MAX_UPLOAD_ENV_VAR = "BIOBLOCK_ADMIN_MAX_UPLOAD_BYTES"
MAX_BATCH_ENV_VAR = "BIOBLOCK_ADMIN_MAX_BATCH_BYTES"
MAX_FILES_ENV_VAR = "BIOBLOCK_BATCH_MAX_FILES"
UPLOAD_CONCURRENCY_ENV_VAR = "BIOBLOCK_BATCH_UPLOAD_CONCURRENCY"
EXTRACTION_CONCURRENCY_ENV_VAR = "BIOBLOCK_BATCH_WORKERS"
DISK_RESERVE_ENV_VAR = "BIOBLOCK_DISK_RESERVE_BYTES"
ASSUMED_RAM_ENV_VAR = "BIOBLOCK_ASSUME_TOTAL_RAM_BYTES"
ASSUMED_DISK_ENV_VAR = "BIOBLOCK_ASSUME_FREE_DISK_BYTES"

#: Memory kept for the loaded Stanford and GLiNER worker, and for the OS and
#: the API process itself. Measured on this project's model worker: ~1.3 GB
#: resident after warm-up.
MODEL_RESERVE_BYTES = 1536 * MiB
OS_RESERVE_BYTES = 1024 * MiB
DEFAULT_DISK_RESERVE_BYTES = 1 * GiB
#: A share of the remaining memory is ever planned for; the rest is margin.
MEMORY_PLANNING_SHARE = 0.5
# Available memory already excludes what the OS and the loaded models use;
# admission keeps only this much of it untouched.
ADMISSION_HEADROOM_BYTES = 256 * MiB

#: Disk kept for the registry, its journal and preview records.
QUEUE_STORAGE_RESERVE_BYTES = 256 * MiB
#: Per staged input byte: the input itself until its file finishes, then an
#: output or an encrypted review copy of about the same size.
STORED_BYTES_PER_INPUT_BYTE = 2
#: Per file being processed: decoded working copies written beside the input.
IN_FLIGHT_BYTES_PER_INPUT_BYTE = 2
#: Batch totals a profile offers before an administrator raises them.
PROFILE_BATCH_BYTES = {"laptop": 1 * GiB, "workstation": 8 * GiB, "gpu": 8 * GiB}
PROFILE_MAX_FILES = {"laptop": 20, "workstation": 50, "gpu": 50}
#: The schema's own bound on files per manifest, whatever is configured.
ABSOLUTE_MAX_FILES = 100

#: What a machine whose memory or disk cannot be measured is assumed to be.
CONSERVATIVE_TOTAL_RAM_BYTES = 4 * GiB
CONSERVATIVE_AVAILABLE_RAM_BYTES = 768 * MiB
CONSERVATIVE_FREE_DISK_BYTES = 2 * GiB
#: In degraded mode no file and no batch exceeds these.
DEGRADED_FILE_BYTES = 8 * MiB
DEGRADED_BATCH_BYTES = 256 * MiB

GPU_NOT_DETECTED = "not_detected"
GPU_DETECTED_NOT_ACTIVE = "detected_not_active"
GPU_ACTIVE = "active"
GPU_DEGRADED = "degraded_to_cpu"

#: Plain explanations of each detection limitation, for the capacity API.
LIMITATION_TEXT = {
    "memory_not_measured": "Memory could not be measured, so conservative limits apply.",
    "memory_configured": "Memory was set by the administrator, not measured.",
    "disk_not_measured": "Free disk space could not be measured, so a conservative "
    "ceiling applies.",
    "disk_configured": "Free disk space was set by the administrator, not measured.",
}


@dataclass(frozen=True)
class HardwareSnapshot:
    total_ram_bytes: int
    available_ram_bytes: int
    logical_cpus: int
    physical_cpus: Optional[int]
    free_disk_bytes: int
    cuda_available: bool = False
    gpu_name: Optional[str] = None
    gpu_vram_bytes: Optional[int] = None
    memory_source: str = "measured"
    disk_source: str = "measured"
    limitations: Tuple[str, ...] = ()

    @property
    def degraded(self) -> bool:
        return any(code.endswith("_not_measured") for code in self.limitations)


@dataclass(frozen=True)
class FormatPolicy:
    """What the code actually does with one format, not what it might one day do."""

    modality: str
    strategy: str
    #: The largest input this handler's own limits accept and its tests
    #: exercise. More RAM never raises a limit past this.
    safety_ceiling_bytes: int
    #: Peak process memory per input byte while one file is processed:
    #: the raw bytes, decoded working form, output and verification copies.
    memory_amplification: float
    #: Disk per input byte: staged input, protected output, review copy.
    disk_multiplier: float
    max_concurrency: int = 1
    checkpoints: bool = False
    preview_copy: bool = False
    #: Evidence for the *limit*: ``tested`` only if a file near it was
    #: processed end to end. Separate from the largest file actually tested.
    evidence: str = INFERRED
    restriction: str = ""
    secondary_limits: Dict[str, int] = field(default_factory=dict)
    #: The largest file of this format anonymized end to end on the real
    #: batch path, with the pinned models.
    largest_tested_bytes: Optional[int] = None
    #: The largest file fully processed through the real release route with
    #: stand-in models: proves bounded memory and complete consumption, not
    #: model quality or model time.
    largest_synthetic_tested_bytes: Optional[int] = None


#: The per-file ceiling for the streamed formats. Their memory does not grow
#: with size, so disk and this administrative bound decide, never RAM.
STREAM_MAX_FILE_ENV_VAR = "BIOBLOCK_STREAM_MAX_FILE_BYTES"
DEFAULT_STREAM_MAX_FILE_BYTES = 8 * GiB
#: Disk per streamed input byte: the staged input, the temporary output, the
#: stored output or sealed review copy, the spill and the previews.
STREAM_DISK_MULTIPLIER = 3.5
#: This preparation does not inherit qualification claims from later reports.
#: Exact measurements are recorded separately for the tested branch and mode.
CSV_LARGEST_SYNTHETIC_BYTES = None
TEXT_LARGEST_SYNTHETIC_BYTES = None
CSV_LARGEST_REAL_MODEL_BYTES = None
TEXT_LARGEST_REAL_MODEL_BYTES = None
WHOLE_FILE_TESTED_BYTES = {"workbook": (None, None)}


def _tested(name: str) -> Dict[str, Optional[int]]:
    real, synthetic = WHOLE_FILE_TESTED_BYTES[name]
    return {"largest_tested_bytes": real, "largest_synthetic_tested_bytes": synthetic}


def stream_max_file_bytes() -> int:
    raw = (os.getenv(STREAM_MAX_FILE_ENV_VAR) or "").strip()
    if not raw:
        return DEFAULT_STREAM_MAX_FILE_BYTES
    if not raw.isdigit() or not MiB <= int(raw) <= 1 << 44:
        raise CapacityConfigurationError("invalid_bioblock_stream_max_file_bytes")
    return int(raw)


def _format_policies() -> Dict[str, FormatPolicy]:
    MAX_UPLOAD_BYTES = {"workbook": 32 * MiB}

    whole = "The handler reads the whole file into memory."
    streamed = (
        "Read, anonymized, verified and audited as a stream: memory stays bounded "
        "while time grows with size. Large free-text files take much longer."
    )
    ceiling = stream_max_file_bytes()
    return {
        "csv": FormatPolicy(
            "csv",
            STREAMING,
            ceiling,
            0.0,
            STREAM_DISK_MULTIPLIER,
            restriction=streamed,
            largest_tested_bytes=CSV_LARGEST_REAL_MODEL_BYTES,
            largest_synthetic_tested_bytes=CSV_LARGEST_SYNTHETIC_BYTES or None,
        ),
        "text": FormatPolicy(
            "text",
            STREAMING,
            ceiling,
            0.0,
            STREAM_DISK_MULTIPLIER,
            restriction=streamed,
            largest_tested_bytes=TEXT_LARGEST_REAL_MODEL_BYTES or None,
            largest_synthetic_tested_bytes=TEXT_LARGEST_SYNTHETIC_BYTES or None,
        ),
        "workbook": FormatPolicy(
            "workbook",
            WHOLE_FILE,
            MAX_UPLOAD_BYTES["workbook"],
            25.0,
            3.0,
            restriction=whole,
            **_tested("workbook"),
        ),
    }


# -- detection ----------------------------------------------------------------------------------


def _psutil_memory() -> Optional[Tuple[int, int]]:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except Exception:
        return None


def _os_memory() -> Optional[Tuple[int, int]]:
    if sys.platform == "win32":

        class Status(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong),
                ("available", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended", ctypes.c_ulonglong),
            ]

        try:
            status = Status()
            status.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total), int(status.available)
        except (AttributeError, OSError):
            pass
        return None
    try:
        values = {}
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                key, _, rest = line.partition(":")
                values[key] = int(rest.split()[0]) * 1024
        if values.get("MemTotal"):
            return values["MemTotal"], values.get(
                "MemAvailable", values.get("MemFree", 0)
            )
    except (OSError, ValueError, IndexError):
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page
        available = os.sysconf("SC_AVPHYS_PAGES") * page
        if total > 0:
            return int(total), int(available)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _cgroup_limit(root: Path = Path("/sys/fs/cgroup")) -> Optional[Tuple[int, int]]:
    """(limit, used) of this process's memory cgroup, when a finite one applies."""
    for limit_name, used_name in (
        ("memory.max", "memory.current"),
        ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes"),
    ):
        try:
            raw = (root / limit_name).read_text(encoding="ascii").strip()
            if not raw.isdigit():
                continue
            limit = int(raw)
            # cgroup v1 reports "no limit" as a huge page-aligned number.
            if limit <= 0 or limit >= 1 << 60:
                continue
            used = int((root / used_name).read_text(encoding="ascii").strip() or 0)
            return limit, used
        except (OSError, ValueError):
            continue
    return None


def _validated_bytes(name: str, minimum: int) -> Optional[int]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    if not raw.isdigit() or int(raw) < minimum:
        raise CapacityConfigurationError(f"invalid_{name.lower()}")
    return int(raw)


def detect_memory(probes=None) -> Tuple[int, int, str, Tuple[str, ...]]:
    """(total, available, source, limitations), trying each source in order."""
    probes = (
        probes
        if probes is not None
        else (("psutil", _psutil_memory), ("os", _os_memory))
    )
    measured = None
    source = ""
    for name, probe in probes:
        try:
            measured = probe()
        except Exception:
            measured = None
        if measured and measured[0] > 0:
            source = name
            break
    if measured and measured[0] > 0:
        total, available = measured
        container = _cgroup_limit()
        if container and container[0] < total:
            total = container[0]
            available = max(0, min(available, container[0] - container[1]))
            source += "+cgroup"
        return total, available, source, ()
    configured = _validated_bytes(ASSUMED_RAM_ENV_VAR, 2 * GiB)
    if configured:
        # Known, not measured: available memory is taken conservatively.
        return (
            configured,
            min(configured // 4, CONSERVATIVE_AVAILABLE_RAM_BYTES * 2),
            "configured",
            ("memory_configured",),
        )
    return (
        CONSERVATIVE_TOTAL_RAM_BYTES,
        CONSERVATIVE_AVAILABLE_RAM_BYTES,
        "fallback",
        ("memory_not_measured",),
    )


def detect_disk(directory: Path) -> Tuple[int, str, Tuple[str, ...]]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return int(shutil.disk_usage(str(directory)).free), "measured", ()
    except (OSError, ValueError):
        pass
    configured = _validated_bytes(ASSUMED_DISK_ENV_VAR, 256 * MiB)
    if configured:
        return configured, "configured", ("disk_configured",)
    return CONSERVATIVE_FREE_DISK_BYTES, "fallback", ("disk_not_measured",)


def detect_gpu(runner=None) -> Tuple[bool, Optional[str], Optional[int]]:
    """Whether an NVIDIA GPU is present. No GPU library is imported to find out."""
    if runner is None:
        if shutil.which("nvidia-smi") is None:
            return False, None, None

        def runner():
            return subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            ).stdout

    try:
        first = (runner() or "").strip().splitlines()[0]
    except (IndexError, OSError, subprocess.SubprocessError):
        return False, None, None
    name, _, memory = first.partition(",")
    clean = "".join(ch for ch in name.strip() if ch.isalnum() or ch in " -_.")[:64]
    try:
        vram = int(float(memory.strip())) * MiB
    except ValueError:
        vram = None
    return bool(clean), clean or None, vram


def detect(data_directory: Path, gpu: bool = True) -> HardwareSnapshot:
    """The machine as it is now, and which of its numbers were actually measured."""
    total, available, memory_source, memory_limits = detect_memory()
    free, disk_source, disk_limits = detect_disk(data_directory)
    present, name, vram = detect_gpu() if gpu else (False, None, None)
    return HardwareSnapshot(
        total_ram_bytes=total,
        available_ram_bytes=available,
        logical_cpus=os.cpu_count() or 1,
        physical_cpus=None,
        free_disk_bytes=free,
        # The model worker's detectors are pinned to CPU (device=-1): a GPU
        # found here is reported, never used.
        cuda_available=present,
        gpu_name=name,
        gpu_vram_bytes=vram,
        memory_source=memory_source,
        disk_source=disk_source,
        limitations=memory_limits + disk_limits,
    )


# -- configuration ------------------------------------------------------------------------------


class CapacityConfigurationError(ValueError):
    pass


def _int_env(name: str, default: int, low: int, high: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    if not raw.isdigit() or not low <= int(raw) <= high:
        raise CapacityConfigurationError(f"invalid_{name.lower()}")
    return int(raw)


@dataclass(frozen=True)
class Configuration:
    profile: str
    admin_max_bytes: Optional[int]
    upload_concurrency: int
    extraction_concurrency: int
    disk_reserve_bytes: int
    admin_max_batch_bytes: Optional[int] = None
    max_files_per_batch: int = 0

    @classmethod
    def from_environment(cls) -> "Configuration":
        profile = (os.getenv(PROFILE_ENV_VAR) or "auto").strip().lower()
        if profile not in PROFILES:
            raise CapacityConfigurationError("invalid_resource_profile")
        return cls(
            profile=profile,
            admin_max_bytes=_validated_bytes(MAX_UPLOAD_ENV_VAR, MiB),
            upload_concurrency=_int_env(UPLOAD_CONCURRENCY_ENV_VAR, 0, 1, 16),
            extraction_concurrency=_int_env(EXTRACTION_CONCURRENCY_ENV_VAR, 0, 1, 8),
            disk_reserve_bytes=_int_env(
                DISK_RESERVE_ENV_VAR, DEFAULT_DISK_RESERVE_BYTES, 64 * MiB, 1 << 50
            ),
            admin_max_batch_bytes=_validated_bytes(MAX_BATCH_ENV_VAR, MiB),
            max_files_per_batch=_int_env(MAX_FILES_ENV_VAR, 0, 1, ABSOLUTE_MAX_FILES),
        )


def _resolved_profile(configuration: Configuration, hardware: HardwareSnapshot) -> str:
    if configuration.profile != "auto":
        return configuration.profile
    if hardware.degraded:
        return "laptop"
    if hardware.cuda_available and hardware.total_ram_bytes >= 24 * GiB:
        return "gpu"
    return "workstation" if hardware.total_ram_bytes >= 24 * GiB else "laptop"


def _concurrency(
    configuration: Configuration, hardware: HardwareSnapshot, profile: str
) -> tuple:
    """(uploads, extraction jobs). Model inference is always one: one serialized worker."""
    planning = max(0, hardware.total_ram_bytes - MODEL_RESERVE_BYTES - OS_RESERVE_BYTES)
    by_memory = max(1, int(planning // (4 * GiB)))
    by_cpu = max(1, (hardware.logical_cpus or 1) // 4)
    default_extraction = (
        1 if profile == "laptop" or hardware.degraded else min(4, by_memory, by_cpu)
    )
    extraction = configuration.extraction_concurrency or default_extraction
    uploads = configuration.upload_concurrency or (2 if profile == "laptop" else 4)
    return uploads, extraction


# -- the policy ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class FormatCapacity:
    modality: str
    enabled: bool
    strategy: str
    effective_max_bytes: int
    limited_by: str
    evidence: str
    restriction: str
    preview: bool
    secondary_limits: Dict[str, int]
    largest_tested_bytes: Optional[int] = None
    largest_synthetic_tested_bytes: Optional[int] = None


@dataclass(frozen=True)
class BatchLimits:
    max_files: int
    max_total_bytes: int
    limited_by: str
    max_concurrent_uploads: int


class ResourcePolicy:
    """Advertised capacity from a stable snapshot, admission from a live one."""

    def __init__(
        self,
        hardware: HardwareSnapshot,
        configuration: Configuration,
        formats: Optional[Dict[str, FormatPolicy]] = None,
    ):
        self.hardware = hardware
        self.configuration = configuration
        self.formats = formats if formats is not None else _format_policies()
        self.profile = _resolved_profile(configuration, hardware)
        self.upload_concurrency, self.extraction_concurrency = _concurrency(
            configuration, hardware, self.profile
        )
        self.model_concurrency = 1

    def _memory_limit(self, policy: FormatPolicy, total_ram: int) -> int:
        planning = max(0, total_ram - MODEL_RESERVE_BYTES - OS_RESERVE_BYTES)
        per_job = planning * MEMORY_PLANNING_SHARE / max(1, self.extraction_concurrency)
        return int(per_job / policy.memory_amplification)

    def _disk_limit(self, policy: FormatPolicy, free_disk: int) -> int:
        usable = max(0, free_disk - self.configuration.disk_reserve_bytes)
        return int(
            usable / (policy.disk_multiplier * max(1, self.extraction_concurrency))
        )

    def capacity(self, modality: str) -> FormatCapacity:
        policy = self.formats.get(modality)
        if policy is None or policy.strategy in (DISABLED, UNKNOWN):
            return FormatCapacity(
                modality,
                False,
                policy.strategy if policy else UNKNOWN,
                0,
                "disabled",
                UNVERIFIED,
                policy.restriction if policy else "Not supported.",
                False,
                {},
            )
        candidates = {
            "format_ceiling": policy.safety_ceiling_bytes,
            "disk": self._disk_limit(policy, self.hardware.free_disk_bytes),
        }
        if self.configuration.admin_max_bytes is not None:
            candidates["administrative"] = self.configuration.admin_max_bytes
        if policy.strategy != STREAMING:
            candidates["memory"] = self._memory_limit(
                policy, self.hardware.total_ram_bytes
            )
        if self.hardware.degraded:
            candidates["degraded_detection"] = DEGRADED_FILE_BYTES
        limited_by, limit = min(candidates.items(), key=lambda item: (item[1], item[0]))
        evidence = UNVERIFIED if self.hardware.degraded else policy.evidence
        return FormatCapacity(
            modality,
            limit > 0,
            policy.strategy,
            max(0, limit),
            limited_by,
            evidence,
            policy.restriction,
            policy.preview_copy,
            dict(policy.secondary_limits),
            policy.largest_tested_bytes,
            policy.largest_synthetic_tested_bytes,
        )

    def largest_file_bytes(self) -> int:
        return max(
            (self.capacity(name).effective_max_bytes for name in self.formats),
            default=0,
        )

    def batch_limits(
        self, live: Optional[HardwareSnapshot] = None, reserved_bytes: int = 0
    ) -> BatchLimits:
        """How many files and bytes one new batch may declare, given what is already staged."""
        live = live or self.hardware
        max_files = (
            self.configuration.max_files_per_batch or PROFILE_MAX_FILES[self.profile]
        )
        largest = self.largest_file_bytes()
        usable = (
            live.free_disk_bytes
            - self.configuration.disk_reserve_bytes
            - QUEUE_STORAGE_RESERVE_BYTES
            - reserved_bytes * STORED_BYTES_PER_INPUT_BYTE
            - self.extraction_concurrency * largest * IN_FLIGHT_BYTES_PER_INPUT_BYTE
        )
        candidates = {
            "disk": max(0, int(usable // STORED_BYTES_PER_INPUT_BYTE)),
            "files_times_file_limit": max_files * largest,
        }
        if self.configuration.admin_max_batch_bytes is not None:
            candidates["administrative"] = self.configuration.admin_max_batch_bytes
        else:
            candidates["profile_default"] = PROFILE_BATCH_BYTES[self.profile]
        if live.degraded or self.hardware.degraded:
            candidates["degraded_detection"] = DEGRADED_BATCH_BYTES
        limited_by, total = min(candidates.items(), key=lambda item: (item[1], item[0]))
        return BatchLimits(
            max_files, max(0, total), limited_by, self.upload_concurrency
        )

    def admit(
        self,
        modality: str,
        size_bytes: int,
        live: HardwareSnapshot,
        active_bytes: int = 0,
    ) -> Optional[str]:
        """None when this file may be taken now; otherwise a retryable or final code."""
        capacity = self.capacity(modality)
        if not capacity.enabled:
            return "format_disabled"
        if size_bytes > capacity.effective_max_bytes:
            return "file_exceeds_capacity"
        policy = self.formats[modality]
        disk_needed = int((size_bytes + active_bytes) * policy.disk_multiplier)
        if live.free_disk_bytes - self.configuration.disk_reserve_bytes < disk_needed:
            return "insufficient_disk_retry_later"
        memory_needed = int(size_bytes * policy.memory_amplification)
        if policy.strategy != STREAMING and (
            live.available_ram_bytes - ADMISSION_HEADROOM_BYTES < memory_needed
        ):
            return "insufficient_memory_retry_later"
        return None

    def public_dict(
        self,
        model_state: Optional[Dict[str, Any]] = None,
        live: Optional[HardwareSnapshot] = None,
        reserved_bytes: int = 0,
    ) -> Dict[str, Any]:
        live = live or self.hardware
        disk_ok = live.free_disk_bytes > self.configuration.disk_reserve_bytes
        batch = self.batch_limits(live, reserved_bytes)
        limitations = sorted(set(self.hardware.limitations) | set(live.limitations))
        inference = dict((model_state or {}).get("inference") or {})
        active = bool(inference.get("acceleration_active"))
        degraded = bool(inference.get("degraded"))
        return {
            "schema_version": SCHEMA_VERSION,
            "resource_profile": self.profile,
            # Where the models actually run, as the worker verified it; CPU
            # when nothing has said otherwise.
            "model_device": inference.get("selected_device") or "cpu",
            "gpu_status": (
                GPU_ACTIVE
                if active
                else (
                    GPU_DEGRADED
                    if degraded
                    else (
                        GPU_DETECTED_NOT_ACTIVE
                        if self.hardware.cuda_available
                        else GPU_NOT_DETECTED
                    )
                )
            ),
            "cuda_available": self.hardware.cuda_available,
            "cuda_used": active,
            "model_ready": bool((model_state or {}).get("ready")),
            # Speed only: no file or batch ceiling depends on the device.
            "inference": {
                "status": inference.get("status") or "Running in standard mode",
                "acceleration_active": active,
                "degraded": degraded,
                "selected_device": inference.get("selected_device") or "cpu",
                "cuda_runtime": inference.get("cuda_runtime") if active else None,
                "model_batch_size": inference.get("model_batch_size"),
                "inference_concurrency": 1,
                "reason": inference.get("reason"),
            },
            "detection": {
                "degraded": self.hardware.degraded or live.degraded,
                "memory_source": self.hardware.memory_source,
                "disk_source": live.disk_source,
                "limitations": limitations,
                "explanations": [
                    LIMITATION_TEXT[code]
                    for code in limitations
                    if code in LIMITATION_TEXT
                ],
            },
            "max_files_per_batch": batch.max_files,
            "max_batch_total_bytes": batch.max_total_bytes,
            "batch_total_limited_by": batch.limited_by,
            "max_concurrent_uploads": batch.max_concurrent_uploads,
            "recommended_concurrent_uploads": self.upload_concurrency,
            "recommended_concurrent_extraction_jobs": self.extraction_concurrency,
            "model_inference_concurrency": self.model_concurrency,
            "free_disk_admission": "ok" if disk_ok else "insufficient",
            # Only the formats whose strategy says so; the others still read
            # each file whole.
            "anonymization_streaming": sorted(
                name
                for name, policy in self.formats.items()
                if policy.strategy == STREAMING
            ),
            "formats": [
                self._format_entry(item)
                for item in (self.capacity(name) for name in sorted(self.formats))
            ],
        }

    @staticmethod
    def _format_entry(item: FormatCapacity) -> Dict[str, Any]:
        return {
            "modality": item.modality,
            "enabled": item.enabled,
            # A file is accepted only if it can be protected, so the upload
            # limit is the anonymization limit; upload itself is streamed.
            "upload": {
                "supported": item.enabled,
                "max_bytes": item.effective_max_bytes,
                "streamed_to_disk": True,
                "evidence_level": item.evidence,
                "largest_tested_bytes": item.largest_tested_bytes,
            },
            "anonymization": {
                "supported": item.enabled,
                "max_bytes": item.effective_max_bytes,
                "processing_strategy": item.strategy,
                "limited_by": item.limited_by,
                "evidence_level": item.evidence,
                "largest_tested_bytes": item.largest_tested_bytes,
                "largest_real_model_tested_bytes": item.largest_tested_bytes,
                "largest_synthetic_tested_bytes": item.largest_synthetic_tested_bytes,
                "time_grows_with_size": item.strategy == STREAMING,
                "secondary_limits": item.secondary_limits,
                "restriction_reason": item.restriction,
            },
            "processing_strategy": item.strategy,
            "effective_max_bytes": item.effective_max_bytes,
            "limited_by": item.limited_by,
            "secondary_limits": item.secondary_limits,
            "preview_available": item.preview,
            "review_download_available": True,
            "evidence_level": item.evidence,
            "restriction_reason": item.restriction,
            "largest_tested_bytes": item.largest_tested_bytes,
        }


def for_snapshot(
    hardware: HardwareSnapshot, configuration: Optional[Configuration] = None
) -> ResourcePolicy:
    return ResourcePolicy(hardware, configuration or Configuration.from_environment())


def with_free_disk(
    hardware: HardwareSnapshot, free_disk_bytes: int
) -> HardwareSnapshot:
    return replace(hardware, free_disk_bytes=free_disk_bytes)
