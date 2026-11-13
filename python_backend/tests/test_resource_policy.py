"""The hardware-adaptive capacity policy, from synthetic machines only."""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services import resource_policy as rp  # noqa: E402

GiB, MiB = rp.GiB, rp.MiB
DEFAULTS = rp.Configuration("auto", None, 0, 0, rp.DEFAULT_DISK_RESERVE_BYTES)
LAPTOP = rp.HardwareSnapshot(8 * GiB, 3 * GiB, 8, 4, 200 * GiB)
WORKSTATION = rp.HardwareSnapshot(64 * GiB, 40 * GiB, 32, 16, 2000 * GiB)
CUDA = rp.HardwareSnapshot(
    32 * GiB, 20 * GiB, 16, 8, 1000 * GiB, True, "synthetic", 16 * GiB
)


def formats(policy):
    return {item["modality"]: item for item in policy.public_dict()["formats"]}


def test_a_laptop_runs_one_extraction_job_and_one_model_request():
    policy = rp.ResourcePolicy(LAPTOP, DEFAULTS)

    assert policy.profile == "laptop"
    assert policy.extraction_concurrency == 1 and policy.upload_concurrency == 2
    assert policy.model_concurrency == 1


def test_a_workstation_scales_jobs_but_never_model_inference():
    policy = rp.ResourcePolicy(WORKSTATION, DEFAULTS)

    assert policy.profile == "workstation"
    assert 1 < policy.extraction_concurrency <= 4
    assert policy.public_dict()["model_inference_concurrency"] == 1


def test_cuda_is_reported_but_not_claimed_as_used():
    body = rp.ResourcePolicy(CUDA, DEFAULTS).public_dict()

    assert body["resource_profile"] == "gpu" and body["cuda_available"] is True
    assert body["cuda_used"] is False and body["model_device"] == "cpu"


def test_large_memory_never_lifts_a_format_above_its_safety_ceiling():
    huge = rp.HardwareSnapshot(1024 * GiB, 900 * GiB, 128, 64, 100_000 * GiB)
    policy = rp.ResourcePolicy(huge, DEFAULTS)

    for name, format_policy in policy.formats.items():
        capacity = policy.capacity(name)
        assert capacity.effective_max_bytes <= format_policy.safety_ceiling_bytes
        assert capacity.limited_by == "format_ceiling"
    assert policy.extraction_concurrency <= 4


def test_the_effective_limit_is_the_smallest_of_its_bounds():
    small = rp.HardwareSnapshot(4 * GiB, 1 * GiB, 4, 2, 200 * GiB)
    policy = rp.ResourcePolicy(small, DEFAULTS)
    workbook = policy.capacity("workbook")

    planning = 4 * GiB - rp.MODEL_RESERVE_BYTES - rp.OS_RESERVE_BYTES
    amplification = policy.formats["workbook"].memory_amplification
    expected = int(planning * rp.MEMORY_PLANNING_SHARE / 1 / amplification)
    assert workbook.limited_by == "memory" and workbook.effective_max_bytes == expected


def test_an_administrative_cap_lowers_every_format():
    capped = rp.ResourcePolicy(
        WORKSTATION, rp.Configuration("auto", 2 * MiB, 0, 0, GiB)
    )

    for item in formats(capped).values():
        assert item["effective_max_bytes"] <= 2 * MiB
    assert capped.capacity("csv").limited_by == "administrative"


def test_low_disk_lowers_the_limit_and_refuses_admission_for_now():
    low = rp.with_free_disk(LAPTOP, rp.DEFAULT_DISK_RESERVE_BYTES + 30 * MiB)
    policy = rp.ResourcePolicy(low, DEFAULTS)

    assert policy.capacity("workbook").limited_by == "disk"
    assert policy.capacity("workbook").effective_max_bytes == 10 * MiB
    # A streamed format needs more disk per byte (input, output and spill), never RAM.
    assert policy.capacity("csv").limited_by == "disk"
    assert policy.capacity("csv").effective_max_bytes == int(30 * MiB / 3.5)
    full = rp.with_free_disk(LAPTOP, rp.DEFAULT_DISK_RESERVE_BYTES)
    assert rp.ResourcePolicy(LAPTOP, DEFAULTS).admit("csv", MiB, full) == (
        "insufficient_disk_retry_later"
    )
    assert rp.ResourcePolicy(full, DEFAULTS).public_dict()["free_disk_admission"] == (
        "insufficient"
    )


def test_admission_refuses_on_live_memory_but_accepts_small_files():
    policy = rp.ResourcePolicy(LAPTOP, DEFAULTS)
    tight = rp.HardwareSnapshot(8 * GiB, 300 * MiB, 8, 4, 200 * GiB)

    assert policy.admit("workbook", MiB, tight) is None
    assert (
        policy.admit("workbook", 20 * MiB, tight) == "insufficient_memory_retry_later"
    )
    assert policy.admit("workbook", 64 * MiB, LAPTOP) == "file_exceeds_capacity"
    assert policy.admit("hologram", 1, LAPTOP) == "format_disabled"
    # Streamed formats hold a bounded working set whatever their size.
    assert policy.admit("csv", 200 * MiB, tight) is None
    assert policy.admit("text", 200 * MiB, tight) is None


def test_only_csv_and_text_stream_and_no_limit_is_claimed_as_tested():
    body = rp.ResourcePolicy(LAPTOP, DEFAULTS).public_dict()

    assert body["anonymization_streaming"] == ["csv", "text"]
    for item in body["formats"]:
        expected = (
            rp.STREAMING if item["modality"] in ("csv", "text") else rp.WHOLE_FILE
        )
        assert item["processing_strategy"] == expected
        assert item["evidence_level"] in (rp.INFERRED, rp.UNVERIFIED)
        assert item["review_download_available"] is True
    csv_entry = formats(rp.ResourcePolicy(LAPTOP, DEFAULTS))["csv"]
    assert csv_entry["anonymization"]["largest_real_model_tested_bytes"] is None
    assert csv_entry["anonymization"]["time_grows_with_size"] is True
    # The configured maximum is far above anything tested; the two stay apart.
    assert csv_entry["anonymization"]["max_bytes"] > 0
    assert csv_entry["anonymization"]["largest_real_model_tested_bytes"] is None


def test_a_snapshot_reporting_no_memory_admits_no_whole_file_format():
    unknown = rp.HardwareSnapshot(0, 0, 1, None, 200 * GiB)
    policy = rp.ResourcePolicy(unknown, DEFAULTS)

    assert not any(
        policy.capacity(name).enabled
        for name in policy.formats
        if name not in ("csv", "text")
    )


def test_the_stream_ceiling_is_configurable_and_validated(monkeypatch):
    monkeypatch.setenv(rp.STREAM_MAX_FILE_ENV_VAR, str(64 * MiB))
    assert rp.ResourcePolicy(WORKSTATION, DEFAULTS).capacity(
        "csv"
    ).effective_max_bytes == (64 * MiB)
    monkeypatch.setenv(rp.STREAM_MAX_FILE_ENV_VAR, "12")
    with pytest.raises(rp.CapacityConfigurationError):
        rp.ResourcePolicy(WORKSTATION, DEFAULTS)


# -- adaptive batch limits ------------------------------------------------------------------------


def test_a_stronger_machine_and_configuration_raise_the_batch_cap():
    laptop = rp.ResourcePolicy(LAPTOP, DEFAULTS).batch_limits()
    stronger = rp.ResourcePolicy(WORKSTATION, DEFAULTS).batch_limits()
    configured = rp.ResourcePolicy(
        WORKSTATION,
        rp.Configuration(
            "auto",
            None,
            0,
            0,
            rp.DEFAULT_DISK_RESERVE_BYTES,
            admin_max_batch_bytes=64 * GiB,
            max_files_per_batch=100,
        ),
    ).batch_limits()

    assert laptop.max_total_bytes == 1 * GiB and laptop.limited_by == "profile_default"
    assert stronger.max_total_bytes > laptop.max_total_bytes
    assert configured.max_total_bytes > stronger.max_total_bytes
    assert configured.max_files == 100 and stronger.max_files > laptop.max_files


def test_low_disk_shrinks_the_batch_cap_and_open_batches_count():
    low = rp.with_free_disk(LAPTOP, rp.DEFAULT_DISK_RESERVE_BYTES + 900 * MiB)
    policy = rp.ResourcePolicy(low, DEFAULTS)
    limits = policy.batch_limits()
    busier = policy.batch_limits(reserved_bytes=100 * MiB)

    assert limits.limited_by == "disk" and limits.max_total_bytes < 1 * GiB
    assert busier.max_total_bytes == max(0, limits.max_total_bytes - 100 * MiB)


def test_the_batch_cap_never_raises_a_per_file_limit():
    policy = rp.ResourcePolicy(
        WORKSTATION,
        rp.Configuration(
            "auto",
            None,
            0,
            0,
            rp.DEFAULT_DISK_RESERVE_BYTES,
            admin_max_batch_bytes=64 * GiB,
        ),
    )

    assert (
        policy.batch_limits().max_total_bytes
        > policy.capacity("csv").effective_max_bytes
    )
    assert policy.admit(
        "csv", policy.capacity("csv").effective_max_bytes + 1, WORKSTATION
    ) == ("file_exceeds_capacity")


def test_capacity_reports_upload_and_anonymization_separately():
    body = rp.ResourcePolicy(LAPTOP, DEFAULTS).public_dict()
    csv_entry = formats(rp.ResourcePolicy(LAPTOP, DEFAULTS))["csv"]

    assert (
        body["max_files_per_batch"] == 20 and body["max_batch_total_bytes"] == 1 * GiB
    )
    assert body["max_concurrent_uploads"] == 2
    assert csv_entry["upload"]["streamed_to_disk"] is True
    assert csv_entry["anonymization"]["processing_strategy"] == rp.STREAMING
    assert csv_entry["anonymization"]["largest_tested_bytes"] is None
    assert csv_entry["anonymization"]["evidence_level"] == rp.INFERRED
    assert body["detection"]["degraded"] is False


# -- detection ------------------------------------------------------------------------------------


def test_detection_tries_each_memory_source_in_order():
    def broken():
        raise OSError("probe failed")

    total, available, source, limitations = rp.detect_memory(
        (
            ("psutil", lambda: None),
            ("os", broken),
            ("third", lambda: (16 * GiB, 9 * GiB)),
        )
    )

    assert (total, available, source, limitations) == (16 * GiB, 9 * GiB, "third", ())


def test_failed_memory_detection_is_degraded_conservative_and_still_admits_small_files():
    total, available, source, limitations = rp.detect_memory((("os", lambda: None),))
    snapshot = rp.HardwareSnapshot(
        total,
        available,
        4,
        None,
        200 * GiB,
        memory_source=source,
        limitations=limitations,
    )
    policy = rp.ResourcePolicy(snapshot, DEFAULTS)
    body = policy.public_dict()

    assert source == "fallback" and snapshot.degraded
    assert body["detection"]["degraded"] is True
    assert body["detection"]["explanations"] and "C:" not in repr(body["detection"])
    assert all(item["evidence_level"] == rp.UNVERIFIED for item in body["formats"])
    assert policy.capacity("csv").effective_max_bytes == rp.DEGRADED_FILE_BYTES
    assert policy.batch_limits().max_total_bytes <= rp.DEGRADED_BATCH_BYTES
    assert policy.admit("csv", 200 * 1024, snapshot) is None
    assert policy.admit("csv", 20 * MiB, snapshot) == "file_exceeds_capacity"


def test_configured_memory_is_used_only_when_measurement_fails(monkeypatch):
    monkeypatch.setenv(rp.ASSUMED_RAM_ENV_VAR, str(16 * GiB))
    total, _available, source, limitations = rp.detect_memory((("os", lambda: None),))

    assert (total, source, limitations) == (
        16 * GiB,
        "configured",
        ("memory_configured",),
    )
    monkeypatch.setenv(rp.ASSUMED_RAM_ENV_VAR, "12")
    with pytest.raises(rp.CapacityConfigurationError):
        rp.detect_memory((("os", lambda: None),))


def test_a_container_limit_lowers_measured_memory(monkeypatch):
    monkeypatch.setattr(rp, "_cgroup_limit", lambda: (2 * GiB, 512 * MiB))

    total, available, source, _ = rp.detect_memory(
        (("os", lambda: (32 * GiB, 20 * GiB)),)
    )

    assert total == 2 * GiB and available == 1536 * MiB and source == "os+cgroup"


def test_cgroup_files_are_read_and_unlimited_is_ignored(tmp_path):
    (tmp_path / "memory.max").write_text("max")
    assert rp._cgroup_limit(tmp_path) is None
    (tmp_path / "memory.max").write_text(str(3 * GiB))
    (tmp_path / "memory.current").write_text(str(GiB))
    assert rp._cgroup_limit(tmp_path) == (3 * GiB, GiB)


def test_unmeasurable_disk_uses_a_conservative_ceiling(tmp_path, monkeypatch):
    def refuse(_path):
        raise OSError("no disk")

    monkeypatch.setattr(rp.shutil, "disk_usage", refuse)
    free, source, limitations = rp.detect_disk(tmp_path)

    assert (free, source, limitations) == (
        rp.CONSERVATIVE_FREE_DISK_BYTES,
        "fallback",
        ("disk_not_measured",),
    )


def test_a_detected_gpu_is_reported_as_detected_not_active():
    present, name, vram = rp.detect_gpu(lambda: "NVIDIA RTX 4090, 24564\n")
    snapshot = rp.HardwareSnapshot(
        64 * GiB, 40 * GiB, 32, 16, 2000 * GiB, present, name, vram
    )
    body = rp.ResourcePolicy(snapshot, DEFAULTS).public_dict()

    assert present and name == "NVIDIA RTX 4090" and vram == 24564 * MiB
    assert body["gpu_status"] == rp.GPU_DETECTED_NOT_ACTIVE
    assert body["cuda_used"] is False and body["model_device"] == "cpu"
    assert rp.detect_gpu(lambda: "") == (False, None, None)
    assert rp.ResourcePolicy(LAPTOP, DEFAULTS).public_dict()["gpu_status"] == (
        rp.GPU_NOT_DETECTED
    )


@pytest.mark.parametrize(
    "name,value",
    [
        (rp.PROFILE_ENV_VAR, "supercomputer"),
        (rp.MAX_UPLOAD_ENV_VAR, "12"),
        (rp.MAX_UPLOAD_ENV_VAR, "lots"),
        (rp.EXTRACTION_CONCURRENCY_ENV_VAR, "0"),
        (rp.EXTRACTION_CONCURRENCY_ENV_VAR, "99"),
        (rp.UPLOAD_CONCURRENCY_ENV_VAR, "-1"),
        (rp.DISK_RESERVE_ENV_VAR, "1"),
    ],
)
def test_invalid_configuration_is_refused(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(rp.CapacityConfigurationError):
        rp.Configuration.from_environment()


def test_explicit_configuration_is_honoured(monkeypatch):
    monkeypatch.setenv(rp.PROFILE_ENV_VAR, "workstation")
    monkeypatch.setenv(rp.EXTRACTION_CONCURRENCY_ENV_VAR, "2")
    monkeypatch.setenv(rp.UPLOAD_CONCURRENCY_ENV_VAR, "3")
    policy = rp.for_snapshot(LAPTOP)

    assert (
        policy.profile,
        policy.extraction_concurrency,
        policy.upload_concurrency,
    ) == ("workstation", 2, 3)


def _gpu_laptop():
    return rp.HardwareSnapshot(
        8 * GiB, 3 * GiB, 8, 4, 200 * GiB, True, "Test GPU", 4 * GiB
    )


def test_acceleration_is_reported_only_when_the_worker_verified_it():
    policy = rp.ResourcePolicy(_gpu_laptop(), DEFAULTS)
    idle = policy.public_dict({"ready": True})
    active = policy.public_dict(
        {
            "ready": True,
            "inference": {
                "acceleration_active": True,
                "selected_device": "cuda",
                "cuda_runtime": "12.1",
                "status": "Hardware acceleration active",
                "model_batch_size": 8,
            },
        }
    )

    assert (
        idle["gpu_status"] == rp.GPU_DETECTED_NOT_ACTIVE and idle["cuda_used"] is False
    )
    assert idle["inference"]["status"] == "Running in standard mode"
    assert active["gpu_status"] == rp.GPU_ACTIVE and active["cuda_used"] is True
    assert active["model_device"] == "cuda"
    assert active["inference"]["inference_concurrency"] == 1


def test_a_degraded_runtime_says_standard_mode_temporarily():
    body = rp.ResourcePolicy(_gpu_laptop(), DEFAULTS).public_dict(
        {
            "ready": True,
            "inference": {
                "acceleration_active": False,
                "degraded": True,
                "selected_device": "cpu",
                "status": "Temporarily using standard mode",
                "reason": "cuda_out_of_memory",
            },
        }
    )

    assert body["gpu_status"] == rp.GPU_DEGRADED and body["cuda_used"] is False
    assert body["inference"]["status"] == "Temporarily using standard mode"


def test_acceleration_never_raises_a_file_or_batch_ceiling():
    policy = rp.ResourcePolicy(_gpu_laptop(), DEFAULTS)
    idle = policy.public_dict({"ready": True})
    active = policy.public_dict(
        {
            "ready": True,
            "inference": {"acceleration_active": True, "selected_device": "cuda"},
        }
    )

    for key in (
        "max_files_per_batch",
        "max_batch_total_bytes",
        "max_concurrent_uploads",
        "model_inference_concurrency",
    ):
        assert idle[key] == active[key]
    assert [f["upload"]["max_bytes"] for f in idle["formats"]] == [
        f["upload"]["max_bytes"] for f in active["formats"]
    ]
