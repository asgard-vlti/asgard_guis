"""Save and restore CRED1 settings across Solarstein mode changes."""

import json
import math
import os
import tempfile
import time
from pathlib import Path

import asgard_guis.utils as agu


SNAPSHOT_PATH = Path("/home/asg/.config/s_labmode_camera_settings.json")
MAX_AGE_SECONDS = 2 * 60 * 60
DEFAULT_FPS = 1000.0
DEFAULT_GAIN = 5
DEFAULT_NBREADS = 1


def _snapshot_path(path):
    return SNAPSHOT_PATH if path is None else Path(path)


def _validate_settings(data):
    if not isinstance(data, dict):
        raise ValueError("camera settings snapshot must be a JSON object")

    gain = data.get("gain")
    fps = data.get("fps")
    saved_at = data.get("saved_at")
    nbreads = data.get("nbreads", DEFAULT_NBREADS)
    if isinstance(gain, bool) or not isinstance(gain, int) or gain <= 0:
        raise ValueError("camera settings snapshot has invalid gain")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("camera settings snapshot has invalid FPS")
    if (
        isinstance(saved_at, bool)
        or not isinstance(saved_at, (int, float))
        or not math.isfinite(saved_at)
        or saved_at <= 0
    ):
        raise ValueError("camera settings snapshot has invalid timestamp")
    if (
        isinstance(nbreads, bool)
        or not isinstance(nbreads, int)
        or nbreads < 1
    ):
        raise ValueError("camera settings snapshot has invalid NDMR read count")
    return {
        "gain": gain,
        "fps": float(fps),
        "saved_at": float(saved_at),
        "nbreads": 1 if nbreads <= 2 else nbreads,
    }


def read_snapshot(path=None):
    with _snapshot_path(path).open(encoding="utf-8") as snapshot_file:
        return _validate_settings(json.load(snapshot_file))


def send_camera_command(socket, command):
    response = agu.send_and_get_response(socket, command)
    if response is None or not response.strip():
        raise RuntimeError(f"camera did not respond to {command!r}")
    if response.strip().lower().startswith(("error", "err:", "failed")):
        raise RuntimeError(f"camera rejected {command!r}: {response}")
    return response.strip()


def save_snapshot_if_absent(socket, path=None):
    path = _snapshot_path(path)
    if path.exists():
        read_snapshot(path)
        return False

    try:
        gain = int(send_camera_command(socket, "get_gain"))
        fps = float(send_camera_command(socket, "get_fps"))
        nbreads = json.loads(send_camera_command(socket, "status"))["nbreads"]
    except ValueError as exc:
        raise ValueError("camera returned invalid gain, FPS, or status") from exc
    except (KeyError, TypeError) as exc:
        raise ValueError("camera status is missing NDMR read count") from exc

    settings = _validate_settings(
        {"gain": gain, "fps": fps, "nbreads": nbreads, "saved_at": time.time()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(settings, temporary_file)
            temporary_file.write("\n")
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            read_snapshot(path)
            return False
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return True


def settings_for_sky(path=None, now=None):
    path = _snapshot_path(path)
    try:
        settings = read_snapshot(path)
    except (OSError, ValueError):
        return DEFAULT_FPS, DEFAULT_GAIN, DEFAULT_NBREADS, "missing or invalid snapshot"

    age = (time.time() if now is None else now) - settings["saved_at"]
    if not 0 <= age <= MAX_AGE_SECONDS:
        return DEFAULT_FPS, DEFAULT_GAIN, DEFAULT_NBREADS, "stale snapshot"
    return settings["fps"], settings["gain"], settings["nbreads"], "saved settings"


def restore_sky_settings(socket, path=None, now=None):
    path = _snapshot_path(path)
    fps, gain, nbreads, source = settings_for_sky(path, now)
    send_camera_command(socket, f"ndmr_mode {nbreads}")
    send_camera_command(socket, f"set_fps {fps}")
    send_camera_command(socket, f"set_gain {gain}")
    path.unlink(missing_ok=True)
    return fps, gain, nbreads, source
