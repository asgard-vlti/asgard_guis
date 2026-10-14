"""Load the PDU address and outlet assignments bundled with asgard_guis."""

from dataclasses import dataclass
from importlib import resources
from ipaddress import ip_address
from types import MappingProxyType
from typing import Mapping, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.8-3.10
    import tomli as tomllib


class PDUConfigError(ValueError):
    """The PDU configuration cannot be used safely."""


@dataclass(frozen=True)
class PDUConfig:
    ip: str
    outlets: Mapping[int, Tuple[str, ...]]

    def outlets_for(self, device: str) -> Tuple[int, ...]:
        """Return every outlet assigned to a device, in numeric order."""
        matches = tuple(
            sorted(number for number, devices in self.outlets.items() if device in devices)
        )
        if not matches:
            raise PDUConfigError(f"No PDU outlets configured for {device!r}")
        return matches

    def devices_for(self, outlet: int) -> Tuple[str, ...]:
        """Return the devices assigned to an outlet."""
        try:
            return self.outlets[outlet]
        except KeyError as exc:
            raise PDUConfigError(f"PDU outlet {outlet} is not configured") from exc


def parse_pdu_config(contents: str) -> PDUConfig:
    """Parse and validate the contents of a PDU TOML file."""
    try:
        data = tomllib.loads(contents)
    except tomllib.TOMLDecodeError as exc:
        raise PDUConfigError(f"Invalid PDU TOML: {exc}") from exc

    pdu = data.get("pdu")
    ip = pdu.get("ip") if isinstance(pdu, dict) else None
    if not isinstance(ip, str):
        raise PDUConfigError("PDU configuration needs a [pdu] ip string")
    try:
        ip_address(ip)
    except ValueError as exc:
        raise PDUConfigError(f"Invalid PDU IP address: {ip!r}") from exc

    raw_outlets = data.get("outlets")
    if not isinstance(raw_outlets, dict) or not raw_outlets:
        raise PDUConfigError("PDU configuration needs a nonempty [outlets] table")

    outlets = {}
    for raw_number, raw_devices in raw_outlets.items():
        if not raw_number.isdecimal() or raw_number.startswith("0"):
            raise PDUConfigError(f"Invalid PDU outlet number: {raw_number!r}")
        number = int(raw_number)
        if not 1 <= number <= 999 or number in outlets:
            raise PDUConfigError(f"Invalid or duplicate PDU outlet number: {raw_number!r}")
        if (
            not isinstance(raw_devices, list)
            or not raw_devices
            or any(not isinstance(device, str) or not device.strip() for device in raw_devices)
        ):
            raise PDUConfigError(
                f"PDU outlet {number} needs a nonempty list of device names"
            )
        devices = tuple(raw_devices)
        if len(set(devices)) != len(devices):
            raise PDUConfigError(f"PDU outlet {number} repeats a device name")
        outlets[number] = devices

    return PDUConfig(ip=ip, outlets=MappingProxyType(outlets))


def load_pdu_config() -> PDUConfig:
    """Read the packaged PDU configuration."""
    # TODO: Move this TOML to ~/.config and share it across the instrument repositories.
    try:
        contents = resources.read_text("asgard_guis", "pdu.toml", encoding="utf-8")
    except OSError as exc:
        raise PDUConfigError(f"Could not read packaged PDU configuration: {exc}") from exc
    return parse_pdu_config(contents)
