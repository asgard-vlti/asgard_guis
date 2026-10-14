"""Read the status of every outlet listed in the packaged PDU configuration."""

from asgard_guis.PDU import AtenEcoPDU
from asgard_guis.pdu_config import load_pdu_config


def main():
    """Print one outlet number and status per line."""
    config = load_pdu_config()
    pdu = AtenEcoPDU(config.ip)
    if not pdu.connect(verbose=False):
        return 1

    failed = False
    try:
        for outlet in sorted(config.outlets):
            status = pdu.read_outlet_status(outlet)
            if not status:
                status = "unknown"
                failed = True
            print(f"{outlet}: {status}")
    finally:
        pdu.close(verbose=False)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
