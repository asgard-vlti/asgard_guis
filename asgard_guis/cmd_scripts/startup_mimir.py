import asgard_guis.PDU
from asgard_guis.pdu_config import load_pdu_config
import time


def main():
    """Power on the Mimir PDU outlets and verify that they report on."""

    config = load_pdu_config()
    mimir_outlets = config.outlets_for("Mimir")
    pdu = asgard_guis.PDU.AtenEcoPDU(config.ip)
    pdu.connect()

    for outlet in mimir_outlets:
        pdu.switch_outlet_status(outlet, "on")

    print("Powering on Mimir outlets...")
    time.sleep(10)

    is_on = [False for _ in mimir_outlets]
    start_time = time.time()
    max_time = 50

    while (not all(is_on)) and (time.time() - start_time < max_time):
        for outlet in mimir_outlets:
            status = pdu.read_outlet_status(outlet)
            if status == "on":
                is_on[mimir_outlets.index(outlet)] = True
                print(f"Outlet {outlet} is on")

        time.sleep(1)

    if all(is_on):
        print("--- Mimir is on, expecting 10 mins to boot ---")
    else:
        print("ERROR: Mimir is not on, check PDU status manually")
        for outlet, status in zip(mimir_outlets, is_on):
            if not status:
                print(f"Outlet {outlet} is not reporting on")


if __name__ == "__main__":
    main()
