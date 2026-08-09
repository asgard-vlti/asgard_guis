import asgard_guis.PDU
import time

# FIXME have these imported from somewhere common
MIMIR_OUTLETS = [1, 8]
PDU_IP_ADDRESS = "192.168.100.11"


def main():
    """Power on the Mimir PDU outlets and verify that they report on."""

    pdu = asgard_guis.PDU.AtenEcoPDU(PDU_IP_ADDRESS)
    pdu.connect()

    for outlet in MIMIR_OUTLETS:
        pdu.switch_outlet_status(outlet, "on")

    print("Powering on Mimir outlets...")
    time.sleep(10)

    is_on = [False for _ in MIMIR_OUTLETS]
    start_time = time.time()
    max_time = 50

    while (not all(is_on)) and (time.time() - start_time < max_time):
        for outlet in MIMIR_OUTLETS:
            status = pdu.read_outlet_status(outlet)
            if status == "on":
                is_on[MIMIR_OUTLETS.index(outlet)] = True
                print(f"Outlet {outlet} is on")

        time.sleep(1)

    if all(is_on):
        print("Mimir is on, expecting 10 mins to boot")
    else:
        print("ERROR: Mimir is not on, check PDU status manually")
        for outlet, status in zip(MIMIR_OUTLETS, is_on):
            if not status:
                print(f"Outlet {outlet} is not reporting on")


if __name__ == "__main__":
    main()
