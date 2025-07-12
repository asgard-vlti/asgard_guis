import PDU
import time

MIMIR_OUTLETS = [1, 8]

pdu = PDU.AtenEcoPDU("192.168.100.11")
pdu.connect()

for outlet in MIMIR_OUTLETS:
    pdu.switch_outlet_status(outlet, "on")

time.sleep(8)

is_on = [False for _ in MIMIR_OUTLETS]
for outlet in MIMIR_OUTLETS:
    status = pdu.read_outlet_status(outlet)
    if status == "on":
        is_on[MIMIR_OUTLETS.index(outlet)] = True

if all(is_on):
    print("Mimir is on, expecting 10 mins to boot")
else:
    print("Mimir is not on, check PDU status")
    for outlet, status in zip(MIMIR_OUTLETS, is_on):
        if not status:
            print(f"Outlet {outlet} is still off")
