import PDU
import time

MIMIR_OUTLETS = [1, 8]

pdu = PDU.AtenEcoPDU("192.168.100.11")
pdu.connect()

for outlet in MIMIR_OUTLETS:
    pdu.switch_outlet_status(outlet, "off")

time.sleep(8)

is_off = [False for _ in MIMIR_OUTLETS]
for outlet in MIMIR_OUTLETS:
    status = pdu.read_outlet_status(outlet)
    if status == "off":
        is_off[MIMIR_OUTLETS.index(outlet)] = True

if all(is_off):
    print("Mimir is off")
else:
    print("Mimir is not off, check PDU status")
    for outlet, status in zip(MIMIR_OUTLETS, is_off):
        if not status:
            print(f"Outlet {outlet} is still on")
