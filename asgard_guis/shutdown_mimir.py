import PDU
import time

MIMIR_OUTLETS = [1, 8]

pdu = PDU.AtenEcoPDU("192.168.100.11")
pdu.connect()

input("Did you type sudo shutdown -h now on Mimir? Press Enter to continue...")

for outlet in MIMIR_OUTLETS:
    pdu.switch_outlet_status(outlet, "off")

print("Powering off Mimir outlets...")


is_off = [False for _ in MIMIR_OUTLETS]
max_time = 60
start_time = time.time()

while (not all(is_off)) and (time.time() - start_time < max_time):
    for outlet in MIMIR_OUTLETS:
        status = pdu.read_outlet_status(outlet)
        if status == "off":
            is_off[MIMIR_OUTLETS.index(outlet)] = True
            print(f"Outlet {outlet} is off")

    time.sleep(1)

if all(is_off):
    print("Mimir is off")
else:
    print("ERROR: Mimir is not off, check PDU status manually")
    for outlet, status in zip(MIMIR_OUTLETS, is_off):
        if not status:
            print(f"Outlet {outlet} is still on")
