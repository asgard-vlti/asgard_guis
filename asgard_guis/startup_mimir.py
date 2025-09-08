import PDU
import time

MIMIR_OUTLETS = [1, 8]

pdu = PDU.AtenEcoPDU("192.168.100.11")
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
