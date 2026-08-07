import asgard_guis.PDU
import time
import subprocess
import os

# FIXME have these imported from somewhere common
MIMIR_OUTLETS = [1, 8]
PDU_IP_ADDRESS = "192.168.100.11"


def _confirm_or_abort(message):
    conf = input(message)
    if conf == "y" or conf == "Y":
        return True  # Continue
    elif conf == "n" or conf == "N":
        return False  # Abort
    else:
        print(f"Unable to interpret response '{conf}', please try again.")
        return _confirm_or_abort(message)


def main():
    """Shut down Mimir and power off its configured PDU outlets."""

    pdu = asgard_guis.PDU.AtenEcoPDU(PDU_IP_ADDRESS)
    pdu.connect()

    proceed_to_poweroff = True

    # Execute the shutdown command on mimir
    # This requires passwordless sudo shutdown command to be active on mimir
    # To do this, something like the following should be added to visudo:
    # %group_name ALL=(ALL) NOPASSWD: /usr/sbin/shutdown
    # (Can replace %group_name with user_name , noting lack of %)
    # Gather the required env vars for the remsh command
    rhost = os.getenv("RHOST", None)
    ruser = os.getenv("RUSER", None)
    display = os.getenv("DISPLAY", None)
    # Execute the shutdown command
    shutdown_result = None
    if not any(_ is None for _ in [rhost, ruser, display]):
        shutdown_result = subprocess.call(
            [
                "remsh",
                f"{rhost}",
                "-l",
                f"{ruser}",
                "-n",
                f"DISPLAY={display} xterm -e ssh -XC mimir sudo /usr/sbin/shutdown -h now 1>&- 2>&- &",
            ]
        )
    else:
        proceed_to_poweroff = _confirm_or_abort(
            "WARNING: At least one env var required for the shutdown "
            "command to be sent is missing. This means this script "
            "cannot execute the shutdown command on mimir, and "
            "you will need to trigger that manually first. "
            "Do you still want to "
            "attempt the power off? (y/n):"
        )

    if proceed_to_poweroff and (shutdown_result is None or shutdown_result != 0):
        proceed_to_poweroff = _confirm_or_abort(
            f"WARNING: Mimir shutdown may not have occurred "
            f"(return code: {'n/a' if shutdown_result is None else shutdown_result.result}). "
            f"Either verify shutdown is proceeding and/or trigger manually and "
            f"then type Y to proceed, or type N to abort. (y/n):"
        )

    if not proceed_to_poweroff:
        print("ERROR: Mimir power off has been aborted.")
        return

    for outlet in MIMIR_OUTLETS:
        pdu.switch_outlet_status(outlet, "off")


    # Add a 30 second wait to allow shutdown to progress
    print("Waiting 30s to ensure shutdown has completed...")
    time.sleep(30.0)

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


if __name__ == "__main__":
    main()
