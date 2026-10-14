# a temperature watch dog that uses the controllino, polls it and saves some data
# monitoring only, no PI setting\
# to be run from the base directory of the asgard_alignment package

import time
import asgard_guis.PDU
from asgard_guis.pdu_config import load_pdu_config
import os
import asgard_guis.utils
import argparse
import csv


def str_to_ls(s):
    s = str(s)
    # find [ and ]
    start = s.find("[") + 1
    end = s.find("]")
    return list(s[start:end].split(","))


def monitored_outlets(config):
    """Monitor the MDS and camera outlets, counting shared outlets once."""
    return tuple(
        sorted(set(config.outlets_for("MDS")) | set(config.outlets_for("C-RED")))
    )


def main():
    parser = argparse.ArgumentParser(description="Launch temperature watchdog logger")
    parser.add_argument(
        "--duration", type=float, default=8, help="Duration in hours (default: 8)"
    )
    parser.add_argument(
        "--sampling",
        type=int,
        default=5,
        help="Sampling interval in seconds (default: 5)",
    )
    args = parser.parse_args()

    config = load_pdu_config()
    outlets_of_interest = monitored_outlets(config)

    duration = args.duration * 60 * 60  # convert hours to seconds
    sampling = args.sampling

    cur_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    savepth = os.path.expanduser(
        os.path.join("~", "logs", "templogs", f"tempWD_{cur_datetime}.log")
    )

    print("opening file at", savepth)

    pdu = asgard_guis.PDU.AtenEcoPDU(config.ip)
    pdu.connect()
    print("Connected to PDU")

    mds = asgard_guis.utils.open_socket_connection("MDS")

    keys = str_to_ls(asgard_guis.utils.send_and_get_response(mds, "temp_status keys"))

    start_time = time.time()
    with open(savepth, "w", newline="") as f:
        # temp probes first, followed by PI infos prefixed by servo name
        csv.writer(f).writerow(
            ["Time", *keys]
            + [
                f"outlet{o} ({' / '.join(config.devices_for(o))})"
                for o in outlets_of_interest
            ]
        )
        print("Starting monitoring...")
        try:
            while time.time() - start_time < duration:
                temp_status = asgard_guis.utils.send_and_get_response(
                    mds, "temp_status now"
                )
                temp_status = str_to_ls(temp_status)
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(temp_status)
                f.write(
                    current_time
                    + ","
                    + ",".join(f"{float(temp):.2f}" for temp in temp_status)
                )

                # read PDU currents
                out_vals = []
                for outlet in outlets_of_interest:
                    try:
                        res = pdu.read_power_value("olt", outlet, "curr", "simple")
                        out_vals.append(float(res))
                    except Exception as e:
                        print(f"Error getting outlet info for {outlet}: {e}")
                        out_vals.append(None)
                f.write(",")

                f.write(",".join(f"{o:.2f}" for o in out_vals))

                f.write("\n")
                f.flush()
                print(f"{current_time}: {temp_status}")

                time.sleep(sampling)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Exiting gracefully and saving log.")


if __name__ == "__main__":
    main()
