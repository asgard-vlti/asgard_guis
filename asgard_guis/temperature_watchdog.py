# a temperature watch dog that uses the controllino, polls it and saves some data
# monitoring only, no PI setting\
# to be run from the base directory of the asgard_alignment package

import time
import PDU
import os
import utils
import argparse


def str_to_ls(s):
    return list(str(s)[1:-1].split(","))


def main():
    parser = argparse.ArgumentParser(description="Temperature watchdog logger")
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

    duration = args.duration * 60 * 60  # convert hours to seconds
    sampling = args.sampling

    cur_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    savepth = os.path.expanduser(
        os.path.join("~", "logs", "templogs", f"tempWD_{cur_datetime}.log")
    )

    pdu = PDU.AtenEcoPDU("192.168.100.11")
    pdu.connect()
    print("Connected to PDU")

    outlets_of_interest = [5, 6]
    outlet_names = ["MDS", "C RED"]

    mds = utils.open_socket_connection("MDS")

    keys = str_to_ls(utils.send_and_get_response(mds, "temp_status keys"))

    start_time = time.time()
    with open(savepth, "w") as f:
        # temp probes first, followed by PI infos prefixed by servo name
        f.write(
            "Time,"
            + ",".join(keys)
            + ","
            + ",".join(
                [
                    f"outlet{o} ({desc})"
                    for o, desc in zip(outlets_of_interest, outlet_names)
                ]
            )
            + "\n"
        )
        try:
            while time.time() - start_time < duration:
                temp_status = utils.send_and_get_response(mds, "temp_status now")
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write(
                    current_time + "," + ",".join(f"{temp:.2f}" for temp in temp_status)
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
