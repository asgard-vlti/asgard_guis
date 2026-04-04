"""Send simulated watchdog payloads over ZMQ to the status REP endpoint.

This script mimics the real publisher by opening a ZMQ REQ connection,
sending JSON watchdog payloads, and waiting for ACK responses.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import time

import zmq


def sim_msg() -> str:
    cnt = int(time.time()) % 10000
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    timestamp = now_utc.isoformat().replace("+00:00", "Z")

    payload = {
        "BTT1": {
            "process": "open",
            "status": '{"cnt":' + str(cnt) + ',"flux":119221.0,"tx":1.61,"ty":2.173}',
            "zmq": "open",
        },
        "BTT2": {
            "process": "open",
            "status": '{"cnt":'
            + str(cnt)
            + ',"flux":90724.9,"tx":-5.526,"ty":-10.945}',
            "zmq": "open",
        },
        "BTT3": {
            "process": "open",
            "status": '{"cnt":'
            + str(cnt)
            + ',"flux":27516.6,"tx":35.318,"ty":-20.465}',
            "zmq": "open",
        },
        "BTT4": {
            "process": "open",
            "status": '{"cnt":' + str(cnt) + ',"flux":108671.2,"tx":2.152,"ty":5.303}',
            "zmq": "open",
        },
        "CRED1": {
            "process": "open",
            "status": (
                '{"cam_status":"running","fps":2000.0,"nbreads":1,'
                '"shm_error":false,"skipped_frames":0,"tsig_len":5}'
            ),
            "zmq": "open",
        },
        "DM": {"process": "open", "status": '"running"', "zmq": "open"},
        "Eng gui": {"process": "open", "status": "running"},
        "HDLR": {
            "process": "closed",
            "status": '{"cnt":9534,"locked":false}',
            "zmq": "open",
        },
        "MDS": {
            "process": "open",
            "status": "NACK: Unkown custom command\n",
            "zmq": "open",
        },
        "back_end": {
            "process": "open",
            "status": ('{"reply": {"time": "' + timestamp + '", "content": "OK:"}}'),
            "zmq": "open",
        },
    }

    return json.dumps(payload)


def _new_req_socket(context: zmq.Context, connect_endpoint: str) -> zmq.Socket:
    socket = context.socket(zmq.REQ)
    # Drop unsent messages on close so reconnects are immediate.
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(connect_endpoint)
    return socket


def run_simulator(
    connect_endpoint: str,
    min_period: float,
    max_period: float,
    request_timeout_ms: int,
    request_retries: int,
) -> None:
    context = zmq.Context.instance()
    socket = _new_req_socket(context, connect_endpoint)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    sequence = 0

    logging.info("Watchdog simulator REQ connected to %s", connect_endpoint)

    while True:
        sequence += 1
        payload = sim_msg()
        retries_left = None if request_retries == -1 else request_retries
        delivered = False
        attempt = 0

        while not delivered and (retries_left is None or retries_left > 0):
            attempt += 1
            socket.send_string(payload)
            events = dict(poller.poll(timeout=request_timeout_ms))

            if socket in events and events[socket] == zmq.POLLIN:
                ack = socket.recv_string()
                logging.info(
                    "Sent update #%d (%d bytes), received reply: %s",
                    sequence,
                    len(payload),
                    ack,
                )
                delivered = True
                break

            if retries_left is not None:
                retries_left -= 1
                if retries_left == 0:
                    logging.error(
                        "No ACK for update #%d after %d attempt(s); skipping update",
                        sequence,
                        request_retries,
                    )
                    break

                logging.warning(
                    "No ACK for update #%d within %d ms; retrying (%d retries left)",
                    sequence,
                    request_timeout_ms,
                    retries_left,
                )
            else:
                logging.warning(
                    "No ACK for update #%d within %d ms; retrying (attempt %d, infinite mode)",
                    sequence,
                    request_timeout_ms,
                    attempt,
                )

            poller.unregister(socket)
            socket.close()
            socket = _new_req_socket(context, connect_endpoint)
            poller.register(socket, zmq.POLLIN)

        time.sleep(random.uniform(min_period, max_period))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send simulated watchdog status updates over ZMQ."
    )
    parser.add_argument(
        "--connect-endpoint",
        default="tcp://localhost:7051",
        help="ZMQ endpoint for the status REP server.",
    )
    parser.add_argument(
        "--min-period",
        type=float,
        default=3.0,
        help="Minimum delay between updates in seconds.",
    )
    parser.add_argument(
        "--max-period",
        type=float,
        default=7.0,
        help="Maximum delay between updates in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible send intervals.",
    )
    parser.add_argument(
        "--request-timeout-ms",
        type=int,
        default=2500,
        help="ACK wait timeout per attempt in milliseconds.",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=3,
        help="Number of Lazy Pirate resend attempts before giving up; use -1 to retry indefinitely.",
    )
    args = parser.parse_args()

    if args.min_period <= 0 or args.max_period <= 0:
        raise ValueError("--min-period and --max-period must be > 0")
    if args.min_period > args.max_period:
        raise ValueError("--min-period cannot be greater than --max-period")
    if args.request_timeout_ms <= 0:
        raise ValueError("--request-timeout-ms must be > 0")
    if args.request_retries == 0 or args.request_retries < -1:
        raise ValueError("--request-retries must be > 0 or -1 for infinite retries")

    random.seed(args.seed)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_simulator(
        args.connect_endpoint,
        args.min_period,
        args.max_period,
        args.request_timeout_ms,
        args.request_retries,
    )


if __name__ == "__main__":
    main()
