# Base for spiral-search GUI in python

import logging
import os
from datetime import datetime
from pathlib import Path


def _setup_offset_logger(logging_folder, logger_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("/vltuser/asg/logs/") / logging_folder
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{logger_name}_{timestamp}.log"
    print(f"Logging {logger_name} offsets to: {log_file}")

    logger = logging.getLogger(f"{logger_name}_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger, log_file


class SpiralSearchIntegrator:
    def __init__(self, debug=False, logging_folder="."):
        self.accumulated_offsets = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}
        self.debug = debug
        self.logger, self.log_file = _setup_offset_logger(
            logging_folder, "spiral_search"
        )
        self.logger.info("SpiralSearchIntegrator started")
        self.logger.info("Initial accumulated offsets: %s", self.accumulated_offsets)

    def send_image_relative_offset(self, beam, x, y):
        # Prepare offsets for each beam (1-4)
        offsets = []
        for i in range(1, 5):
            if i == beam:
                offsets.extend([str(x), str(y), "0", "0", "0"])
            else:
                offsets.extend(["0", "0", "0", "0", "0"])
        # Build the command string
        cmd = (
            'msgSend wvgvlti issifControl OFFSACQ "'
            + ",".join([f"{i},{','.join(offsets[(i-1)*5:i*5])}" for i in range(1, 5)])
            + '"'
        )

        self.accumulated_offsets[beam][0] += x
        self.accumulated_offsets[beam][1] += y

        print(self.accumulated_offsets)
        self.logger.info(
            "Beam %s move (x=%s, y=%s) | accumulated offsets: %s",
            beam,
            x,
            y,
            self.accumulated_offsets,
        )

        if self.debug:
            print(f"[DEBUG] Would run: {cmd}")
        else:
            os.system(cmd)

    def write_pointing_offsets_to_db(self):
        # Write accumulated offsets into database for beams 0-3
        self.logger.info("Writing pointing offsets to DB: %s", self.accumulated_offsets)
        for i in range(4):
            x_offset = self.accumulated_offsets[i + 1][0]
            y_offset = self.accumulated_offsets[i + 1][1]
            cmd_x = f'dbWrite "<alias>mimir.hdlr_x_pof({i})" {x_offset}'
            cmd_y = f'dbWrite "<alias>mimir.hdlr_y_pof({i})" {y_offset}'
            self.logger.info(
                "Beam index %s DB write values: x_offset=%s, y_offset=%s",
                i,
                x_offset,
                y_offset,
            )
            if self.debug:
                print(f"[DEBUG] Would run: {cmd_x}")
                print(f"[DEBUG] Would run: {cmd_y}")
            else:
                os.system(cmd_x)
                print(f"Sent: {cmd_x}")
                os.system(cmd_y)
                print(f"Sent: {cmd_y}")


class PupilSearchIntegrator:
    def __init__(self, debug=False, logging_folder="."):
        self.accumulated_offsets = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}
        self.debug = debug
        self.logger, self.log_file = _setup_offset_logger(logging_folder, "pupil_search")
        self.logger.info("PupilSearchIntegrator started")
        self.logger.info("Initial accumulated offsets: %s", self.accumulated_offsets)

    def send_image_relative_offset(self, beam, x, y):
        # Prepare offsets for each beam (1-4)
        offsets = []
        for i in range(1, 5):
            if i == beam:
                offsets.extend(["0", "0", str(x), str(y), "0"])
            else:
                offsets.extend(["0", "0", "0", "0", "0"])
        # Build the command string
        cmd = (
            'msgSend wvgvlti issifControl OFFSACQ "'
            + ",".join([f"{i},{','.join(offsets[(i-1)*5:i*5])}" for i in range(1, 5)])
            + '"'
        )

        self.accumulated_offsets[beam][0] += x
        self.accumulated_offsets[beam][1] += y

        print(self.accumulated_offsets)
        self.logger.info(
            "Beam %s move (x=%s, y=%s) | accumulated offsets: %s",
            beam,
            x,
            y,
            self.accumulated_offsets,
        )

        if self.debug:
            print(f"[DEBUG] Would run: {cmd}")
        else:
            os.system(cmd)

    # def write_pointing_offsets_to_db(self):
    #     # Write accumulated offsets into database for beams 0-3
    #     for i in range(4):
    #         x_offset = self.accumulated_offsets[i + 1][0]
    #         y_offset = self.accumulated_offsets[i + 1][1]
    #         cmd_x = f'dbWrite "<alias>mimir.hdlr_x_pof({i})" {x_offset}'
    #         cmd_y = f'dbWrite "<alias>mimir.hdlr_y_pof({i})" {y_offset}'
    #         if self.debug:
    #             print(f"[DEBUG] Would run: {cmd_x}")
    #             print(f"[DEBUG] Would run: {cmd_y}")
    #         else:
    #             os.system(cmd_x)
    #             os.system(cmd_y)
