# Base for spiral-search GUI in python

import os


class SpiralSearchIntegrator:
    def __init__(self, debug=False):
        self.accumulated_offsets = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}
        self.debug = debug

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

        if self.debug:
            print(f"[DEBUG] Would run: {cmd}")
        else:
            os.system(cmd)

    def write_pointing_offsets_to_db(self):
        # Write accumulated offsets into database for beams 0-3
        for i in range(4):
            x_offset = self.accumulated_offsets[i + 1][0]
            y_offset = self.accumulated_offsets[i + 1][1]
            cmd_x = f'dbWrite "<alias>mimir.hdlr_x_pof({i})" {x_offset}'
            cmd_y = f'dbWrite "<alias>mimir.hdlr_y_pof({i})" {y_offset}'
            if self.debug:
                print(f"[DEBUG] Would run: {cmd_x}")
                print(f"[DEBUG] Would run: {cmd_y}")
            else:
                os.system(cmd_x)
                print(f"Sent: {cmd_x}")
                os.system(cmd_y)
                print(f"Sent: {cmd_y}")


class PupilSearchIntegrator:
    def __init__(self, debug=False):
        self.accumulated_offsets = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}
        self.debug = debug

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
