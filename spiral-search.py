# Base for spiral-search GUI in python

import os


# send image offset command
# (here you have to replace xx by variables giving the x-offsets of
# the spiral-search step, yy by variables giving the y-offsets)

os.system("msgSend wvgvlti issifControl OFFSACQ \"1,xx,yy,0,0,0,2,xx,yy,0,0,0,3,xx,yy,0,0,0,4,xx,yy,0,0,0\" ")



# write pointing-offsets into database
# (here you have to replace 0 by variables corresponding
# to integrated offsets)

os.system("dbWrite \"<alias>mimir.hdlr_x_pof(0)\" 0")
os.system("dbWrite \"<alias>mimir.hdlr_x_pof(1)\" 0")
os.system("dbWrite \"<alias>mimir.hdlr_x_pof(2)\" 0")
os.system("dbWrite \"<alias>mimir.hdlr_x_pof(3)\" 0")
os.system("dbWrite \"<alias>mimir.hdlr_x_pof(0)\" 0")
os.system("dbWrite \"<alias>mimir.hdlr_x_pof(1)\" 0")
os.system("dbWrite \"<alias>mimir.hdlr_x_pof(2)\" 0")
os.system("dbWrite \"<alias>mimir.hdlr_x_pof(3)\" 0")

os

