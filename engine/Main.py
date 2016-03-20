#!/usr/bin/python

import sys

# Redirect stuff that would normally go to stdout
# and stderr to a file.
fclient = sys.stdout
logfile = "log_engine.txt"
sys.stdout = sys.stderr = open(logfile, 'w', 0) # 0 = unbuffered

from GTP import GTP
from Engine import IdiotEngine
from TFEngine import TFEngine
import Train

#engine = IdiotEngine()
engine = TFEngine("linear", Train.inference_linear, "/home/greg/coding/ML/go/NN/engine/ckpts_linear")

gtp = GTP(engine, fclient)

gtp.loop()

