#!/usr/bin/python

import sys
import os

# Redirect stuff that would normally go to stdout
# and stderr to a file.
fclient = sys.stdout
logfile = "log_engine.txt"
sys.stdout = sys.stderr = open(logfile, 'w', 0) # 0 = unbuffered

#print "PATH =", os.environ['PATH']
#print "LD_LIBRARY_PATH =", os.environ['LD_LIBRARY_PATH']

from GTP import GTP
from Engine import IdiotEngine
from TFEngine import TFEngine
import Train

#engine = IdiotEngine()
#engine = TFEngine("linear", Train.inference_linear, "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_linear")
engine = TFEngine("conv_conv_full", Train.inference_conv_conv_full, "/home/greg/coding/ML/go/NN/engine/checkpoints/ckpts_conv_conv_full")

gtp = GTP(engine, fclient)

gtp.loop()

