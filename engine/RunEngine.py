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
import Models

#engine = IdiotEngine()
#engine = TFEngine("linear", Models.Linear(N=9, Nfeat=12, minibatch_size=1000, learning_rate=0))
#engine = TFEngine("conv3full", Models.Conv3Full(N=9, Nfeat=12, minibatch_size=1000, learning_rate=0))
#engine = TFEngine("conv4full", Models.Conv4Full(N=9, Nfeat=16, minibatch_size=1000, learning_rate=0))
#engine = TFEngine("conv5full", Models.Conv5Full(N=9, Nfeat=16, minibatch_size=1000, learning_rate=0))
engine = TFEngine("conv8", Models.Conv8(N=19, Nfeat=16))

gtp = GTP(engine, fclient)

gtp.loop()

