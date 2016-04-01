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
from KGSEngine import KGSEngine
import MoveModels
from Book import PositionRecord
from Book import MoveRecord

#engine = IdiotEngine()
#engine = TFEngine("linear", Models.Linear(N=9, Nfeat=12, minibatch_size=1000, learning_rate=0))
#engine = TFEngine("conv3full", Models.Conv3Full(N=9, Nfeat=12, minibatch_size=1000, learning_rate=0))
#engine = TFEngine("conv4full", Models.Conv4Full(N=9, Nfeat=16, minibatch_size=1000, learning_rate=0))
#engine = TFEngine("conv5full", Models.Conv5Full(N=9, Nfeat=16, minibatch_size=1000, learning_rate=0))
#engine = TFEngine("conv8", Models.Conv8(N=19, Nfeat=16))
#engine = TFEngine("conv12", Models.Conv12(N=19, Nfeat=16))
#engine = TFEngine("firstmovetest", Models.FirstMoveTest(19, 15))
#engine = TFEngine("conv6posdep", Models.Conv6PosDep(N=19, Nfeat=15))
#engine = TFEngine("conv8posdep", Models.Conv8PosDep(N=19, Nfeat=15))
#engine = TFEngine("conv10posdep", Models.Conv10PosDep(N=19, Nfeat=15))
#engine = TFEngine("conv10posdep", Models.Conv10PosDep(N=19, Nfeat=15))
###engine = TFEngine("conv10posdepELU", MoveModels.Conv10PosDepELU(N=19, Nfeat=15))
#engine = TFEngine("conv12posdep", Models.Conv12PosDep(N=19, Nfeat=15))

engine = KGSEngine(TFEngine("conv10posdepELU", MoveModels.Conv10PosDepELU(N=19, Nfeat=15)))

gtp = GTP(engine, fclient)

gtp.loop()

