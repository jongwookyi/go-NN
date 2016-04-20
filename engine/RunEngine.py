#!/usr/bin/python

import GTP
fclient = GTP.redirect_all_output("log_engine.txt")

from GTP import GTP
from TFEngine import TFEngine
from KGSEngine import KGSEngine
import MoveModels
from Book import PositionRecord
from Book import MoveRecord

engine = KGSEngine(TFEngine("conv12posdepELU", MoveModels.Conv12PosDepELU(N=19, Nfeat=21)))

gtp = GTP(engine, fclient)
gtp.loop()

