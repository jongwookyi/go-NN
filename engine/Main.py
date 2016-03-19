#!/usr/bin/python

from GTP import GTP
from Engine import IdiotEngine

engine = IdiotEngine()

logfile = "log_%s.txt" % engine.name()
gtp = GTP(engine, logfile)

gtp.loop()

