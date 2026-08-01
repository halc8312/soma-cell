# coding: utf-8
"""Generate SOMA-CELL 0.6-P1 long-run text/session reports."""
from __future__ import division
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (HERE, os.path.join(HERE, '..', '0_6_p0'), os.path.join(HERE, '..', 'baseline')):
    candidate = os.path.abspath(candidate)
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
import SOMA_CELL_0_6_P1_pythonista as p1

if __name__ == '__main__':
    print(p1.generate_report())
