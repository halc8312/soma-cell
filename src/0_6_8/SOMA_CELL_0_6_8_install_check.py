#!/usr/bin/env python3
# coding: utf-8
"""Fail-closed environment check for the RTX/CUDA workstation."""
from __future__ import division
import json,sys
import SOMA_CELL_0_6_8_gpu as gpu


def main():
    report=gpu.environment_report()
    report['ok_for_cpu_validation']=bool(report.get('torch_installed'))
    report['ok_for_cuda_execution']=bool(report.get('cuda_available'))
    report['recommended_validation_precision']='float64'
    report['recommended_throughput_precision']='float32 after fp64 lockstep gates pass'
    print(json.dumps(report,indent=2,sort_keys=True))
    if not report['ok_for_cpu_validation']:
        return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
