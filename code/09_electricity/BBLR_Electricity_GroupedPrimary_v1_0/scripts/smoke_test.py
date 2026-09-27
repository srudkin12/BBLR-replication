#!/usr/bin/env python3
import subprocess,sys
subprocess.run([sys.executable,str(__import__('pathlib').Path(__file__).with_name('run_grouped_primary.py')),'--workers','2','--bootstrap','4'],check=True)
print('PASS_GROUPED_PRIMARY_SMOKE')
