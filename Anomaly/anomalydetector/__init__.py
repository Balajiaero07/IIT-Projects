import logging
import os

# logging
logging.basicConfig(level=logging.INFO)

# root dirs
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT_DIR, '.out')
RES_DIR = os.path.join(ROOT_DIR, '.res')

# dataset dir
EVENTLOG_DIR = os.path.join(RES_DIR, 'eventlogs')
TMP_DIR = os.path.join(RES_DIR, '.temp')
MODELS_DIR = os.path.join(RES_DIR, 'models')

# evaluation dir
EVAL_DIR = os.path.join(OUT_DIR, 'evaluation')

# model out dir
MODEL_OUT_DIR = os.path.join(OUT_DIR, 'model')

# model plot out dir
PLOT_OUT_DIR = os.path.join(OUT_DIR, 'plots')

# generate dirs if non-existent
dirs = [
    ROOT_DIR,
    OUT_DIR,
    RES_DIR,
    EVAL_DIR,
    TMP_DIR,
    MODEL_OUT_DIR,
    MODELS_DIR,
    EVENTLOG_DIR,
    PLOT_OUT_DIR
]

for d in dirs:
    if not os.path.exists(d):
        os.mkdir(d)
