import numpy as np
import torch

from zetta_utils import log

log.add_supress_traceback_module(np)
log.add_supress_traceback_module(torch)
