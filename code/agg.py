import os, json, pandas as pd, numpy as np
def rd(n): return [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", f"{n}.jsonl"))]
def ms(x): return f"{np.mean(x)*100:.1f}±{np.std(x)*100:.1f}"
