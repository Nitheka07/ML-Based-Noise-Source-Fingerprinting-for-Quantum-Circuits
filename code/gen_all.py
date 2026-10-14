import sys, time, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
budget = float(sys.argv[1]); t0 = time.time(); n = 0
for draw in range(3):
    for s in SHOTS_LIST:
        f = f"{DATA}/s{s}_d{draw}.csv"
        if os.path.exists(f): continue
        if time.time() - t0 > budget: print("budget hit"); sys.exit(0)
        gen(s, draw).to_csv(f, index=False); n += 1
print("generated", n, "files;", len(os.listdir(DATA)), "total", f"{time.time()-t0:.0f}s")
