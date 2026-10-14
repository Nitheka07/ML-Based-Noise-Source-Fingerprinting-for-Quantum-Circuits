import numpy as np, pandas as pd, os, json, time, warnings
from itertools import product
warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error, ReadoutError

SEVERITIES = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]
NOISE_TYPES = ["depolarizing", "amplitude_damping", "phase_damping", "readout_error"]
CIRC_ORDER = ["bell", "ghz", "w_state", "ghz4", "bell_pairs"]
SHOTS_LIST = [50, 100, 200, 400, 600, 720, 1024, 2048, 4096, 8192, 16384, 32768]
META = ["circuit", "noise_type", "severity", "trial"]
ENG = ["tvd", "entropy", "zero_bias", "unexpected_bitstring_rate", "symmetric_swap_rate", "parity_violation_rate"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder that contains code/, datasets/, results/
DATA = os.path.join(ROOT, "datasets"); RES = os.path.join(ROOT, "results"); FIGS = os.path.join(ROOT, "figs")
os.makedirs(DATA, exist_ok=True); os.makedirs(RES, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

def bell():
    qc = QuantumCircuit(2, 2); qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1]); return qc, {"00": .5, "11": .5}
def ghz():
    qc = QuantumCircuit(3, 3); qc.h(0); qc.cx(0, 1); qc.cx(1, 2); qc.measure([0, 1, 2], [0, 1, 2]); return qc, {"000": .5, "111": .5}
def w_state():
    qc = QuantumCircuit(3, 3); qc.ry(1.9106332362490184, 0); qc.ch(0, 1); qc.cx(1, 2); qc.cx(0, 1); qc.x(0)
    qc.measure([0, 1, 2], [0, 1, 2]); return qc, {"001": 1/3, "010": 1/3, "100": 1/3}
def ghz4():
    qc = QuantumCircuit(4, 4); qc.h(0); qc.cx(0, 1); qc.cx(1, 2); qc.cx(2, 3); qc.measure(range(4), range(4))
    return qc, {"0000": .5, "1111": .5}
def bell_pairs():
    qc = QuantumCircuit(4, 4); qc.h(0); qc.cx(0, 1); qc.h(2); qc.cx(2, 3); qc.measure(range(4), range(4))
    return qc, {"0000": .25, "0011": .25, "1100": .25, "1111": .25}
CIRCUITS = {"bell": bell(), "ghz": ghz(), "w_state": w_state(), "ghz4": ghz4(), "bell_pairs": bell_pairs()}

def noise_model(nt, p, nq):
    nm = NoiseModel()
    if nt == "depolarizing":
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["h", "x", "id", "ry"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx", "ch"])
    elif nt == "amplitude_damping":
        e = amplitude_damping_error(p)
        nm.add_all_qubit_quantum_error(e, ["h", "x", "id", "ry"]); nm.add_all_qubit_quantum_error(e.tensor(e), ["cx", "ch"])
    elif nt == "phase_damping":
        e = phase_damping_error(p)
        nm.add_all_qubit_quantum_error(e, ["h", "x", "id", "ry"]); nm.add_all_qubit_quantum_error(e.tensor(e), ["cx", "ch"])
    else:
        ro = ReadoutError([[1 - p, p], [p, 1 - p]])
        for q in range(nq): nm.add_readout_error(ro, [q])
    return nm

def probs(counts, nq):
    tot = sum(counts.values())
    return {"".join(b): counts.get("".join(b), 0) / tot for b in product("01", repeat=nq)}

def features(counts, ideal, nq):
    pr = probs(counts, nq); full = {s: ideal.get(s, 0.0) for s in pr}
    tvd = 0.5 * sum(abs(pr[s] - full[s]) for s in pr)
    ent = -sum(p * np.log2(p) for p in pr.values() if p > 0)
    z = lambda d: sum(p * (s.count("0") / nq) for s, p in d.items())
    support = set(ideal); unexp = sum(p for s, p in pr.items() if s not in support)
    comp = lambda s: "".join("1" if c == "0" else "0" for c in s)
    seen, bal = set(), 0.0
    for s in pr:
        if s in support or s in seen: continue
        c = comp(s); seen.update([s, c])
        if c in support: continue
        bal += 2 * min(pr[s], pr.get(c, 0.0))
    par = lambda d: sum(p for s, p in d.items() if s.count("1") % 2 == 0)
    pe, ie = par(pr), par(full)
    return {"tvd": tvd, "entropy": ent, "zero_bias": z(pr) - z(full), "unexpected_bitstring_rate": unexp,
            "symmetric_swap_rate": bal / unexp if unexp > 1e-9 else 0.0,
            "parity_violation_rate": 0.5 * (abs(pe - ie) + abs((1 - pe) - (1 - ie)))}

def gen(shots, draw, trials=6):
    rows = []
    for ci, cname in enumerate(CIRC_ORDER):
        qc, ideal = CIRCUITS[cname]; nq = qc.num_qubits
        for ni, nt in enumerate(NOISE_TYPES):
            for si, sev in enumerate(SEVERITIES):
                sim = AerSimulator(noise_model=noise_model(nt, sev, nq)); tqc = transpile(qc, sim)
                for trial in range(trials):
                    seed = (draw * 1000003 + ci * 10007 + ni * 1009 + si * 101 + trial * 7 + shots) % (2**31 - 1)
                    counts = sim.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()
                    r = {"circuit": cname, "noise_type": nt, "severity": sev, "trial": trial}
                    r.update(features(counts, ideal, nq)); r.update({f"p_{s}": v for s, v in probs(counts, nq).items()})
                    rows.append(r)
    df = pd.DataFrame(rows)
    pc = sorted([c for c in df.columns if c.startswith("p_")], key=lambda c: (len(c), c))
    return df[META + ENG + pc].fillna(0.0)

def load(shots, draw):
    return pd.read_csv(f"{DATA}/s{shots}_d{draw}.csv")
def feat_cols(df, kind="all"):
    raw = [c for c in df.columns if c.startswith("p_")]
    return {"all": ENG + raw, "eng": ENG, "raw": raw}[kind]
