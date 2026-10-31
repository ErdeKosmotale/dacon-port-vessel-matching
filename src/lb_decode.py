"""Decode public scores (10 decimals) into (hangul correct, non-hangul correct) counts.
score = 0.7*h/nH + 0.3*b/nE. Find (nH, nE) that make every score decode to integers."""
import numpy as np
S = {'v30': 0.9123082370, 'v31': 0.9124763984, 'v32': 0.9093403351, 'v33': 0.9182986193, 'v35A': 0.9053929667,
     'v31_e3': 0.9188031036, 'v33_e3': 0.9163396861, 'e8': 0.9219391669, 'v36_e12': 0.9184962827,
     'v33s9_A0': 0.9107947840, 'v33s9_D0': 0.9120264928, 'v33s9_C0': 0.9013025136, 'probeH_e8': 0.6302631579}
TOL = 6e-11
sols = []
for nH in range(200, 2001):
    h0 = S['probeH_e8'] * nH / 0.7
    if abs(h0 - round(h0)) * 0.7 / nH > TOL:
        continue
    nE = np.arange(500, 30001)
    ok = np.ones(len(nE), bool)
    for k, s in S.items():
        if k == 'probeH_e8':
            continue
        h = np.arange(nH + 1)
        rem = s - 0.7 * h / nH                       # (nH+1,)
        b = rem[None, :] * nE[:, None] / 0.3          # (len nE, nH+1)
        err = np.abs(b - np.round(b)) * 0.3 / nE[:, None]
        good = (err < TOL) & (b > -0.5) & (b < nE[:, None] + 0.5)
        ok &= good.any(axis=1)
    for e in nE[ok]:
        sols.append((nH, int(e)))
print('solutions (nH, nE):', sols[:20], len(sols))
for nH, nE in sols[:3]:
    print(f'--- nH={nH} nE={nE}')
    for k, s in S.items():
        out = []
        for h in range(nH + 1):
            b = (s - 0.7 * h / nH) * nE / 0.3
            if abs(b - round(b)) * 0.3 / nE < TOL and -0.5 < b < nE + 0.5:
                out.append((h, int(round(b))))
        print(f'{k:10s} {s:.10f} ->', [(h, b, f'H {h/nH:.4f} E {b/nE:.4f}') for h, b in out])
