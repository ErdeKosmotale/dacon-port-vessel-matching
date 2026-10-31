"""score = 0.7*a/532 + 0.3*b/3568 -> (a hangul rows, b non-hangul rows). usage: probe_decode.py <score> [<score> ...]"""
import sys
for s in map(float, sys.argv[1:]):
    sol = [(a, round((s - 0.7 * a / 532) * 3568 / 0.3)) for a in range(533)]
    sol = [(a, b) for a, b in sol if 0 <= b <= 3568 and abs(0.7 * a / 532 + 0.3 * b / 3568 - s) < 6e-11]
    print(s, '->', sol)
