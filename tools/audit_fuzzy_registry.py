#!/usr/bin/env python3
"""
Audit registry fuzzy-matching for OCR-like variants.

Generates common OCR error variants for each registered plate and checks whether
the existing fuzzy-match logic would correct them back to the original plate.

Outputs a CSV report to `reports/fuzzy_audit_<timestamp>.csv` with columns:
original_plate,variant,matched_plate,distance,matched_is_original

Usage:
  python tools\audit_fuzzy_registry.py [--max-distance N]

This script intentionally copies a small Levenshtein implementation and uses
the same DB path as the Flask app to avoid importing the Flask module.
"""
import argparse
import csv
import os
import sqlite3
import time
import re

# DB path (same as in app.py)
DB_PATH = 'society_vehicles.db'


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]


def load_registered_plates():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute('SELECT plate_number FROM owners')
        rows = [r[0] for r in cur.fetchall() if r[0]]
        return rows
    finally:
        conn.close()


SUB_MAP = {
    '0': ['O'], 'O': ['0'],
    '1': ['I', 'L'], 'I': ['1', 'L'], 'L': ['1', 'I'],
    '2': ['Z'], 'Z': ['2'],
    '5': ['S'], 'S': ['5'],
    '8': ['B'], 'B': ['8'],
    '4': ['A'], 'A': ['4'],
    '6': ['G'], 'G': ['6'],
    '7': ['T'], 'T': ['7']
}


def normalize_plate(s: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


def generate_variants(plate: str, max_variants: int = 500):
    """Generate substitution and deletion variants for a plate.

    Returns a deduped set of variants (excluding the original plate).
    """
    plate = normalize_plate(plate)
    variants = set()

    # single-character substitutions
    for i, ch in enumerate(plate):
        if ch in SUB_MAP:
            for repl in SUB_MAP[ch]:
                v = plate[:i] + repl + plate[i+1:]
                if v != plate:
                    variants.add(v)

    # single-character deletions
    for i in range(len(plate)):
        v = plate[:i] + plate[i+1:]
        if v:
            variants.add(v)

    # small safeguard: limit number of variants
    variants = set(list(variants)[:max_variants])
    return variants


def find_best_match(candidate: str, registry: list):
    cand = normalize_plate(candidate)
    best = (None, 999)
    for reg in registry:
        regc = normalize_plate(reg)
        d = levenshtein(cand, regc)
        if d < best[1]:
            best = (regc, d)
    return best


def run_audit(max_distance: int = 3, out_dir: str = 'reports'):
    plates = load_registered_plates()
    if not plates:
        print('No registered plates found in DB. Exiting.')
        return 1

    registry_norm = [normalize_plate(p) for p in plates]
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'fuzzy_audit_{ts}.csv')

    total_variants = 0
    corrected_back = 0
    corrected_to_other = 0
    no_match = 0

    with open(out_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['original_plate', 'variant', 'matched_plate', 'distance', 'matched_is_original'])

        for plate in plates:
            orig = normalize_plate(plate)
            variants = generate_variants(orig)
            for var in variants:
                total_variants += 1
                matched, dist = find_best_match(var, registry_norm)
                if matched is None:
                    no_match += 1
                    writer.writerow([orig, var, '', '', 'no'])
                else:
                    if dist <= max_distance:
                        matched_is_original = 'yes' if matched == orig else 'no'
                        if matched == orig:
                            corrected_back += 1
                        else:
                            corrected_to_other += 1
                        writer.writerow([orig, var, matched, dist, matched_is_original])
                    else:
                        no_match += 1
                        writer.writerow([orig, var, '', dist, 'no'])

    print('Audit complete')
    print(f'Plates checked: {len(plates)}')
    print(f'Total variants generated: {total_variants}')
    print(f'Variants corrected back to original plate (<= {max_distance}): {corrected_back}')
    print(f'Variants corrected to a different plate (<= {max_distance}): {corrected_to_other}')
    print(f'Variants with no match within threshold: {no_match}')
    print(f'Report written to: {out_path}')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--max-distance', type=int, default=3, help='Max Levenshtein distance to consider a match')
    p.add_argument('--out-dir', type=str, default='reports', help='Directory to write CSV report')
    args = p.parse_args()

    rc = run_audit(max_distance=args.max_distance, out_dir=args.out_dir)
    raise SystemExit(rc)