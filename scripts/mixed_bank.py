#!/usr/bin/env python3
"""A mixed EXL3 expert bank for `"experts": "mixed"` (s0ma change 12): K3 in the layers named, K2 in the rest,
assembled from the two banks already on disk (no copies: the manifest points at their bank files).

  build OUT --k2 DIR --k3 DIR [--k3-layers 9,12,17,23-27,30-31,38]
      OUT/records-mixed.tsv  the K3 bank's records for the K3 layers and the K2 bank's for the others
      OUT/receipts.json      each layer's receipt from the bank its records come from (ds41d checks them at load)
      OUT/mixed-p80.placement  k2-p80 with each K3 layer's VRAM capacity (pins + dynamic budget) restated in K3 slots
      OUT/layers.json        {"k3_layers": [...], "k2": DIR, "k3": DIR, "expert_slots": N}

The default K3 layers are where ISTA-DASLab's RCO allocation (arXiv 2605.00649) gave this model's routed experts
more bits at 2.25 bpw. ds41d runs one store with a slot class per width, so a K2 expert never takes a K3-sized
slot; the placement states every VRAM slot so ds41d can size each class."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k2_bank  # noqa: E402

K2_REC, K3_REC = 8892416, 13316096
DEFAULT_K3 = '9,12,17,23-27,30-31,38'


def parse_layers(text):
    out = set()
    for part in filter(None, (x.strip() for x in str(text).split(','))):
        lo, _, hi = part.partition('-')
        lo, hi = int(lo), int(hi or lo)
        if not (0 <= lo <= hi <= 39):
            raise ValueError(f'layers are 0..39: {part}')
        out.update(range(lo, hi + 1))
    return sorted(out)


def records(path):
    """lines of a DS41MOE01 manifest by target layer (comments dropped)"""
    by = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        eid = int(line.split()[1])
        by.setdefault(eid // 384, []).append(line)
    return by


def placement(src, k3_layers):
    """k2-p80 with the K3 layers' VRAM capacity (pins + budget, in K2 slots) restated in K3 slots: the same bytes"""
    lines = Path(src).read_text().splitlines()
    pins = [0] * 43
    for x in lines:
        if x.startswith('vram '):
            e = int(x.split()[1])
            pins[e // 384 if e < 15360 else 40 + (e - 15360) // 128] += 1
    out, slots = [], 0
    for x in lines:
        if x.startswith('#'):
            continue
        if x.startswith('budget '):
            _, ly, b = x.split()
            ly, b = int(ly), int(b)
            if ly in k3_layers:
                cap = (pins[ly] + b) * K2_REC // K3_REC
                if cap < pins[ly]:
                    raise SystemExit(f'layer {ly}: its {pins[ly]} pinned experts do not fit its VRAM at K3 ({cap} slots)')
                b = cap - pins[ly]
            if ly < 40:
                slots += pins[ly] + b
            x = f'budget {ly} {b}'
        out.append(x)
    head = [f'# DS41PLACEMENT01 mixed-p80 (scripts/mixed_bank.py): k2-p80 with layers {fmt(k3_layers)} at K3: their',
            '# VRAM capacity (pins + dynamic budget) restated in 13.3 MB slots, the same bytes; the other layers as k2-p80.',
            f'# {slots:,} target slots in all.']
    return head + out, slots


def fmt(layers):
    layers, out, i = sorted(layers), [], 0
    while i < len(layers):
        j = i
        while j + 1 < len(layers) and layers[j + 1] == layers[j] + 1:
            j += 1
        out.append(str(layers[i]) if i == j else f'{layers[i]}-{layers[j]}')
        i = j + 1
    return ','.join(out)


def build(out, k2, k3, k3_layers, root=Path(__file__).resolve().parents[1]):
    out.mkdir(parents=True, exist_ok=True)
    m2, m3 = k2 / 'records-k2.tsv', k3 / 'records-k3.tsv'
    for m in (m2, m3):
        if not m.exists():
            raise SystemExit(f'no {m}: the mixed bank is assembled from the K2 and K3 banks')
    r2, r3 = records(m2), records(m3)
    rc2 = json.loads((k2 / 'receipts.json').read_text())
    rc3 = json.loads((k3 / 'receipts.json').read_text())
    lines = [f'# DS41MOE01 EXL3 mixed bank (scripts/mixed_bank.py): K3 layers {fmt(k3_layers)} from {k3}, '
             f'K2 elsewhere from {k2}']
    receipts = {}
    for layer in range(40):
        src, rc, rec = (r3, rc3, K3_REC) if layer in k3_layers else (r2, rc2, K2_REC)
        got = src.get(layer, [])
        if len(got) != 2 * 384:
            raise SystemExit(f'layer {layer}: the {"K3" if rec == K3_REC else "K2"} bank has {len(got) // 2} of its '
                             '384 records')
        if f' {rec} ' not in got[0]:
            raise SystemExit(f'layer {layer}: records of another size than {rec}')
        if str(layer) not in rc:
            raise SystemExit(f'layer {layer}: no receipt in its bank (verify that bank first)')
        lines += got
        receipts[str(layer)] = rc[str(layer)]
    (out / 'records-mixed.tsv').write_text('\n'.join(lines) + '\n')
    (out / 'receipts.json').write_text(json.dumps(receipts, indent=1) + '\n')
    src = root / 'profiles/k2-p80.placement'
    if not src.exists():
        k2_bank.placement(root)
    text, slots = placement(src, set(k3_layers))
    (out / 'mixed-p80.placement').write_text('\n'.join(text) + '\n')
    (out / 'layers.json').write_text(json.dumps({'k3_layers': k3_layers, 'k2': str(k2), 'k3': str(k3),
                                                 'expert_slots': slots}) + '\n')
    gib = (sum(1 for _ in k3_layers) * K3_REC + (40 - len(k3_layers)) * K2_REC) * 384 / 2**30
    print(f'mixed bank in {out}: K3 layers {fmt(k3_layers)} ({len(k3_layers)} of 40), {gib:.0f} GiB of experts, '
          f'{slots:,} VRAM slots ({(len(k3_layers) * 3 + (40 - len(k3_layers)) * 2) / 40:.2f} bits a weight on average)')


def slots_of(path):
    """the target VRAM slots a placement states (pins + dynamic budgets of layers 0..39)"""
    n = 0
    for x in Path(path).read_text().splitlines():
        if x.startswith('vram ') and int(x.split()[1]) < 15360:
            n += 1
        elif x.startswith('budget ') and int(x.split()[1]) < 40:
            n += int(x.split()[2])
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['build'])
    p.add_argument('out', type=Path)
    p.add_argument('--k2', type=Path, required=True)
    p.add_argument('--k3', type=Path, required=True)
    p.add_argument('--k3-layers', default=DEFAULT_K3)
    a = p.parse_args()
    build(a.out, a.k2.resolve(), a.k3.resolve(), parse_layers(a.k3_layers))


if __name__ == '__main__':
    main()
