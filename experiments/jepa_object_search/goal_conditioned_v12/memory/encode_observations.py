"""Encode new junctions and real return observations with the shared V11 RGB encoder."""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch


def run(project, splits):
    old, new = project/'jepa_hidden_v11', project/'jepa_route_v12'
    sys.path.insert(0, str(old/'models'))
    from encode import Encoder, encode_one
    enc = Encoder(project)
    torch.set_num_threads(2)
    for split in splits:
        import json
        experimental = split in ('fresh','extra_train','fresh2')
        if split in ('extra_train','fresh2'):
            replay = new/'expansion'
        elif split=='fresh':
            replay = new
        else:
            replay = new/'continuation'/split
        collection = 'train' if split=='extra_train' else 'test'
        raw = replay/'hssd'/collection if experimental else old/'hssd'/split
        if experimental:
            expected = len(json.loads((replay/'hssd/cohort.json').read_text())[collection])
        else:
            expected = len(list(raw.glob('*/receipt.json')))
        receipts = raw if split=='extra_train' else replay/'validation/returns'
        houses_done = set()
        while len(houses_done) < expected:
            for receipt in sorted(receipts.glob('*/receipt.json')):
                house = receipt.parent.name
                if house in houses_done:
                    continue
                for src in sorted((raw/house).glob('*.npz')):
                    if experimental:
                        key = hashlib.sha256(str(src.relative_to(replay)).encode()).hexdigest()
                        output = new/'models/cache/hssd'/split/'actor'/(key+'.npz')
                        if not output.is_file():
                            encode_one(enc, src, output, 'hssd', split)
                    if split=='extra_train':
                        continue
                    dst = new/'memory/cache_return'/split/house/src.name
                    if dst.exists():
                        continue
                    with np.load(src) as s, np.load(receipt.parent/src.name) as rt:
                        words = s['words'].tolist()
                        frames = rt['return_rgb']
                        n, taps = frames.shape[:2]
                        _, cue = enc.encode(frames.reshape(-1,*frames.shape[2:]), enc.words(words))
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    temp = dst.with_suffix('.pending.npz')
                    np.savez_compressed(temp, words=words, cue=cue.reshape(n,taps,len(words)))
                    temp.replace(dst)
                    if len(enc.cache)>450:
                        enc.cache.clear()
                houses_done.add(house)
            if len(houses_done)<expected:
                time.sleep(5)
        print(split, len(houses_done), flush=True)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('project',type=Path)
    args.add_argument('splits',nargs='+',
                      choices=['calibration','train','fresh','extra_train','fresh2'])
    ns=args.parse_args()
    run(ns.project, ns.splits)
