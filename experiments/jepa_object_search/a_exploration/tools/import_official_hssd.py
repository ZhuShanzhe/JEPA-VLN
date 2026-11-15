"""Validate the author's v0.2.5 episode archive; publish it and the frozen split counts.
Usage: python import_official_hssd.py /path/to/objectnav_hssd-hab_v0.2.5.zip
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import tempfile
import zipfile

p=argparse.ArgumentParser()
p.add_argument('archive',type=pathlib.Path)
p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('/root/autodl-tmp/jepa_object_search/a_exploration/official/hssd_v025'))
args=p.parse_args()
source=args.archive.resolve(strict=True)
with zipfile.ZipFile(source) as z:
    assert sum(i.file_size for i in z.infolist())<1024**3,'unexpected expanded archive size'
    assert z.testzip() is None, 'bad archive CRC'
    selected={}
    rows=collections.defaultdict(list);scenes=collections.defaultdict(set);classes=collections.defaultdict(set)
    for entry in z.infolist():
        path=pathlib.PurePosixPath(entry.filename)
        assert '..' not in path.parts and not path.is_absolute()
        if path.suffixes[-2:]!=['.json','.gz'] or '__MACOSX' in path.parts or path.name.startswith('._'):continue
        names=[s for s in ('train','val') if s in path.parts]
        if not names:continue
        assert len(names)==1
        split=names[0];sub=pathlib.PurePosixPath(*path.parts[path.parts.index(split):]);assert str(sub) not in selected
        blob=z.read(entry);data=json.loads(gzip.decompress(blob))
        assert 'episodes' in data
        for ep in data['episodes']:
            pair=(ep['scene_id'],ep['episode_id'])
            rows[split].append(pair);scenes[split].add(pair[0]);classes[split].add(ep['object_category'])
        selected[str(sub)]=entry
    assert rows['val'] and rows['train']
    assert all(len(v)==len(set(v)) for v in rows.values())
    report={'source':'https://github.com/3dlg-hcvc/hssd#objectnav-episode-datasets',
            'archive_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'splits':{s:dict(episodes=len(rows[s]),scene_count=len(scenes[s]),categories=sorted(classes[s])) for s in rows}}
    print(json.dumps(report,indent=2))
    dst=args.output
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        assert json.loads((dst/'receipt.json').read_text())==report,'another release already exists'
    else:
        with tempfile.TemporaryDirectory(dir=dst.parent) as staging:
            out=pathlib.Path(staging)/'dataset'
            for name,ent in selected.items():
                f=out/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(z.read(ent))
            (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
            os.rename(out,dst)
