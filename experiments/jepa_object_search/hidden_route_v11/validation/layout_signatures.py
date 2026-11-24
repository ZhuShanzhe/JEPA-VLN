"""Audit HSSD stage geometry independently of file names, textures or object labels."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def transform(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=float).reshape(4, 4).T
    x, y, z, w = node.get("rotation", (0, 0, 0, 1))
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    out = np.eye(4)
    out[:3, :3] = rotation @ np.diag(node.get("scale", [1., 1., 1.]))
    out[:3, 3] = node.get("translation", [0., 0., 0.])
    return out


def load_vertex_positions(path):
    raw = path.read_bytes()
    assert raw[:4] == b"glTF" and struct.unpack_from("<I", raw, 4)[0] == 2
    size, kind = struct.unpack_from("<II", raw, 12)
    assert kind == 0x4E4F534A
    spec = json.loads(raw[20:20 + size])
    offset = 20 + size + 8
    points = []
    def walk(idx, parent):
        node = spec["nodes"][idx]
        world = parent @ transform(node)
        if "mesh" in node:
            for primitive in spec["meshes"][node["mesh"]]["primitives"]:
                acc = spec["accessors"][primitive["attributes"]["POSITION"]]
                view = spec["bufferViews"][acc["bufferView"]]
                if acc["type"] != "VEC3" or acc["componentType"] != 5126 or "sparse" in acc:
                    raise ValueError("unsupported position layout")
                xyz = np.ndarray((acc["count"], 3), dtype="<f4", buffer=raw,
                                 offset=offset + view.get("byteOffset", 0) + acc.get("byteOffset", 0),
                                 strides=(view.get("byteStride", 12), 4))
                points.append(xyz @ world[:3, :3].T + world[:3, 3])
        for child in node.get("children", []):
            walk(child, world)
    for node in spec["scenes"][spec.get("scene", 0)]["nodes"]:
        walk(node, np.eye(4))
    return np.concatenate(points)


def stage_digest(points):
    # Compare right-angle yaw variants after removing translation. Mesh grouping,
    # triangle ordering and duplicated vertex attributes do not enter the digest.
    signatures = []
    x, y, z = points.T
    for horizontal in ((x, z), (z, -x), (-x, -z), (-z, x)):
        shape = np.column_stack((horizontal[0], y, horizontal[1]))
        points_cm = np.rint((shape-shape.min(0)) * 100.).astype("<i4")
        locations = np.unique(points_cm, axis=0)
        signatures.append(hashlib.sha256(locations.astype("<i4").tobytes()).hexdigest())
    return min(signatures)


def main(folder, out):
    fingerprints, by_shape = {}, {}
    for source in sorted(folder.glob("*.glb")):
        digest = stage_digest(load_vertex_positions(source))
        fingerprints[source.stem] = digest
        by_shape.setdefault(digest, []).append(source.stem)
    if not fingerprints:
        raise ValueError("no HSSD stage assets found")
    groups = [ids for ids in by_shape.values() if len(ids) > 1]
    out.write_text(json.dumps(dict(stages=len(fingerprints), distinct_signatures=len(by_shape),
                                   repeated_groups=groups, signatures=fingerprints), indent=2))
    print("stages", len(fingerprints), "distinct", len(by_shape), "repeated groups", len(groups))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage_folder", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    main(args.stage_folder, args.output)
