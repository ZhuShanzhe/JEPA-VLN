"""Check action budgets, physical hard cases and initially unclear cues; score no learned selections."""
from collections import Counter
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
def audit(split,homes):
    counts=Counter()
    for house in homes:
        folder=HERE/("private_render_blind60" if split=="blind" else "private_render")/(split+"_"+house)
        with np.load(folder/"controls.npz") as z:
            data={k:z[k] for k in z.files}
        g=data["group"];a=data["action"];r=data["reach"].astype(bool);pix=data["pixels"]
        checks=data["first_blocked"]
        assert np.all((a!=0).sum(1)+data["scan_cost"]<=data["budget"])
        assert np.array_equal(r,(checks==0)&(data["end_error_m"]<.14))
        assert np.all(data["visible"][r]==(pix[r]>=369)) and (data["visible"][~r]==-1).all()
        assert len(set(g))==len(set(data["source_index"]))
        counts.update(houses=1,groups=len(set(g)),rows=len(g),reachable_rows=int(r.sum()))
        for name in sorted(set(g)):
            rows=g==name;o=rows&(data["kind"]=="old");v=rows&~o
            assert o.sum()==7 and np.array_equal(data["plan_id"][o],np.arange(7))
            assert np.allclose(data["pose"][o,:8],data["pose"][o,:8][:1],atol=1e-5)
            assert not np.any(np.all(np.isclose(data["pose"][v,:8],data["pose"][o,:8][:1],atol=1e-5),1))
            for j in set(data["view_id"][v]):
                selected=v&(data["view_id"]==j)
                assert np.allclose(data["pose"][selected,:8],data["pose"][selected,:8][:1],atol=1e-5)
            unc=int(data["start_pixels"][o][0])<369
            both=unc and int(data["scan_pixels"][o][0])<369
            hard=not r[o].any()
            rescued=hard and np.any(r[v]&(pix[v]>=369))
            counts.update(initially_unclear=unc,both_views_unclear=both,all_original_blocked=hard,
                          original_blocked_alternative_reaches=hard and r[v].any(),
                          original_blocked_alternative_reaches_and_clear=rescued,
                          both_unclear_and_original_blocked=hard and both,
                          both_unclear_original_blocked_and_possible_rescue=rescued and both)
        if split!="blind":continue
        prefix=HERE/"private_render"/folder.name/"controls.npz"
        if not prefix.exists():continue
        with np.load(prefix) as first:
            used=data["source_index"]<len(np.unique(first["source_index"]))
            for k in ("action","start_pixels","scan_pixels","first_blocked","reach","pixels","kind","view_id","plan_id","budget"):
                assert np.array_equal(data[k][used],first[k]),k
            counts["prefix_sources_verified"]+=len(np.unique(first["source_index"]))
    return {k:int(v) for k,v in counts.items()}

if __name__=="__main__":
    manifest=json.loads((HERE/"view_design.json").read_text())
    result={s:audit(s,h) for s,h in manifest["cohorts"].items()}
    result["third_blind_scene_material"]={"house":"108736611_177263226","treatment":"Previous official-data derived single-asset PBR roughness texture transform compatibility overlay shared by both policies; images, collision mesh and layout unchanged; metallic/roughness appearance can change.","original_asset_sha256":"e3af95747a8a5c5ed0a89db082c92a37381f71bfd5447dce40d56870785aed3e","derived_asset_sha256":"e9b234cce4d2de4a6c97e2ce20ba9c0e32d196d23ff6d21310c73df155bd853a"}
    (HERE/"view_data_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
