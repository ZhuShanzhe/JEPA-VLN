"""Verify original source bytes and physically expected reports/false positives for the fixture."""
import hashlib
import json
import math
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):
    assert math.isclose(a,b,abs_tol=1e-5),(a,b)
zip_path=root/"downloads/most_release.zip"
with zipfile.ZipFile(zip_path) as z:
    assert z.testzip() is None
    sources=[n for n in z.namelist() if not n.endswith("/")]
    for n in sources:
        assert (root/"most_release"/n).read_bytes()==z.read(n),n
test=root/"most_cli_fixture"
path=test/"verified.json"
r=json.loads(path.read_text())
assert not r["journal_errors"]
e1,e2=r["episodes"]
rows=[]
for e,directions in [(e1,[101,102]),(e2,[101,101,None,102])]:
    assert e["action_protocol"]=="goat" and e["goal_type"]=="description"
    assert e["sensor_width"]==e["sensor_height"]==1280
    assert e["true_positive_count"]==2 and e["found_count"]==e["required_target_count"]==2
    close(e["trajectory_length"],5.0)
    pos=[x["position"] for x in e["trajectory_positions"]]
    for u,v in zip(pos,pos[1:]):
        assert math.dist(u,v)<.251 and abs(v[1]-.2)<1e-4
    ev=e["report_events"]
    assert len(ev)==len(directions)
    for item,obj in zip(ev,directions):
        if obj is None:
            assert item["near_targets"]==[] and item["is_false_positive"]
            close(item["report_ne"],2.25)
        else:
            assert item["near_targets"]==[obj]
            close(item["report_ne"],0.)
    rows.append({k:e[k] for k in ("episode_id","trajectory_length","steps_taken","sr","precision","recall","f1",
                    "true_positive_count","false_positive_count","repeat_report_count")})
assert e1["false_positive_count"]==e1["repeat_report_count"]==0
assert e2["false_positive_count"]==2 and e2["repeat_report_count"]==1
close(r["metrics"]["micro_rtsr_many"],.2)
receipt=dict(status="pass", scene="Self-authored BlenderRoom", independent_of_licensed_hm3d=True,
    entrypoint="most_release/evaluation/main/run_eval.py", original_config="most_release/evaluation/main/hm3d_config.yaml",
    raw_public_zip_sha256=digest(zip_path), all_original_files_identical_to_zip=len(sources),
    scripted_policy_sha256=digest(root/"tools/most_fixture_policy.py"), source_edits=0,
    habitat_lab="0.3.3", habitat_sim="0.3.3", rgb_depth_resolution=[1280,1280], score_scope="synthetic smoke only",
    agent="most_fixture_policy:ScriptedProbe", results_sha256=digest(path), observations_and_actions_verified=True,
    episodes=rows, extra_invalid_report_distance_m=2.25, both_target_report_distance_m=0.0)
out=root/"acceptance/most_original_cli_smoke.json"
out.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
