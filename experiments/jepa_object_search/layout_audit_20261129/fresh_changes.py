"""Check whether paired improvements were actually associated with different physical paths."""
from collections import Counter
import json
import numpy as np
from fresh_panel import DIR, HERE, read


def analyze():
    gain, loss = Counter(), Counter()
    for house in read(HERE/"fresh_design.json")["houses"]:
        old = read(DIR/"baseline"/("score_"+house+".json"))
        grid = read(DIR/"grid"/("score_"+house+".json"))
        for i,(before,after) in enumerate(zip(old,grid)):
            ob = before["old"]["outcome"]=="correct"
            ng = after["strict"]["outcome"]=="correct"
            if ob==ng:
                continue
            tally = gain if ng else loss
            key = str(i)+".npz"
            with np.load(DIR/"baseline/sensors"/house/key) as b, np.load(DIR/"grid/sensors"/house/key) as g:
                t0,t1=before["old"]["cost"], after["strict"]["cost"]
                old_act,grid_act=b["action"][:t0+1],g["action"][:t1+1]
                n=min(len(old_act),len(grid_act))
                changes=np.flatnonzero(old_act[:n]!=grid_act[:n])
                first=int(changes[0]) if len(changes) else n
                pose=g["odom"][:t1+1]
                dv=np.diff(pose[:,:2],axis=0)
                advanced=np.linalg.norm(dv,axis=1)>=.23
                positions=np.flatnonzero(advanced & (np.arange(len(dv))+1>=first))
                headings={int(round(pose[j,2]/(np.pi/6)))%12 for j in positions}
                tally["groups"]+=1
                tally["different_executed_prefix"]+=int(first<len(grid_act) or first<len(old_act))
                tally["grid_advanced_on_multiple_headings_after_divergence"]+=int(len(headings)>=2)
                tally["old_route_ever_clear"]+=int(before["ever_clear"])
                tally["grid_route_ever_clear"]+=int(after["ever_clear"])
    assert (gain["groups"],loss["groups"])==(4,1)
    result=dict(new_correct=gain, lost_correct=loss, comparison="Old refined baseline against grid and strict helper on the new 72. Count directions of actual translations after first different action. A gained report with previous visibility can reflect access to a more informative view.")
    (HERE/"fresh_changes.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)

if __name__=="__main__":
    analyze()
