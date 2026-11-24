"""No-model invariants for cold start, evidence freshness, budgets and report reset."""
import math
from types import SimpleNamespace
import numpy as np

from evidence_policy import claim_rays, clue_deficit, score_routes


def check():
    ray = ((0, 0), 0)
    cal = {"train_first_appearance_frequency": .1, "evidence": [.05, .25]}
    memory = SimpleNamespace(time=6, cue_model=cal, cells={},
                             xy=np.zeros(2), yaw=0., cell=lambda p:tuple(np.rint(p/.75).astype(int)))
    assert clue_deficit(memory, set(), -1) == (1., 0)
    memory.cells[ray] = (6, None, .25)
    assert clue_deficit(memory, set(), -1) == (0., 1)
    memory.time = 30
    assert math.isclose(clue_deficit(memory, set(), -1)[0], 1-math.exp(-1))
    assert clue_deficit(memory, set(), 10) == (1., 0)
    assert claim_rays(memory, [(7, -1, False, None)], [7]) == {ray}
    assert clue_deficit(memory, {ray}, -1) == (1., 0)
    memory.cells[((0,0),2)] = (30, None, .25)
    assert clue_deficit(memory, {ray}, 7) == (0., 1)
    base, p, new = [.6,.55,.4], [.2,.2,.1], [0.,1.,.9]
    cold = score_routes(base,p,new,1,300)
    strong = score_routes(base,p,new,0,300)
    assert cold["index"] == 1 and strong["index"] == 0
    assert cold["no_exploration_index"] == 0 and strong["adaptive_weight"] == 0.
    assert score_routes(base,p,new,1,24)["adaptive_weight"] == cold["adaptive_weight"]/2
    assert max(score_routes(base,p,[0.,0.,0.],1,300)["exploration_bonus"]) == 0.
    assert score_routes([.5]*3, [.1,.8,.2], [0.]*3,1,100)["index"] == 1


if __name__ == "__main__":
    check()
    print("Passed offline evidence-gate invariants")
