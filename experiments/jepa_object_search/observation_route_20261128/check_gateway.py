from gateway import ObservationGateway


def main():
    g = ObservationGateway()
    args = dict(upper_mode="explore", accepted_cue=False, anchors=[], blocked=False,
                budget=30, region={"key":"room"})
    def mode(**change):
        return g.admit(**(args | change))[0]
    assert mode()=="discover"
    assert mode(native_priority=True) is None
    assert mode(occupied=True) is None
    assert mode(blocked=True) is None
    assert mode(budget=0) is None
    assert mode(region=None) is None
    assert mode(upper_mode="inspect",accepted_cue=True,anchors=[(2.,3.)])=="verify"
    assert mode(upper_mode="inspect",accepted_cue=True) is None
    assert mode(upper_mode="inspect",accepted_cue=False) is None
    cont = dict(id="continue", actions=[1,1], new_surfaces=0,surface_keys=[])
    steer = dict(id="left", actions=[2,1], new_surfaces=1,surface_keys=[(2,3)])
    def gate(rows,budget=25):
        return g.discoverable(dict(candidates=rows,continue_id="continue"),(1.,2.,0.),1.,budget)
    assert not gate([cont])[0]
    assert not gate([cont,steer|dict(actions=[1])])[0]
    assert not gate([cont,steer|dict(new_surfaces=0,surface_keys=[])])[0]
    assert gate([cont,steer])==(True,"reachable_new_surface")
    assert gate([cont,steer],22)==(False,"recently_considered_surface")
    assert gate([cont,steer|dict(surface_keys=[(3,3)])],22)[0]
    print("PASS: 15 gateway cases incl. no-box entry, map evidence, duplicate, cue, native priority")

if __name__ == "__main__":
    main()
