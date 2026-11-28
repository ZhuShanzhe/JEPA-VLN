"""Episode-local handoff guard; absence of a stable identity never authorizes a report."""
import math
from reobserve import Reobserve, STEP


class RecoveryBook:
    """Keep one per episode; stable_track must be the map track, not a frame-local box ID."""
    def __init__(self):
        self.launched=[]

    def open(self, anchor, category, stable_track=None):
        if anchor.get("anchor_source")!="rgbd_measured" or anchor.get("anchor_height_m") is None:
            return None
        xyz=(*anchor["anchor_xy"],anchor["anchor_height_m"])
        for old_goal,track,location,old in self.launched:
            if old_goal!=category:
                continue
            # Distinct persistent tracks can be near each other. Without persistent
            # tracks, suppress only a second recovery at essentially the same 3D point.
            if (stable_track==track if stable_track is not None and track is not None
                    else math.dist(xyz,location)<STEP/2):
                return None
        control=Reobserve(anchor,category)
        self.launched.append((category,stable_track,xyz,control))
        return control
