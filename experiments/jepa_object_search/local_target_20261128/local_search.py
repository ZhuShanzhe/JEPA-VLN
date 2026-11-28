"""Keep a camera goal in a metric map across navigation decisions."""
import math


def angle_error(a, b):
    return abs(math.remainder(a - b, math.tau))


class LocalTargetTracker:
    region_radius = 1.0
    position_tolerance = .18
    angle_tolerance = math.radians(15.01)
    chunk = 4

    def __init__(self):
        self.region = None
        self.target = None
        self.history = []
        self.serial = 0
        self.tick = 0

    def inside(self, pose):
        return self.region is not None and math.dist(pose[:2], self.region) <= self.region_radius

    def _remember(self, site, kind):
        self.history.append((site, kind, self.tick))

    def _eligible(self, site):
        for old, why, tick in reversed(self.history):
            if why != "arrived" and self.tick - tick > 4:
                continue
            if (math.dist(site["anchor_xy"], old["anchor_xy"]) < .55
                    and math.dist(site["site_xy"], old["site_xy"]) < .8
                    and angle_error(site["site_yaw"], old["site_yaw"]) < math.radians(55)):
                return False
        return True

    def _region_move(self, row, event, released):
        self.region = tuple(row["position"]) if row is not None else None
        return dict(action_source="region", actions=list(row["actions"]) if row else [], event=event,
                    released=released, region_xy=self.region, goal_id=None)

    def _acquire(self, site, pose):
        self.serial += 1
        self.target = dict(site, goal_id=self.serial)
        self.age = self.stalled = self.contacts = 0
        self.last_pose = tuple(pose)

    def _near(self, pose, site):
        return (math.dist(pose[:2], site["site_xy"]) <= self.position_tolerance
                and angle_error(pose[2], site["site_yaw"]) <= self.angle_tolerance)

    def arrival(self, pose, plan_to_site):
        """An actual camera observation is required. Queuing or finishing a chunk is insufficient."""
        t = self.target
        if t is None or not self._near(pose, t) or plan_to_site(t)["status"] != "arrived":
            return None
        self._remember(t, "arrived")
        self.target = self.region = None
        return dict(event="arrived", goal_id=t["goal_id"], age=self.age,
                    site_xy=t["site_xy"], site_yaw=t["site_yaw"], anchor_xy=t["anchor_xy"],
                    distance_m=math.dist(pose[:2], t["site_xy"]))

    def decide(self, pose, region, sites, plan_to_site, *, region_entered=None, blocked=False):
        self.tick += 1
        entered = self.inside(pose) if region_entered is None else region_entered
        released = self.arrival(pose, plan_to_site)
        if released:
            return self._region_move(region, "region_after_observation", released)
        available = sorted((s for s in sites if self._eligible(s) and not self._near(pose, s)),
                           key=lambda s: -s["score"])
        if not self.target and not entered:
            return self._region_move(region, "regional_approach", None)
        if self.target:
            t = self.target
            self.contacts = self.contacts + 1 if blocked else 0
            moved = math.dist(pose[:2], self.last_pose[:2]) >= .09
            turned = angle_error(pose[2], self.last_pose[2]) >= math.radians(20)
            self.stalled = 0 if moved or turned else self.stalled + 1
            self.last_pose = tuple(pose)
            route = plan_to_site(t)
            different = next((s for s in available if math.dist(s["anchor_xy"], t["anchor_xy"]) > .85
                              and s["score"] >= t["score"] + .18
                              and s.get("support", 0.) >= t.get("support", 0.) + .10), None)
            reason = None
            if not route.get("actions") or self.contacts >= 2 or self.stalled >= 2:
                reason = "no_reachable_route" if not route.get("actions") else "blocked_without_progress"
            elif self.age >= 2 and different is not None:
                reason = "stronger_new_support"
            if reason is None:
                return self._dispatch(pose, route, "retained", None)
            self._remember(t, reason)
            released = dict(goal_id=t["goal_id"], event=reason, geometry=route.get("reason"), age=self.age,
                            distance_m=math.dist(pose[:2], t["site_xy"]))
            self.target = None
            available = [s for s in available if self._eligible(s)]
            if reason == "stronger_new_support":
                available = [different]
        for site in available:
            route = plan_to_site(site)
            if route.get("actions"):
                self._acquire(site, pose)
                return self._dispatch(pose, route, "acquired", released)
        return self._region_move(region, "region_no_local_site", released)

    def _dispatch(self, pose, route, event, released):
        t = self.target
        self.age += 1
        return dict(action_source="local", actions=list(route["actions"][:self.chunk]),
                    event=event, released=released, goal_id=t["goal_id"], age=self.age,
                    site_xy=t["site_xy"], site_yaw=t["site_yaw"], anchor_xy=t["anchor_xy"],
                    surface=t.get("surface"), support=float(t.get("support", 0)),
                    score=float(t["score"]), distance_m=math.dist(pose[:2], t["site_xy"]))
