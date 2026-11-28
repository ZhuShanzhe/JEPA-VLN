"""Shared decision path for MOST exploration and independent Habitat approach testing."""
import math
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"waypoints"))
from observation_sites import project_patches, propose_sites, plan_to_site
from local_search import LocalTargetTracker


def geometric_plan(grid, pose, scale, site, remaining, chunk):
    # Decide geometric validity independently of the final episode's action budget.
    result = plan_to_site(grid, pose, scale, site, budget=42, max_leg_steps=chunk)
    return dict(result, actions=result["actions"][:remaining])


def fast_continue(tracker, grid, pose, scale, budget, blocked=False):
    """Replan once from the new measured map without spending four actions to repeat a scan."""
    if not tracker.target or not getattr(tracker,"may_skip_scan",False) or blocked or not budget:
        return None
    from map_views import visible_map_surfaces
    surfaces = visible_map_surfaces(grid,pose,scale)
    if len(surfaces-tracker.scan_surfaces) >= 3:
        return None
    metric_pose = (pose[0]*scale, pose[1]*scale, pose[2])
    route = geometric_plan(grid,pose,scale,tracker.target,budget,tracker.chunk)
    if route["status"] != "active" or not route["actions"]:
        return None
    decision = tracker.decide(metric_pose,None,[],lambda _:route)
    tracker.may_skip_scan = False
    decision.update(region_phase="retained_goal_geometry_check",region_entered=tracker.inside(metric_pose),
                    patches=0,anchors=0,sites=0,anchor_source=None,region_key=None)
    return decision


def start_worker(project, log_path, cpu=False):
    left, right = socket.socketpair()
    try:
        env = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(key, None)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        py = project.parent/"jepa_instance_bench_20261016/policy_env/bin/python"
        with Path(log_path).open("ab") as log:
            p = subprocess.Popen([str(py), "-Bu", str(HERE/"local_worker.py"), str(project), str(right.fileno())],
                                 pass_fds=(right.fileno(),), stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, env=env)
        return Connection(left.detach()), p
    finally:
        left.close()
        right.close()


def request(channel, **msg):
    channel.send(msg)
    if not channel.poll(240):
        raise TimeoutError("local navigation scorer exceeded response time")
    return channel.recv()


def pick(tracker, memory, grid, pose, scale, rows, views, goal, budget, channel,
         depth_m=None, hfov=90., blocked=False, replay_memory=False, audit_sites=None):
    prediction = request(channel, mode="observe", views=views, plans=[r["actions"] for r in rows], goal=goal)
    metric_pose = (pose[0]*scale, pose[1]*scale, pose[2])
    for a, p in zip((-math.pi/6, 0., math.pi/6), prediction["current"]):
        if replay_memory:
            memory.mark_view(metric_pose[:2], pose[2]+a, observed_prior=p)
        else:
            memory.update_context(metric_pose[:2], pose[2]+a, p)
    region = None
    if rows:
        if len(rows) != len(prediction["score"]) or len(rows) != len(prediction["evidence"]):
            raise ValueError("regional score count mismatch")
        for row, p, e in zip(rows, prediction["score"], prediction["evidence"]):
            row.update(prior=p, evidence=e)
        index, region_log = memory.choose(rows)
        region = rows[index]
    else:
        region_log = dict(phase="no_regional_candidates")
    patches = sorted((p for p in prediction["patches"] if p["quality"] > 0 and p["support"] > .01),
                     key=lambda p: -p["support"])
    entered = tracker.inside(metric_pose)
    anchors = sites = []
    if entered or tracker.target:
        anchors = project_patches(grid, pose, scale, patches, depth=depth_m, hfov=hfov)
        sites = propose_sites(grid, pose, scale, anchors, budget=min(budget, 24))
        if sites:
            values = request(channel, mode="local_sites", plans=[s["actions"] for s in sites])
            if len(sites) != len(values):
                raise ValueError("local score count mismatch")
            for site, scores in zip(sites, values):
                site["score"] = float(scores[site["patch_id"]])
    def plan(t):
        return geometric_plan(grid, pose, scale, t, budget, tracker.chunk)
    decision = tracker.decide(metric_pose, region, sites, plan, region_entered=entered, blocked=blocked)
    from map_views import visible_map_surfaces
    tracker.scan_surfaces = visible_map_surfaces(grid,pose,scale)
    tracker.may_skip_scan = decision["action_source"] == "local"
    decision.update(region_phase=region_log["phase"], region_entered=entered, patches=len(patches),
                    anchors=len(anchors), sites=len(sites), anchor_source="rgbd_measured" if depth_m is not None
                    else "observed_map_column_proxy", region_key=region["key"] if region else None)
    if audit_sites is not None:
        audit_sites([r["position"] for r in rows], [s["site_xy"] for s in sites], region)
    return decision, region
