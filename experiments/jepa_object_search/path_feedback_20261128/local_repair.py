"""Actor-facing rejection, edit and re-prediction with the frozen cue and requested endpoint."""
import numpy as np
from path_geometry import from_depth,nominal,path_features
from replan import ProposalSet,embedding_sketch,decide


def alternatives(original,risk_step=-1):
    """Replace a local forward run by equal-endpoint parallel offsets on its left and right."""
    p=tuple(map(int,original))
    moves=[i for i,a in enumerate(p) if a==1]
    if not p or any(a not in (1,2,3) for a in p):raise ValueError("invalid physical action")
    if not moves:return [p]
    i=min(moves,key=lambda t:abs(t-risk_step)) if risk_step>=0 else moves[0]
    left=right=i
    while left>0 and p[left-1]==1:left-=1
    while right+1<len(p) and p[right+1]==1:right+=1
    length=right-left+1
    result=[p]
    for wait,width in ((0,1),(0,2),(1,2)):
        if wait>=length:continue
        for turn,opposite in ((2,3),(3,2)):
            f=(1,)
            outward=(turn,)*3
            inward=(opposite,)*3
            repaired=(p[:left]+f*wait+outward+f*width+inward+f*(length-wait)+
                      inward+f*width+outward+p[right+1:])
            assert np.allclose(nominal(repaired)[-1,:4],nominal(p)[-1,:4],atol=1e-5)
            result.append(repaired)
    return result


def consult(original,depth,context,camera_features,checker,jepa,scorer,budget,max_checks=3,previously_rejected=()):
    """camera_features(plan) supplies 20-D input using the same source box and source depth.

    No simulator is needed for inference. Input source embedding concatenates full image and cue.
    A caller must execute or acquire a fresh observation before invoking another replan cycle.
    """
    if max_checks<1:raise ValueError("a check is required")
    context=np.asarray(context,np.float32).reshape(2048)
    if not 1<=len(original)<=min(108,budget):raise ValueError("original plan exceeds supported budget")
    geo=from_depth(depth)
    if 1 in original and not (np.isfinite(geo[0]) & (geo[0]>.05)).any():
        return dict(status="request_observation",index=None,actions=[],history=[],checks=[],reason="missing_depth")
    def stack(plans):
        length=max(map(len,plans))
        a=np.zeros((len(plans),length),dtype=np.int64)
        x=np.zeros((len(plans),length,17),dtype=np.float32)
        for i,p in enumerate(plans):
            a[i,:len(p)]=p
            x[i,:len(p)]=path_features(p,geo)
        return a,x
    a,x=stack([original])
    first=checker.predict(a,x)
    def evidence(check,i):
        step=int(check["suspect_step"][i])
        return dict(decision=str(check["decision"][i]),basis=str(check["feedback"][i]),
                    p_reach=float(check["p_reach"][i]),threshold=float(check["threshold"][i]),
                    suspect_action_index=step if step>=0 else None,
                    occupied_side={-1:"left",0:"center_or_unspecified",1:"right"}[int(check["suspect_side"][i])] if step>=0 else None)
    if first["decision"][0]=="accept":
        return dict(status="execute",index=0,actions=list(original),history=[0],reason="initial_cleared",checks=[evidence(first,0)])
    excluded={tuple(p) for p in previously_rejected}
    options=[p for p in alternatives(original,int(first["suspect_step"][0])) if len(p)<=min(108,budget)
             and (p==tuple(original) or p not in excluded)]
    if len(options)==1:
        return dict(status="request_observation",index=None,actions=[],history=[0],
                    checks=[evidence(first,0)],reason="no_new_budgeted_alternative")
    a,x=stack(options)
    check=checker.predict(a,x)
    pose=np.stack([camera_features(p) for p in options])
    if not (np.allclose(pose[:,2],1.31) and np.allclose(pose[:,6],0) and np.allclose(pose[:,7],1)):
        raise ValueError("the path repair head was validated at the fixed native camera")
    ctx=np.repeat(context[None,:],len(options),axis=0)
    out=jepa(ctx,a,pose)
    s=ProposalSet(a,x,check["p_reach"],check["threshold"],check["suspect_step"],check["suspect_side"],
                  check["decision"],embedding_sketch(out["predicted_future"],ctx[:,:1024]),out["p_joint"],
                  out["p_visible_if_reach"])
    proposal=decide(s,0,budget,"trained",scorer,max_checks=max_checks)
    proposal["reason"]=str(first["feedback"][0])
    proposal["checks"]=[evidence(check,i) for i in proposal["history"]]
    return proposal


def select_or_repair(candidates,depth,context,camera_features,checker,jepa,scorer,budget,*,max_checks=3,selection="shortest"):
    """Choose among existing equal-endpoint proposals first; generate fresh equal-endpoint edits on rejection."""
    if max_checks<1:raise ValueError("a check is required")
    if selection not in ("shortest", "visible"):raise ValueError("unknown selection rule")
    offered=list(dict.fromkeys(tuple(map(int,p)) for p in candidates if 1<=len(p)<=min(budget,108)))
    if not offered:return dict(status="request_observation",actions=[],reason="no_budgeted_proposal")
    ends=np.array([nominal(p)[-1,:4] for p in offered])
    if not np.allclose(ends,ends[:1],atol=.001):raise ValueError("existing candidates must share a local endpoint and yaw")
    length=max(map(len,offered))
    geo=from_depth(depth)
    if not (np.isfinite(geo[0]) & (geo[0]>.05)).any():
        turns=[i for i,p in enumerate(offered) if 1 not in p]
        if not turns:return dict(status="request_observation",actions=[],reason="missing_depth")
        i=min(turns,key=lambda k:len(offered[k]))
        return dict(status="execute",actions=list(offered[i]),offered_index=i,reason="rotation_only")
    a=np.zeros((len(offered),length),np.int64)
    x=np.zeros((len(offered),length,17),np.float32)
    for i,p in enumerate(offered):
        a[i,:len(p)]=p
        x[i,:len(p)]=path_features(p,geo)
    check=checker.predict(a,x)
    allowed=np.flatnonzero(check["decision"]=="accept")
    if selection=="shortest" and len(allowed):
        i=int(min(allowed,key=lambda k:len(offered[k])))
        return dict(status="execute",actions=list(offered[i]),offered_index=i,reason="shortest_screened_existing",
                    p_reach=float(check["p_reach"][i]),threshold=float(check["threshold"][i]))
    if len(allowed):  # experimental visibility-weighted policy on equal-endpoint proposals
        ctx=np.repeat(np.asarray(context,np.float32).reshape(1,2048),len(offered),axis=0)
        pose=np.stack([camera_features(p) for p in offered])
        if not (np.allclose(pose[:,2],1.31) and np.allclose(pose[:,6],0) and np.allclose(pose[:,7],1)):
            raise ValueError("expected fixed native camera")
        out=jepa(ctx,a,pose)
        value=check["p_reach"]*out["p_visible_if_reach"]
        i=int(allowed[np.argmax(value[allowed])])
        return dict(status="execute",actions=list(offered[i]),offered_index=i,reason="best_screened_existing",
                    p_reach=float(check["p_reach"][i]),threshold=float(check["threshold"][i]),proposal_score=float(value[i]))
    # The initial shortest plan is the canonical seed. Preserve scene, instance and intended endpoint.
    i=min(range(len(offered)),key=lambda k:len(offered[k]))
    result=consult(offered[i],depth,context,camera_features,checker,jepa,scorer,budget,max_checks=max_checks,
                   previously_rejected=offered)
    result["offered_index"]=i
    result["trigger"]="all_existing_rejected"
    return result
