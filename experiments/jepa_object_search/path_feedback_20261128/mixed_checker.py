"""Scene-independent two-stage operating policy. Model inputs are exclusively proposed actions and observed depth."""
import numpy as np
import joblib
from checker import PathChecker,check_inputs


class MixedPathChecker:
    def __init__(self,weights):
        self.core=PathChecker(weights)
        self.limits=weights["operating_points"]

    @classmethod
    def load(cls,path):
        return cls(joblib.load(path))

    def predict(self,action,path):
        a,x=check_inputs(action,path)
        length=(a>0).sum(1)
        out={}
        for arm,m in (("near",length<=36),("long",length>36)):
            if not m.any():continue
            policy=self.limits[arm]
            v=self.core.predict(a[m],x[m],threshold=policy["advisory"])
            p=v["p_reach"]
            permit=policy["admit"]
            block=policy["block"]
            status=np.full(len(p),"uncertain",dtype="<U9")
            if block is not None:status[p<=block]="block"
            if permit is not None:status[p>=permit]="admit"
            v["screening"]=status
            for k,z in v.items():
                if k not in out:out[k]=np.empty(len(a),dtype=z.dtype)
                out[k][m]=z
        return out
