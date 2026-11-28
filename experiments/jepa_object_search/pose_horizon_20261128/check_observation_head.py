"""Regression cases for clearance, camera capability and returning a physically offered candidate."""
import numpy as np
from observation_head import HorizonCameraPredictor,propose


def exercise():
    x=np.zeros((4,20),np.float32)
    x[:,2]=[1.31,1.31,1.65,1.31]
    x[:,5]=x[:,7]=1.
    x[:,0]=[0,.5,0,2.5]
    plans=np.array([[2,3],[1,1],[2,3],[1,1]])
    clearance=np.ones(4,bool)
    source=np.ones(1024)
    crop=np.ones(1024)
    joint=np.array([.2,.65,.90,.99])
    def predict(context,actions,pose):
        assert context.shape==(4,2048)
        return dict(predicted_future=np.repeat(np.arange(4)[:,None],1024,1),p_reach=np.ones(4),
                    p_visible_if_reach=joint,p_joint=joint)
    def pick(**kw):
        return propose(predict,source,crop,plans,x,clearance,0,**kw)
    native=pick()
    assert native["index"]==1 and native["action"]==[1,1] and np.all(native["predicted_future"]==1)
    assert pick(articulated=True)["index"]==2
    clearance[1]=False
    assert pick()["index"]==0
    clearance[0]=False
    assert pick() is None
    clearance[:]=True
    joint[0]=.6
    assert pick()["index"]==0
    def short(ctx,a,p):
        return {"test":np.ones(len(a))}
    def long(ctx,a,p):
        return {"test":np.full(len(a),2.)}
    router=HorizonCameraPredictor(short,long)
    a=np.zeros((4,108),dtype=int)
    for i,n in enumerate([12,11,108,2]):a[i,:n]=1
    assert router(x,a,x)["test"].tolist()==[2.,1.,2.,1.]
    try:
        router(x,np.ones((4,109)),x)
    except ValueError:
        pass
    else:
        raise AssertionError("overlong plan accepted")
    print("Validated single-cue pose selection, fixed and articulated cameras, local range, clearance and embedding return")

if __name__=="__main__":
    exercise()
