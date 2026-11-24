from pathlib import Path
import sys
folder=Path(__file__).resolve()
sys.path[:0]=[str(folder.parents[1]),str(folder.parents[2]/"code")]
from collect_streams import sequence


def main(dst,objects,house,reverse=False):
    for cue in ("congruent","conflicting","hidden"):
        for left in range(2):
            for move in (False,True):
                folder=dst/"reverse"/"streams" if reverse else dst/"streams"
                folder.mkdir(exist_ok=True,parents=True)
                sequence(house,cue,left,move,objects,folder,True,dst/"assets",True,int(reverse))


if __name__=="__main__":main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),sys.argv[3],"--right-first" in sys.argv[4:])
