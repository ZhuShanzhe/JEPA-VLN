"""Export a reading lamp and a beverage station as controlled entrance cues."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_bays import clear, block, save


out=Path(sys.argv[sys.argv.index("--")+1]).resolve()
clear()
block("lamp base",(0,0,.045),(.31,.30,.09),"474444")
block("lamp stem",(0,0,.52),(.045,.045,.96),"737373")
block("shade",(0,0,.99),(.36,.29,.29),"c1ab81")
save(out,"lamp")
clear()
block("serving cabinet",(0,0,.33),(.48,.34,.66),"867b68")
block("coffee machine",(0,.01,.85),(.30,.21,.36),"555859")
block("dispenser",(.01,-.13,.92),(.18,.09,.07),"929591")
save(out,"coffee_station")
