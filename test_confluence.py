import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, ".")
from features.strategy_confluence import StrategyConfluence
sc = StrategyConfluence()

# Test conflicting signals
r = sc.analyze({
    "PA_Structure_Score": 5, "PA_BOS_Bull": 1,
    "SMC_Bear_OB": 1,
    "ICT_Judas_Bear": 1,
    "WY_Phase": 4,
    "EW_In_Wave5": 1, "EW_Wave_Type": 1,
})
print("--- Conflicting Signals ---")
print(f"Signal: {r['signal_label']}")
print(f"Confidence: {r['confidence']:.1%}")
print(f"Agreement: {r['agreement']}")
print(f"Bull: {r['bull_count']}, Bear: {r['bear_count']}, Hold: {r['hold_count']}")
for name, detail in r["breakdown"].items():
    sig = {-1:"SELL",0:"HOLD",1:"BUY"}[detail["signal"]]
    print(f"  {name}: {sig} ({detail['confidence']:.0%}) - {detail['reason']}")
