import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[0] / "worldcupteams" / "AIGamePyLibrary"))
sys.path.insert(0, str(ROOT.parents[0] / "worldcupteams"))
from AIGamePyLibrary import *

InitializeSoccer(
    "NestTest", "Poland",
    Vector3(Float(8), Float(0), Float(3)),
    Vector3(Float(6), Float(0), Float(-6)),
    Vector3(Float(6), Float(0), Float(6)),
    Vector3(Float(-14), Float(0), Float(0)),
)

fnA = CreateFunction("DoubleFn")
doubled = AssignToFunction(MultiplyFloats(fnA.Param1, Float(2)), fnA)
SetFunctionReturn(fnA, doubled)

fnB = CreateFunction("AddOneFn")
plus_one = AssignToFunction(AddFloats(fnB.Param1, Float(1)), fnB)
SetFunctionReturn(fnB, plus_one)

# Test 1: chain of two calls at TOP LEVEL (not inside any body) - A's output feeds B's input
top_level_chain = CustomFunction("AddOneFn", CustomFunction("DoubleFn", Float(5)))  # expect 11

# Test 2: call A FROM WITHIN function C's own body
fnC = CreateFunction("CallsInsideFn")
inner_call = AssignToFunction(CustomFunction("DoubleFn", fnC.Param1), fnC)
SetFunctionReturn(fnC, inner_call)
nested_body_call = CustomFunction("CallsInsideFn", Float(7))  # expect 14 if body-nesting works, else null/0

DebugDrawDisc(Vector3(top_level_chain, Float(0), Float(0)), Float(0.1), Float(0.1), "Yellow")
DebugDrawDisc(Vector3(nested_body_call, Float(0), Float(0)), Float(0.1), Float(0.1), "Magenta")

zero = Vector3(Float(0), Float(0), Float(0))
SoccerController(1, zero, Bool(False), Bool(False))
SoccerController(2, zero, Bool(False), Bool(False))
SoccerController(3, zero, Bool(False), Bool(False))
SoccerController(4, zero, Bool(False), Bool(False))

SaveData(str(ROOT / "scratch_test" / "nested_test.txt"), "grid")
print("wrote nested_test.txt")
