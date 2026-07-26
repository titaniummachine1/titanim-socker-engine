"""Verify — in the simulator, not from docs — every property `titanium.nodefn`
relies on. Each check draws a disc whose X is the ACTUAL value; expected values
are in the printed key below.

  T1  one function, two top-level call sites, different args  -> 11 and 21
  T2  nested: function body calls another function            -> 8 if it works
  T3  Vector3 in / Vector3 out, two call sites                -> x=5 and x=9
  T4  four params                                             -> 1+2+4+8 = 15
  T5  body reads a GLOBAL constant node (not captured)        -> 7+100 = 107
  T6  Vector3 param split inside body (.x via Vector3Split)   -> 3
  T7  call result fed straight into another node's typed port -> 11*2 = 22
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WT = r"C:\gitProjects\worldcup\worldcupteams"
sys.path.insert(0, os.path.join(WT, "AIGamePyLibrary"))
sys.path.insert(0, WT)

from AIGamePyLibrary import *  # noqa: E402
import ball_trajectory_graph as _traj  # noqa: E402,F401  installs Node +/-/* overloads

DISCS = []


def show(label, value_float_node, color):
    DISCS.append((color, label))
    DebugDrawDisc(Vector3(value_float_node, Float(0), Float(0)), Float(0.1), Float(0.1), color)


def main():
    InitializeSoccer(
        "FnProbe", "Poland",
        Vector3(Float(8), Float(0), Float(3)),
        Vector3(Float(6), Float(0), Float(-6)),
        Vector3(Float(6), Float(0), Float(6)),
        Vector3(Float(-14), Float(0), Float(0)),
    )

    # A global constant created BEFORE any body, to check bodies can read
    # globals (this is what nodefn._CONSTANT_NODES relies on).
    global_hundred = Float(100)

    # --- AddOne: Float -> Float -------------------------------------------
    f_add = CreateFunction("PAddOne")
    AssignToFunction(AddFloats(f_add.Param1, Float(1)), f_add)
    SetFunctionReturn(f_add, AddFloats(f_add.Param1, Float(1)))

    # T1: two top-level call sites, different args
    show("T1a expect 11", CustomFunction("PAddOne", Float(10)), "Green")
    show("T1b expect 21", CustomFunction("PAddOne", Float(20)), "Green")

    # T7: call result straight into a typed Float port
    show("T7 expect 22", MultiplyFloats(CustomFunction("PAddOne", Float(10)), Float(2)), "White")

    # --- T2: nesting — a body that calls another function ------------------
    f_nest = CreateFunction("PNest")
    inner = AssignToFunction(CustomFunction("PAddOne", f_nest.Param1), f_nest)
    SetFunctionReturn(f_nest, inner)
    show("T2 expect 8 if nesting works", CustomFunction("PNest", Float(7)), "Red")

    # --- T3: Vector3 in / Vector3 out, two call sites ----------------------
    f_vec = CreateFunction("PScale2")
    scaled = AssignToFunction(ScaleVector3(f_vec.Param1, Float(2)), f_vec)
    SetFunctionReturn(f_vec, scaled)
    v_a = CustomFunction("PScale2", Vector3(Float(2.5), Float(0), Float(0)))
    v_b = CustomFunction("PScale2", Vector3(Float(4.5), Float(0), Float(0)))
    show("T3a expect 5", Vector3Split(v_a).x, "Cyan")
    show("T3b expect 9", Vector3Split(v_b).x, "Cyan")

    # --- T4: four parameters ----------------------------------------------
    f4 = CreateFunction("PSum4")
    s4 = AssignToFunction(
        AddFloats(AddFloats(f4.Param1, f4.Param2), AddFloats(f4.Param3, f4.Param4)), f4
    )
    SetFunctionReturn(f4, s4)
    show("T4 expect 15", CustomFunction("PSum4", Float(1), Float(2), Float(4), Float(8)), "Yellow")

    # --- T5: body reads a global (uncaptured) constant node ----------------
    f_glob = CreateFunction("PAddGlobal")
    g = AddFloats(f_glob.Param1, global_hundred)
    AssignToFunction(g, f_glob)  # note: global_hundred deliberately NOT assigned
    SetFunctionReturn(f_glob, g)
    show("T5 expect 107", CustomFunction("PAddGlobal", Float(7)), "Magenta")

    # --- T6: Vector3Split of a param inside the body -----------------------
    f_split = CreateFunction("PTakeX")
    parts = Vector3Split(f_split.Param1)
    AssignToFunction(parts.x, f_split)
    SetFunctionReturn(f_split, parts.x)
    show("T6 expect 3", CustomFunction("PTakeX", Vector3(Float(3), Float(0), Float(9))), "Orange")

    zero = Vector3(Float(0), Float(0), Float(0))
    for p in (1, 2, 3, 4):
        SoccerController(p, zero, Bool(False), Bool(False))

    out = os.path.join(HERE, "FnProbe.txt")
    SaveData(out, "grid")
    print(f"Wrote {out}")
    print("\nDISC KEY (in draw order):")
    for i, (color, label) in enumerate(DISCS):
        print(f"  {i}  {color:8s} {label}")


if __name__ == "__main__":
    main()
