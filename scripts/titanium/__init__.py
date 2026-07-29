"""Titanium — the competition graph, split by concern instead of one file.

Each submodule is a self-contained namespace of graph nodes, mirroring how
the compiled Unity graph itself is organized conceptually:

    _env          shared foundation (AIGamePyLibrary, operator overloads)
    constants     gameplay tuning values (speeds, kick physics, charge)
    geometry      pure geometric primitives (tangents, forbidden cones)
    ball_physics  ball trajectory prediction (event legs, own-goal threat)
    meet_point    interception solve using PerfectController trajectory
    shot          shot legality + opponent interception circle optimization
    anti_tackle   held-ball rotation dodge (challenger build only)
    positioning   end-of-tick own-goal + teammate spacing (shared with AT)
    tackle        tackle duty + interact policy
    carrier       ball-carrier movement (shoot/pass/walk decisions)
    goalkeeper    GK cover/press/policy + multi-body threat cover
    instant_pass  emergency tackle-pass handoffs
    support_outlets attacking support stations on AT-safe headings
    debug_viz     all DebugDraw/TimePlot visualization
    graph         assembles the above into the actual node graph
    deploy        candidate/test/promote file I/O (not graph-building)
    paths         build/deploy file locations
    profile       per-tick evaluation cost profiling

Shot pipeline priority: Center -> Left Post -> Right Post -> opponent
tangent fallbacks (graze past interception circles using the same
tangent-circle construction as post tangents).

Every module's entry points take already-resolved values (positions,
opponent lists, ...) as parameters rather than reaching into global state —
so a module never needs to know how another module got its inputs, only
what shape they are.
"""
