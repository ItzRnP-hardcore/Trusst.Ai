# AI Truss Builder — Technical Documentation

## 1. Overview

AI Truss Builder is a single-file interactive structural engineering simulation written in Python using `pygame` and `numpy`. It combines two distinct concerns: an automatic topology optimiser (based on A\* graph search) and a real-time structural physics solver (method of joints linear statics with Euler buckling checks). The user interacts with a graphical interface to place nodes, configure material and geometric parameters, and trigger the solver.

---

## 2. Architecture

The application is organised into five conceptual layers:

```
┌─────────────────────────────────────────────────┐
│                   MAIN LOOP                     │  pygame event handling, orchestration
├──────────────┬──────────────────────────────────┤
│  SETTINGS    │        DRAWING                   │  UI: sliders, node/member rendering,
│  PANEL       │        LAYER                     │  force table, load arrows, legend
├──────────────┴──────────────────────────────────┤
│               PHYSICS SOLVER                    │  solve_truss(): method of joints
├─────────────────────────────────────────────────┤
│               A* SEARCH ENGINE                  │  run_astar(): topology optimiser
├─────────────────────────────────────────────────┤
│               PARAMS OBJECT                     │  Params(): all material/geometry data
└─────────────────────────────────────────────────┘
```

All layers share a single `Params` instance (`p`) that is mutated in place by the settings panel sliders. No re-instantiation is needed — all derived quantities (area, second moment of area, yield capacity, maximum member length) are Python `@property` values computed on demand.

---

## 3. The `Params` Class

`Params` stores every physical parameter the solver needs. Derived quantities are computed lazily as properties so they are always consistent with the raw inputs.

### Raw parameters

| Attribute | Unit | Default | Meaning |
|-----------|------|---------|---------|
| `applied_load` | N | 200 | External force applied at each load node |
| `E` | GPa | 200 | Young's modulus (steel ≈ 200 GPa) |
| `yield_mpa` | MPa | 250 | Yield strength (mild steel ≈ 250 MPa) |
| `outer_r_mm` | mm | 10 | Outer radius of hollow circular section |
| `wall_mm` | mm | 2 | Wall thickness |
| `K_factor` | — | 1.0 | Effective length factor (1.0 = pin-pin) |
| `safety_factor` | — | 1.5 | Euler buckling safety factor |
| `pixels_per_m` | px/m | 200 | Drawing scale |
| `_load_angle` | ° | 90 | Direction of applied load (90° = downward) |

### Derived properties

```python
outer_r  = outer_r_mm × 10⁻³             # m
inner_r  = outer_r − wall_mm × 10⁻³      # m
area     = π(outer_r² − inner_r²)        # m²
I        = π/4 (outer_r⁴ − inner_r⁴)    # m⁴
E_pa     = E × 10⁹                       # Pa
T_MAX    = yield_mpa × 10⁶ × area        # N  (yield force capacity)
L_MAX_PX = f(E_pa, I, T_MAX, SF)         # px (max member length before buckling at T_MAX)
```

`L_MAX_PX` is derived by setting `P_cr = T_MAX / SF` and solving the Euler formula for `L`:

```
L = (π / K) × sqrt(E × I / P_cr)
```

This gives the A\* search engine its hard upper bound on member length — any candidate member longer than `L_MAX_PX` cannot carry `T_MAX` without buckling and is rejected outright.

---

## 4. Physics Solver (`solve_truss`)

### 4.1 Sign convention

The solver follows the standard structural engineering convention:

- **Positive internal force → Tension**: the member is being pulled apart at both ends. In the force table and on the canvas, tension is shown in **blue**.
- **Negative internal force → Compression**: the member is being pushed together. Shown in **red**.
- **Buckled**: a compression member whose `|F|` exceeds `P_cr / SF`. Shown in **purple** with a wavy-line overlay.

### 4.2 Method of joints

The method of joints expresses static equilibrium at every node. For a 2D truss with `j` joints, `m` members, and `r` reaction unknowns, equilibrium requires:

```
2j equations  (ΣFx = 0, ΣFy = 0 at each joint)
m + r unknowns (member forces + reactions)
```

The solver assembles a global matrix `A` of size `(2j) × (m + r)` and a load vector `b` of length `2j`, then solves:

```
A x = b
```

for the unknown vector `x` (member forces followed by reaction forces) using `numpy.linalg.lstsq`.

**Matrix assembly:** For each member connecting nodes `n1` and `n2`, the unit vector along the member is `(cx, cy) = (n2 − n1) / |n2 − n1|`. The column for that member in `A` is:

```
rows 2·i1,   2·i1+1  ← +cx, +cy   (force pulls n1 toward member — tension convention)
rows 2·i2,   2·i2+1  ← −cx, −cy   (force pulls n2 toward member)
```

This encoding means a positive solution value for a member corresponds to a member in tension — the convention is baked into the matrix.

**Reaction unknowns:** Each anchor contributes two independent reaction columns, one for x and one for y, each acting only at that anchor's rows.

**Rank check:** If the matrix is rank-deficient (`rank < min(shape)`) the truss is either a mechanism or overconstrained without being solvable — the solver returns failure.

**Residual check:** `‖Ax − b‖ > 10⁻³` flags a poor fit (e.g. inconsistent system) as failure.

### 4.3 Failure checks

After solving, two checks determine whether the truss is structurally acceptable:

**Yield check (both tension and compression):**
```
|F_i| > T_MAX   →   member fails in yielding
```
If any member fails this check, the truss is rejected by the A\* solver (it will not count it as a valid solution).

**Euler buckling check (compression only):**
```
P_cr(L) = π²EI / (KL)²
|F_i| > P_cr / SF   →   member is buckled
```
Buckled members are collected into a set and rendered in purple. The A\* solver only accepts trusses with zero buckled members.

---

## 5. A\* Search Engine (`run_astar`)

### 5.1 Problem formulation

Finding the minimum-material truss is a combinatorial search problem. The state space is the set of all subsets of possible member connections between the available nodes (anchors + loads + any intermediate nodes).

**State:** A `frozenset` of `(node_a, node_b)` tuples representing the current member set.

**Cost `g`:** Total Euclidean length of all members in the current state (in pixels). This is the quantity being minimised — less total length = less material = lighter structure.

**Heuristic `h`:** For each load node, find the shortest-path distance to the nearest anchor through the current member graph (using Dijkstra). Sum over all load nodes. If a load node is not yet connected, use the straight-line distance to the nearest anchor as a lower bound. This heuristic is admissible (never overestimates the true remaining cost) because the graph can only be extended by adding more length.

**Goal condition:** The truss is a valid goal when:
1. Every load node has a path to at least one anchor node
2. `solve_truss` returns success (no yielding, no buckling, system is statically determinate)

### 5.2 Search loop

```
heap ← [(f, counter, start_state)]
explored ← {}   # members_frozenset → best g seen

while heap:
    pop state with lowest f = g + h
    if g ≥ explored[state.members]: skip (already found better)
    if g ≥ best_g: skip (cannot beat current best)
    
    if connected and solve_truss succeeds:
        update best solution
        continue   # don't expand — goal reached
    
    for each frontier node fn:
        for each other node in all_nodes:
            if member (fn, other) would be too long: skip
            if degree limit exceeded: skip
            new_state ← state + (fn, other)
            if new_g < best_g and < explored[new_state]:
                push to heap
```

**Frontier nodes:** The set of nodes that the search can currently extend from. Initially this is the set of load nodes. As new members are added, the far endpoint of each new member joins the frontier.

**Degree limit (`MAX_DEGREE = 20`):** Intermediate nodes (not anchors or loads) are prevented from accumulating too many members. This prunes the search space and reflects practical fabrication constraints.

**State cap (`MAX_STATES = 120 000`):** If the heap counter exceeds this, the search terminates and returns the best solution found so far. This keeps the solver responsive.

### 5.3 Fixed members

Manual members (added by the user in MANUAL mode) are passed as `fixed_members` to `run_astar`. They are included in the initial state and are never removed. The search only adds members on top of them, ensuring the user's topology hints are preserved.

---

## 6. Visualisation

### 6.1 Colour encoding

Member colours encode physical state rather than arbitrary aesthetics. All colours are consistent between the canvas, the force table, and the legend:

| Colour | RGB | Condition |
|--------|-----|-----------|
| Blue (tension) | (40, 140, 255) | `force > 0` |
| Red (compression) | (220, 55, 55) | `force < 0` |
| Purple (buckled) | (210, 55, 215) | `|F| > P_cr / SF` |
| Grey (unloaded) | (90, 110, 160) | No force data |

**Intensity scaling:** Member line thickness and colour saturation both scale with `|F| / max(|F|)` across all members, making the most heavily loaded members visually prominent.

**Buckling gradient:** As a compression member's force approaches its Euler critical load, its colour blends toward purple (a `lerp_col` between the red base and `MBR_BUCKLE`) as an early visual warning before it officially buckles.

### 6.2 Load arrows

The load arrow is drawn with its **tail** offset away from the node and its **arrowhead pointing at (and into) the node**, showing the direction the external force acts on the structure. This matches the convention in free-body diagrams.

### 6.3 Force table

The side panel lists all members ranked by `|F|` descending. Each row shows:
- Member index (`#01`, `#02`, …)
- Type label: `T` (tension, blue), `C` (compression, red), `B!` (buckled, purple)
- Force magnitude in Newtons
- Member length in centimetres
- A proportional bar chart

The label colours are identical to the canvas member colours, so the table and the drawing can be read together without a separate colour mapping.

### 6.4 Settings panel

The settings panel is a slide-in sidebar driven by a simple lerp animation (`_anim_x` interpolates toward 0 when open, toward `−SETTINGS_W` when closed). Each `Slider` widget stores a reference to the `Params` attribute it controls by name (using `getattr`/`setattr`), making it trivial to add new parameters without modifying the event loop.

---

## 7. Key Design Decisions

**Why A\*?** Greedy approaches (always add the shortest member) get stuck in local minima. A\* with an admissible heuristic guarantees optimality within the explored state space, making it suitable for this combinatorial problem.

**Why `frozenset` for state?** Members have no inherent ordering and `(a, b)` should be the same member as `(b, a)`. Using a frozenset of tuples (where pairs are not normalised — both orientations can appear) means two states that differ only in member pair order are treated as distinct. In practice the A\* deduplication via `explored` handles this at the cost of some redundant work. Normalising pairs to `tuple(sorted([a, b]))` would reduce the state space further but adds a small overhead per expansion.

**Why `lstsq` instead of a direct solve?** The truss may be over- or under-constrained depending on node placement. `lstsq` handles both cases gracefully and the rank check immediately detects degenerate cases.

**Why is the settings panel drawn last?** pygame renders in painter's order (back to front). The settings panel slides over the canvas, so it must be drawn after all canvas content. The force table is drawn before the settings panel so that the panel can cover part of it if open.

---

## 8. Known Limitations

- **2D only.** The method of joints is implemented for planar trusses only.
- **Statically determinate only.** Hyperstatic (redundant) trusses will fail the rank check. The solver will not find solutions that require redundant members.
- **No intermediate node generation.** The A\* search connects only the nodes the user has explicitly placed. Adding intermediate nodes requires the user to place them manually before solving.
- **State cap.** For complex layouts with many nodes, the solver may hit `MAX_STATES` before finding the true optimum and return a suboptimal solution.
- **Pixel-coordinate arithmetic.** Nodes are stored as raw pixel tuples from mouse events. Because floating-point equality is used for node lookup, placing two nodes very close together can cause subtle deduplication failures. Snapping nodes to a grid would eliminate this.

---

## 9. Extending the Code

**Adding a new material parameter:** Add an attribute to `Params.__init__`, add a `@property` if a derived value is needed, add a `Slider` entry in `SettingsPanel.__init__`, and reference `p.new_attr` in the solver.

**Adding a new node type:** Add a colour and drawing branch to `draw_node`, a new mode string, and appropriate event handling in the main loop.

**Saving/loading:** All state is in `anchors` (list of tuples), `loads` (list of tuples), `members` (list of 2-tuples of tuples), and `manual_indices` (set of ints). These can be serialised to JSON with a small helper.

**3D extension:** Replace the 2D equilibrium matrix with a 3D one (3 equations per node, 3 direction cosines per member), update the drawing layer to use a 3D-to-2D projection, and replace pygame screen coordinates with 3D world coordinates.
