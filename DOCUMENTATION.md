# DOCUMENTATION.md — AI Truss Builder (v5)

This document is the technical reference for the AI Truss Builder. It explains every major module, data structure, and algorithm decision in plain language aimed at an undergraduate engineering or CS audience.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The Viewport (Zoom & Pan)](#2-the-viewport-zoom--pan)
3. [Physics Parameters — `Params`](#3-physics-parameters--params)
4. [The A\* Search Engine — `run_astar`](#4-the-a-search-engine--run_astar)
5. [State Representation — `TrussStateV4`](#5-state-representation--trusstatev4)
6. [The FEM Solver — `solve_truss`](#6-the-fem-solver--solve_truss)
7. [The Heuristic — `steiner_lower_bound`](#7-the-heuristic--steiner_lower_bound)
8. [Zobrist Incremental Hashing](#8-zobrist-incremental-hashing)
9. [Maxwell Determinacy Pre-filter](#9-maxwell-determinacy-pre-filter)
10. [Minimum Section Sizing](#10-minimum-section-sizing)
11. [The UI Layer](#11-the-ui-layer)
12. [Key Constants and Tuning Knobs](#12-key-constants-and-tuning-knobs)
13. [Known Limitations](#13-known-limitations)

---

## 1. Architecture Overview

The whole project is one Python file. Sections are marked with double-line comments. The call flow for a single solve is:

```
User presses Space
  └─ do_solve()
       └─ run_astar()          ← topology search (pure Python + heapq)
            └─ solve_truss()   ← physics check   (numpy lstsq)
                 └─ euler_pcr() ← buckling check (cached)
       └─ compute_member_sizing()  ← post-solve optional
  └─ draw everything via Viewport transforms
```

There is no async code, no threads. Everything happens in the main pygame event loop. The solver blocks the UI while running (a loading screen is shown).

---

## 2. The Viewport (Zoom & Pan)

### Why a Viewport?

Node coordinates are stored in **world space** — the same integer pixel grid as in v3/v4. The viewport is a purely visual transform that maps world coordinates to the screen you actually see.

### The Two Transforms

```
world → screen:   sx = wx * zoom + pan_x
                  sy = wy * zoom + pan_y

screen → world:   wx = (sx - pan_x) / zoom
                  wy = (sy - pan_y) / zoom
```

`w2s()` and `s2w()` implement these. `w2si()` is `w2s()` rounded to `int` for pygame draw calls.

### Zoom-at-Point

When you scroll the mouse wheel, the world point under the cursor should stay fixed. This is achieved by:

1. Convert the cursor screen position to world space (before the zoom changes).
2. Update `zoom`.
3. Recompute `pan_x` and `pan_y` so that the same world point maps back to the same screen position.

```python
wx, wy = s2w(cursor_sx, cursor_sy)   # world point under cursor
zoom   = new_zoom
pan_x  = cursor_sx - wx * zoom       # re-anchor
pan_y  = cursor_sy - wy * zoom
```

### Right-click: Pan vs. Remove

Right-click drag pans. Right-click without moving removes the nearest node. The flag `_rclick_moved` tracks whether the mouse moved more than 4 pixels between `MOUSEBUTTONDOWN` and `MOUSEBUTTONUP`. If it did, it's a pan. If not, it's a node removal.

### Snap Radius

Node snapping uses a world-space radius of `NODE_SNAP / zoom`. This keeps the snap distance feeling the same in screen pixels at any zoom level.

---

## 3. Physics Parameters — `Params`

All material and geometry values live in one `Params` object. The UI sliders write directly to its attributes.

| Attribute | Meaning | Default |
|---|---|---|
| `applied_load` | Default force magnitude for new load nodes (N) | 200 |
| `E` | Young's modulus (GPa) | 200 (steel) |
| `yield_mpa` | Yield strength (MPa) | 250 |
| `outer_r_mm` | Tube outer radius (mm) | 10 |
| `wall_mm` | Tube wall thickness (mm) | 2 |
| `K_factor` | Euler effective-length factor | 1.0 (pin-pin) |
| `safety_factor` | Applied to both yield and buckling limits | 1.5 |
| `pixels_per_m` | Scale factor for converting px to metres | 200 |

**Derived properties** (computed on the fly):

- `area` = π(r_o² − r_i²) — cross-section area
- `I` = π/4 (r_o⁴ − r_i⁴) — second moment of area
- `T_MAX` = σ_y × A — maximum allowable force (yield governs)
- `euler_pcr(L_px)` — Euler critical buckling load for length L (cached)

### The Buckling Cache

`euler_pcr` rounds its input to 0.5 px before computing, then stores the result in `_pcr_cache`. The search evaluates buckling for thousands of states; many states contain members of identical length. The cache converts most calls from an arithmetic operation to a dictionary lookup.

---

## 4. The A\* Search Engine — `run_astar`

### What Is Being Searched?

The search space is the set of all subsets of edges over the node graph. Each state is a partial truss — some edges present, some not yet decided. The goal is a subset that:

1. Connects every required node (load + no-load) to at least one anchor.
2. Passes the FEM solver (statically valid, within yield limits).
3. Has minimum total member length.

### Cost Function

- `g(state)` = sum of lengths of all members currently in the state (metres × pixels_per_m).
- `h(state)` = Steiner MST lower bound (see Section 7).
- `f(state)` = g + h.

### Expansion

At each step the algorithm pops the lowest-`f` state and tries adding one new member. For each node already in the frontier, it tries connecting it to every other node in `all_nodes_sorted` (pre-sorted by proximity to required nodes). Edges are skipped if:

- They are already in the state.
- Either endpoint exceeds `MAX_DEGREE` (and is not a special node).
- Both endpoints are already anchor-reachable and neither is a required node (**dominated-edge pruning**).
- `new_g >= best_g` (can't beat the current best).
- `new_g >= explored[(new_hash, new_n)]` (not an improvement over a previously seen state with the same hash).

### Beam Trimming

When the heap grows beyond `4 × BEAM_WIDTH`, it is trimmed to the `BEAM_WIDTH` smallest-`f` states. This makes the search incomplete (it might miss the true optimum in extreme cases) but keeps it fast and memory-bounded. In practice the beam is rarely triggered for fewer than 8 nodes.

### Blacklisting

If a connected topology passes the Maxwell check but has buckled members, it is added to a `blacklist` (a set of frozensets of edges). On the next attempt (up to 8), the same topology is skipped during feasibility checks.

---

## 5. State Representation — `TrussStateV4`

```python
class TrussStateV4:
    __slots__ = ("members_fs", "member_hash", "frontier_fs",
                 "g", "h", "f", "n_members")
```

Using `__slots__` saves memory and speeds attribute access.

| Field | Type | Meaning |
|---|---|---|
| `members_fs` | `frozenset` of edge tuples | The edges currently in this state |
| `member_hash` | `int` (64-bit) | Zobrist XOR hash of all edges |
| `frontier_fs` | `frozenset` of node tuples | Nodes reachable so far (can expand from) |
| `g` | `float` | Cost so far (total length) |
| `h` | `float` | Heuristic estimate |
| `f` | `float` | g + h |
| `n_members` | `int` | Len of `members_fs` (cheap duplicate detection) |

The **visited dict** key is `(member_hash, n_members)` — two integers, ~24 bytes. In v3 the key was the full frozenset, which was ~200 bytes per entry and took O(n) to compute and compare.

---

## 6. The FEM Solver — `solve_truss`

The solver uses the **method of joints** for 2-D pin-jointed trusses. It assembles a linear system `A·x = b` where:

- `x` contains the unknown member forces F₁…F_m and reaction components R₁…R_r.
- `A` is the equilibrium matrix: each row enforces ΣF_x = 0 or ΣF_y = 0 at one joint.
- `b` is the vector of applied loads.

For a member between joints i and j with unit vector (cx, cy):

```
Row 2i:   coefficient of F_k is  -cx  (contribution to joint i's x-equilibrium)
Row 2i+1: coefficient of F_k is  -cy
Row 2j:   coefficient of F_k is  +cx
Row 2j+1: coefficient of F_k is  +cy
```

Anchor reactions contribute 2 columns each (x and y reaction).

The system is solved with `numpy.linalg.lstsq`. The rank is checked — a rank-deficient matrix means the structure is a mechanism (not a valid truss).

**After solving:**

- If any |F_i| > T_MAX, the truss fails yield. Returned as `ok=False`.
- For any member in compression (F < 0), compute Euler P_cr. If |F| > P_cr / SF, add to `buckled_set`.

---

## 7. The Heuristic — `steiner_lower_bound`

A\* needs an **admissible heuristic** — one that never overestimates the true remaining cost. The heuristic used here is a Steiner tree lower bound.

**The idea:** how much extra wire do we need to connect all required nodes (those not yet reachable from an anchor)?

1. Build a graph of the current member set with real edge weights.
2. Run multi-source Dijkstra from all anchors through that graph.
3. For each required node not yet reachable, add the straight-line distance to the nearest anchor.

Straight-line distance is always ≤ the actual path length needed, so the heuristic is admissible. It is also much tighter than the v3 BFS heuristic (which used the Euclidean distance from the load node to the nearest anchor, ignoring the existing partial graph entirely).

---

## 8. Zobrist Incremental Hashing

In classic A\* the visited set maps state → best cost. For truss states, the "state" is a set of edges. Comparing two such sets naively is O(n). Building a `frozenset` is also O(n).

**Zobrist hashing** assigns each possible edge a random 64-bit integer token. The hash of a set of edges is the XOR of all their tokens. XOR has the nice property that:

```
hash(S ∪ {e}) = hash(S) XOR token(e)   ← O(1) incremental update
hash(S \ {e}) = hash(S) XOR token(e)   ← same formula
```

Because XOR is its own inverse, adding and removing an edge both flip exactly one bit in the hash. This makes state identity an O(1) operation regardless of how many members are in the state.

Collisions are extremely rare (probability 2⁻⁶⁴ per pair of states) and are further resolved by the `n_members` field — two states with the same hash but different member counts are definitely different.

---

## 9. Maxwell Determinacy Pre-filter

For a 2-D pin-jointed truss to be statically determinate:

```
m + r = 2j
```

where m = number of members, r = number of reaction components (2 per anchor), j = number of joints.

- `m + r < 2j`: under-constrained — it's a mechanism; the solver will find zero forces or fail.
- `m + r = 2j`: statically determinate — unique solution exists.
- `m + r > 2j`: over-constrained — statically indeterminate; the method of joints may still solve it approximately with lstsq.

The pre-filter rejects `under` states immediately, before touching numpy. In the dense-grid test case this eliminates roughly 60% of FEM calls.

---

## 10. Minimum Section Sizing

After a successful solve, `compute_member_sizing` calculates the **minimum tube cross-section** each member needs to carry its actual force.

For tension members: the minimum area is `A_min = |F| / σ_y`.

For compression members: two limits compete:
- Yield: same as tension, gives minimum area.
- Buckling: minimum second moment of area `I_min = |F| × SF × (K·L)² / π²E`.

The tube outer radius is back-calculated from `I_min` using the hollow-circle formula, accounting for the wall thickness ratio `k = t/r_o` (kept constant). The larger of the yield-governed and buckling-governed radii is used.

Results are displayed in "sizing" label mode (press F to cycle) and included in the force table.

---

## 11. The UI Layer

### Settings Panel

A slide-in panel on the left. Powered by `Slider` objects, each bound to a `Params` attribute. Changes to material/geometry/load take effect on the next Space solve. The panel animates open/closed with a simple linear interpolation: `x += (target − x) × 0.18` per frame at 60 Hz.

### Load Editor

When a load node is selected in Load mode, an on-canvas widget appears showing:

- A circular angle dial with tick marks (compass orientation).
- A magnitude readout with ±10 N and ±100 N buttons.

The dial is drawn in screen space adjacent to the node's screen position. Button hit-testing is in screen space (regular `pygame.Rect.collidepoint`).

### Drawing Helpers

All draw functions accept world-space node coordinates and route them through `vp.w2si()` before passing to pygame. Member line thickness scales with zoom (thicker when zoomed in, capped at 2× for extreme zoom). Node icon sizes also scale. Labels are suppressed below 50% zoom to avoid illegible clutter.

### Grid

The grid is drawn in screen space by computing which world-space grid lines are currently visible given the current pan and zoom. Lines too close together (< 6 px in screen space) are not drawn, which prevents a solid-fill appearance at very low zoom levels.

---

## 12. Key Constants and Tuning Knobs

| Constant | Default | Effect |
|---|---|---|
| `MAX_STATES` | 1,200,000 | Hard cap on search iterations before giving up |
| `MAX_DEGREE` | 4 | Maximum connections per non-special node |
| `BEAM_WIDTH` | 8,000 | Open-list cap before trimming |
| `MIN_ZOOM` / `MAX_ZOOM` | 0.15 / 6.0 | Viewport zoom limits |
| `ZOOM_STEP` | 1.15 | Multiplier per scroll tick or key press |
| `BUCKLE_PREVIEW_SEC` | 1.8 | How long the buckled animation plays before retry |
| `DEFAULT_LOAD_ANGLE` | 90° | Initial direction of new load nodes (downward) |

Increasing `BEAM_WIDTH` improves solution quality for large instances at the cost of memory and time. Increasing `MAX_DEGREE` allows denser topologies but expands the search space significantly.

---

## 13. Known Limitations

### 2-D only
All geometry, equilibrium, and rendering are 2-D. Real trusses are 3-D structures; extending this would require a 3-D viewport, 3-D coordinate input, and a 3-D equilibrium matrix (3 equations per joint instead of 2).

### Pin-joint assumption
The FEM solver assumes all joints are pinned (no moment transfer). Real welded trusses carry some bending. For most truss configurations this is a reasonable simplification, but it will underestimate stresses in rigid-joint structures.

### Uniform cross-section
All members are assumed to have the same tube profile (set in the Settings panel). The sizing display shows what each member *needs*, but the solver does not vary cross-sections per member in the optimisation.

### Beam search is incomplete
With `BEAM_WIDTH` active, the search may miss the globally optimal topology for very large instances. For 2–6 nodes and 1–3 anchors, the beam is almost never triggered and the result is optimal.

### No self-weight
Member weight is not included in the applied loads. For long spans this can be significant.

### Single load case
The solver optimises for one set of applied loads. Multiple load cases (e.g., wind + gravity simultaneously) require either superposition or a reformulation of the objective function.
