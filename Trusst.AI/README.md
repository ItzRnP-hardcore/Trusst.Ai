# AI Truss Builder

An interactive 2D truss design tool with an A\* search-based automatic solver, real-time structural physics, and a live parameter panel — all in a single Python file.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Pygame](https://img.shields.io/badge/pygame-2.x-green) ![NumPy](https://img.shields.io/badge/NumPy-required-orange) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What it does

You place **anchor nodes** (fixed supports) and **load nodes** (where external forces are applied), then press **Space** to let the solver automatically find the shortest valid truss that connects them under real structural constraints.

Every member is colour-coded by its internal force state:

| Colour | Meaning |
|--------|---------|
| 🔵 **Blue** | Tension — member is being pulled apart (+ve force) |
| 🔴 **Red** | Compression — member is being pushed together (−ve force) |
| 🟣 **Purple** | Buckled — compression exceeds the Euler critical load |
| ⬜ **Grey** | Unloaded — manual member, not yet solved |

---

## Features

- **A\* optimal solver** — finds the shortest-total-length valid truss automatically
- **Real structural physics** — method of joints (linear statics), yield force check, Euler column buckling check
- **Live settings panel** — adjust load, material (E, σ_y), cross-section (OD, wall), safety factor, and load angle in real time; re-solve with Space
- **Manual member mode** — lock in members you want the solver to keep
- **Force table** — all members ranked by force magnitude with colour-coded bar chart
- **Force/length labels** — toggle what's shown on each member with F
- **Adjustable load direction** — slider for load angle (−180° to 180°), arrow updates live

---

## Requirements

```
Python 3.8+
pygame >= 2.0
numpy >= 1.20
```

Install dependencies:

```bash
pip install pygame numpy
```

---

## Running

```bash
python ai_truss_builder.py
```

---

## Controls

| Key / Mouse | Action |
|-------------|--------|
| `A` | Switch to **Anchor** mode |
| `L` | Switch to **Load** mode |
| `M` | Switch to **Manual member** mode |
| `S` | Toggle settings panel |
| `Space` | Run A\* solver |
| `F` | Cycle force/length labels (`both → force → length → none`) |
| `R` | Clear auto-generated members (keep manual ones) |
| `C` | Clear everything |
| `Q` / `Esc` | Quit |
| **Left-click** | Place anchor / load node, or select nodes in Manual mode |
| **Right-click** | Remove nearest node (and its members) |

---

## Typical workflow

1. Press **A**, click two points to place anchor supports (red triangles)
2. Press **L**, click a point between/above them to place a load (green circle)
3. Press **Space** — the solver finds the optimal truss
4. Press **S** to open the settings panel; try changing the load magnitude, material, or cross-section, then press **Space** again
5. Press **M** to manually add intermediate nodes and members to guide the topology; press **Space** to re-solve incorporating those fixed members
6. Press **F** to cycle through label modes and inspect forces

---

## Physics overview

### Sign convention

The solver uses the standard structural engineering convention:

- **Positive internal force = Tension** (member ends are pulled toward each other)
- **Negative internal force = Compression** (member ends are pushed apart)

### Method of joints

The solver assembles a global equilibrium matrix `A` (2 equations per node — sum of forces in x and y equals zero) and solves the linear system `Ax = b` for member forces and reaction forces using `numpy.linalg.lstsq`. The system is statically determinate when `2j = m + r` (joints, members, reaction unknowns).

### Failure checks

Two failure modes are checked:

1. **Yield** — `|F| > T_MAX` where `T_MAX = σ_y × A_section`. Applies to both tension and compression.
2. **Euler buckling** — for compression members only: `|F| > P_cr / SF` where `P_cr = π²EI / (KL)²`. Purple colour and a wavy-line overlay indicate a buckled member.

### Cross-section

Members are modelled as hollow circular tubes. From the outer radius `r_o` and wall thickness `t`:

```
A  = π(r_o² − r_i²)
I  = π/4 (r_o⁴ − r_i⁴)
```

### A\* search

The solver uses A\* with:

- **State** — the current set of members (a frozenset of node-pair tuples)
- **Cost `g`** — total member length so far (minimised)
- **Heuristic `h`** — sum of Dijkstra shortest-path distances from each load node to the nearest anchor (admissible lower bound on remaining steel needed)
- **Goal** — all load nodes connected to at least one anchor, the statics system is statically determinate, no members yield, no members buckle

The search is bounded by `MAX_STATES = 120 000` expansions to stay responsive.

---

## Project structure

```
ai_truss_builder.py   — entire application (single file)
README.md
```

---

## Configuration constants (top of file)

| Constant | Default | Description |
|----------|---------|-------------|
| `WIDTH / HEIGHT` | 1100 × 720 | Window dimensions |
| `MAX_STATES` | 120 000 | A\* expansion limit |
| `MAX_DEGREE` | 20 | Max members per intermediate node |
| `NODE_SNAP` | 22 px | Snap radius for node selection |

---

## License

MIT
