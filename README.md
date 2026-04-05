# AI Truss Builder

**Automated structural truss design using A\* search and 2-D finite element analysis — runs entirely in your terminal with a live interactive canvas.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![pygame](https://img.shields.io/badge/pygame-2.x-green) ![numpy](https://img.shields.io/badge/numpy-required-orange) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What It Does

You place anchor nodes (supports) and load nodes (where forces act) on a canvas. Press **Space** and the program automatically figures out which members (bars) to connect them with — giving you the lightest valid truss that won't yield or buckle under your loads.

Under the hood it's running an A\* search over the space of possible truss topologies, guided by a Steiner-MST lower-bound heuristic, and checking each candidate with a proper FEM solver (method of joints).

![demo screenshot placeholder](docs/screenshot.png)

---

## Quick Start

### 1. Install dependencies

```bash
pip install pygame numpy
```

### 2. Run

```bash
python truss_builder_v5.py
```

That's it — no config files, no build step, no server.

---

## Controls

| Key / Action | What it does |
|---|---|
| **A** | Switch to Anchor mode — click to place supports |
| **L** | Switch to Load mode — click to place force nodes |
| **N** | Switch to No-load mode — intermediate waypoint nodes |
| **M** | Manual member mode — click two nodes to add a bar |
| **Space** | Run the A\* solver |
| **S** | Toggle the Settings panel (materials, geometry, load) |
| **F** | Cycle member labels: force → length → sizing → none |
| **R** | Clear auto-generated members (keep manual ones) |
| **C** | Clear everything |
| **H / Home** | Reset zoom and pan to default |
| **+ / −** | Zoom in / out |
| **Mouse wheel** | Zoom centred on cursor (or adjust load magnitude in Load mode) |
| **Middle-drag** | Pan the canvas |
| **Right-drag** | Also pans — right-click on a node (no drag) removes it |
| **Q / Esc** | Quit |

### In Load mode with a node selected

| Key | Action |
|---|---|
| `[` / `]` | Magnitude ±10 N |
| `Shift+[` / `Shift+]` | Magnitude ±100 N |
| Scroll wheel | Magnitude ±10 N |
| `Ctrl+Scroll` | Angle ±5° |
| Arrow keys | Angle ±1° (Left/Right) or ±15° (Up/Down) |
| `Shift+Arrow` | ×10 on angle step |

---

## Features

- **Physics-aware search** — the solver uses the method of joints (static equilibrium) and checks both yield failure and Euler column buckling
- **Per-node load vectors** — each load node has its own magnitude and direction; edit them live with the on-canvas dial
- **No-load (waypoint) nodes** — guide the topology through specific points without applying force there
- **Manual members** — pin any bar in place before running the solver; A\* builds around them
- **Buckle retry** — if a found topology has members that buckle, the program automatically blacklists it and searches for the next best option (up to 8 attempts)
- **Live settings panel** — adjust Young's modulus, yield strength, tube geometry, safety factor, and pixels-per-metre without restarting
- **Minimum-section sizing** — after solving, F-cycle to "sizing" mode to see the minimum tube radius and wall thickness each member needs
- **Zoom and pan** — work on large layouts at any scale; zoom 15% – 600%
- **Colour-coded forces** — blue = tension, red = compression, purple = buckled, grey = unloaded

---

## Colour Key

| Colour | Meaning |
|---|---|
| 🔵 Blue | Tension (member is being pulled) |
| 🔴 Red | Compression (member is being pushed) |
| 🟣 Purple | Buckled (compression exceeds Euler P\_cr) |
| ⚫ Grey | Unloaded |
| ◇ White diamond | No-load intermediate node |

---

## How the Search Works (Short Version)

1. Start with any manually pinned members.
2. Maintain a priority queue of partial topologies, sorted by `f = g + h` where:
   - `g` = total member length so far
   - `h` = Steiner MST lower bound on remaining wire needed
3. Each expansion adds one new member.
4. When a topology connects all required nodes to an anchor, run the FEM solver.
5. If the solution passes yield and buckling checks, it's optimal.
6. If it buckles, blacklist it and keep searching.

For the full technical story — Zobrist hashing, Maxwell pre-filter, dominated-edge pruning, beam trimming — see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## Project Structure

```
ai-truss-builder/
├── truss_builder_v5.py   ← main file (run this)
├── truss_builder_v4.py   ← v4 without zoom/pan (reference)
├── truss_builder_v3.py   ← original version
├── README.md
├── DOCUMENTATION.md      ← deep-dive into every module
└── docs/
    └── screenshot.png
```

---

## Version History

| Version | Highlights |
|---|---|
| v5 | Zoom & pan viewport; smart right-click (pan vs. remove); zoom indicator |
| v4 | Zobrist hashing; Steiner MST heuristic; Maxwell pre-filter; beam trimming; buckle cache |
| v3 | Per-node load magnitudes; no-load (waypoint) nodes; on-canvas load editor |

---

## Dependencies

- Python 3.9+
- `pygame` >= 2.0
- `numpy` >= 1.22

Install both with: `pip install pygame numpy`

---

## License

MIT — use it, modify it, learn from it.
