"""
AI Truss Builder — A* + Live Settings Panel  (v5)
==================================================
v5 Additions
------------
  ZOOM & PAN VIEWPORT
    The canvas now has a full 2-D viewport transform so you can zoom in to
    inspect joint details and pan to use a much larger working area.

    Controls
    ~~~~~~~~
      Mouse wheel          Zoom in / out (centred on cursor)
      Middle-click drag    Pan the canvas
      Right-click drag     Also pans (when NOT over a node)
      Home / H             Reset zoom & pan to default view
      +  /  -  (numpad)   Zoom in / out by fixed steps

    All node coordinates are stored in **world space** (the same pixel units
    as before, at zoom=1).  The viewport transform is purely visual.

    Technical notes
    ~~~~~~~~~~~~~~~
      world_to_screen(wx, wy) = (wx * zoom + pan_x,  wy * zoom + pan_y)
      screen_to_world(sx, sy) = ((sx - pan_x) / zoom, (sy - pan_y) / zoom)

    Node snapping, member hit-testing, load-arrow drawing, and all placement
    logic operate in world space.  Only the final pygame draw calls are in
    screen space.

v4 Optimizations (search algorithm overhaul — unchanged)
---------------------------------------------------------
  Zobrist hashing · Steiner MST heuristic · Maxwell pre-filter
  Dominated-edge pruning · Beam-width open-list cap
  Geometry-sorted neighbours · Cached Euler buckling

Controls (full list)
--------------------
  A    Anchor mode        L  Load mode        N  No-load node mode
  M    Manual member      S  Settings panel
  SPACE  Run A* solver    F  Cycle labels      R  Reset auto members
  C    Clear all          Q / Esc  Quit
  H / Home               Reset viewport
  +  /  -                Zoom in / out
  Wheel                  Zoom (centred on cursor)
  Middle-drag            Pan
  Right-click (no node)  Pan
  Right-click (on node)  Remove node  ← unchanged

Colour convention
-----------------
  BLUE   = Tension        RED    = Compression
  PURPLE = Buckled        GREY   = Unloaded
  WHITE diamond = No-load intermediate node
"""

import pygame
import numpy as np
import math, heapq, sys, time, random
from collections import deque

# ── Window ─────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT  = 1200, 740
PANEL_H        = 110
SETTINGS_W     = 310
NODE_SNAP      = 22          # world-space snap radius (scales with zoom for feel)

# ── Viewport defaults ──────────────────────────────────────────────────────────
DEFAULT_ZOOM   = 1.0
MIN_ZOOM       = 0.15
MAX_ZOOM       = 6.0
ZOOM_STEP      = 1.15        # multiplicative step per wheel tick / key press

# ── Palette ────────────────────────────────────────────────────────────────────
BG           = (10,  12,  20)
GRID_COL     = (20,  25,  40)
WHITE        = (228, 238, 255)
DIM          = ( 80, 100, 135)
ANCHOR_COL   = (255,  65,  65)
LOAD_COL     = ( 50, 215, 135)
NOLOAD_COL   = (160, 175, 220)

MBR_TENSION  = ( 40, 140, 255)
MBR_COMPRESS = (220,  55,  55)
MBR_BUCKLE   = (210,  55, 215)
MBR_UNLOADED = ( 90, 110, 160)
MBR_MANUAL   = ( 90, 110, 160)

PANEL_COL    = ( 13,  16,  28)
SETTINGS_BG  = ( 16,  20,  36)
SETTINGS_HDR = ( 22,  28,  50)
HIGHLIGHT    = (255, 210,   0)
TEXT_COL     = (148, 170, 208)
SEL_COL      = (255, 255,  70)
HOVER_COL    = (200, 200, 255)
SLIDER_TRACK = ( 35,  42,  68)
SLIDER_FILL  = ( 55, 150, 255)
SLIDER_KNOB  = (220, 230, 255)
DIVIDER      = ( 32,  40,  65)
TAG_BG       = ( 28,  36,  60)

DIAL_RING    = ( 55,  70, 110)
DIAL_NEEDLE  = ( 50, 215, 135)
DIAL_TEXT    = (220, 235, 255)

BTN_BG       = ( 28,  38,  65)
BTN_HOV      = ( 45,  60, 100)
BTN_ACT      = ( 55, 150, 255)

# ── Search / physics limits ────────────────────────────────────────────────────
MAX_STATES         = 1_200_000
MAX_DEGREE         = 4
BEAM_WIDTH         = 8_000
LABEL_CYCLE        = ["both", "force", "length", "sizing", "none"]
DEFAULT_LOAD_ANGLE = 90.0
BUCKLE_PREVIEW_SEC = 1.8
MIN_LOAD_N         = 1.0
MAX_LOAD_N         = 9999.0

# ── Zobrist hashing ────────────────────────────────────────────────────────────
_ZOB_SEED = 0xDEADBEEF_CAFEF00D
random.seed(_ZOB_SEED)

def _zobrist_for(edge):
    a, b = (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])
    h = hash((a, b)) & 0xFFFF_FFFF_FFFF_FFFF
    h ^= (h >> 30) * 0xBF58476D1CE4E5B9 & 0xFFFF_FFFF_FFFF_FFFF
    h ^= (h >> 27) * 0x94D049BB133111EB & 0xFFFF_FFFF_FFFF_FFFF
    h ^=  h >> 31
    return h & 0xFFFF_FFFF_FFFF_FFFF

_zob_cache: dict = {}

def zob(edge):
    key = (edge[0], edge[1]) if edge[0] <= edge[1] else (edge[1], edge[0])
    v = _zob_cache.get(key)
    if v is None:
        v = _zobrist_for(key)
        _zob_cache[key] = v
    return v


# ══════════════════════════════════════════════════════════════════════════════
# VIEWPORT  (world ↔ screen transform)
# ══════════════════════════════════════════════════════════════════════════════

class Viewport:
    """
    Manages pan + zoom.  All stored node data is in world space.
    Screen space is what pygame draws to.
    """
    def __init__(self):
        self.zoom  = DEFAULT_ZOOM
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._panning     = False   # middle or right-drag pan
        self._pan_last    = (0, 0)
        self._pan_origin  = (0, 0)

    # ── transforms ────────────────────────────────────────────────────────────
    def w2s(self, wx, wy):
        """World → screen (float)."""
        return wx * self.zoom + self.pan_x, wy * self.zoom + self.pan_y

    def s2w(self, sx, sy):
        """Screen → world (float)."""
        return (sx - self.pan_x) / self.zoom, (sy - self.pan_y) / self.zoom

    def w2si(self, wx, wy):
        """World → screen (int tuple, for pygame draw calls)."""
        x, y = self.w2s(wx, wy)
        return int(x), int(y)

    # ── zoom ──────────────────────────────────────────────────────────────────
    def zoom_at(self, screen_x, screen_y, factor):
        """Zoom by `factor`, keeping the point (screen_x, screen_y) stationary."""
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        # Adjust pan so the world point under the cursor stays fixed
        wx, wy   = self.s2w(screen_x, screen_y)
        self.zoom = new_zoom
        self.pan_x = screen_x - wx * self.zoom
        self.pan_y = screen_y - wy * self.zoom

    def zoom_step(self, direction, cx=None, cy=None):
        """direction: +1 zoom in, -1 zoom out. cx/cy default to canvas centre."""
        if cx is None: cx = (WIDTH - (SETTINGS_W if False else 0)) // 2
        if cy is None: cy = (HEIGHT - PANEL_H) // 2
        self.zoom_at(cx, cy, ZOOM_STEP ** direction)

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self):
        self.zoom  = DEFAULT_ZOOM
        self.pan_x = 0.0
        self.pan_y = 0.0

    # ── pan events ────────────────────────────────────────────────────────────
    def start_pan(self, sx, sy):
        self._panning    = True
        self._pan_last   = (sx, sy)

    def update_pan(self, sx, sy):
        if not self._panning:
            return
        dx = sx - self._pan_last[0]
        dy = sy - self._pan_last[1]
        self.pan_x += dx
        self.pan_y += dy
        self._pan_last = (sx, sy)

    def stop_pan(self):
        self._panning = False

    @property
    def is_panning(self):
        return self._panning

    # ── snap radius in world space (so it feels consistent at any zoom) ────────
    def snap_radius_world(self):
        return NODE_SNAP / self.zoom

    # ── canvas clip rect (excludes settings panel and status bar) ─────────────
    def canvas_rect(self, settings_open):
        x0 = SETTINGS_W if settings_open else 0
        return pygame.Rect(x0, 0, WIDTH - x0, HEIGHT - PANEL_H)


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS PARAMS
# ══════════════════════════════════════════════════════════════════════════════

class Params:
    def __init__(self):
        self.applied_load  = 200.0
        self.E             = 200.0
        self.yield_mpa     = 250.0
        self.outer_r_mm    = 10.0
        self.wall_mm       = 2.0
        self.K_factor      = 1.0
        self.safety_factor = 1.5
        self.pixels_per_m  = 200.0
        self._pcr_cache: dict = {}

    @property
    def outer_r(self):  return self.outer_r_mm * 1e-3
    @property
    def inner_r(self):  return max(0.0, self.outer_r - self.wall_mm * 1e-3)
    @property
    def t_r_ratio(self):
        if self.outer_r < 1e-12: return 0.2
        return min(0.5, self.wall_mm * 1e-3 / self.outer_r)
    @property
    def area(self):     return math.pi * (self.outer_r**2 - self.inner_r**2)
    @property
    def I(self):        return math.pi / 4 * (self.outer_r**4 - self.inner_r**4)
    @property
    def E_pa(self):     return self.E * 1e9
    @property
    def T_MAX(self):    return self.yield_mpa * 1e6 * self.area
    @property
    def L_MAX_PX(self):
        p_cr_min = self.T_MAX / self.safety_factor
        if p_cr_min <= 0: return float("inf")
        L_m = (math.pi / self.K_factor) * math.sqrt(self.E_pa * self.I / p_cr_min)
        return L_m * self.pixels_per_m

    def euler_pcr(self, length_px: float) -> float:
        key = round(length_px * 2) / 2
        v = self._pcr_cache.get(key)
        if v is None:
            Lm = key / self.pixels_per_m
            if Lm < 1e-9:
                v = float("inf")
            else:
                v = (math.pi**2 * self.E_pa * self.I) / (self.K_factor * Lm)**2
            self._pcr_cache[key] = v
        return v

    def invalidate_cache(self):
        self._pcr_cache.clear()

    def summary(self):
        return (f"E={self.E:.0f}GPa  sy={self.yield_mpa:.0f}MPa  "
                f"OD={self.outer_r_mm*2:.1f}mm  t={self.wall_mm:.1f}mm  "
                f"T_MAX={self.T_MAX:.1f}N")


# ══════════════════════════════════════════════════════════════════════════════
# PER-MEMBER MINIMUM SIZING
# ══════════════════════════════════════════════════════════════════════════════

def min_section_for_force(force_N, length_px, p):
    abs_f = abs(force_N)
    sy    = p.yield_mpa * 1e6
    E_pa  = p.E_pa
    K     = p.K_factor
    SF    = p.safety_factor
    k     = p.t_r_ratio
    Lm    = length_px / p.pixels_per_m

    A_min   = abs_f / sy if sy > 0 else 0.0
    r_yield = math.sqrt(A_min / (math.pi * (1 - (1-k)**2))) if A_min > 0 else 0.0

    r_buckle = 0.0
    if force_N < 0 and Lm > 1e-9:
        I_min    = abs_f * SF * (K * Lm)**2 / (math.pi**2 * E_pa)
        coeff    = math.pi / 4 * (1 - (1-k)**4)
        r_buckle = (I_min / coeff) ** 0.25 if coeff > 0 else 0.0

    r_o = max(r_yield, r_buckle, 1e-4)
    t   = k * r_o
    r_i = r_o - t
    A   = math.pi * (r_o**2 - r_i**2)
    vol = A * Lm * 1e6

    return {
        "r_o_mm":     r_o * 1e3,
        "t_mm":       t   * 1e3,
        "area_mm2":   A   * 1e6,
        "volume_cm3": vol,
        "dominated":  "buckling" if r_buckle >= r_yield else "yield",
    }

def compute_member_sizing(members, forces, p):
    sizing = {}
    for mem in members:
        f    = forces.get(mem, forces.get((mem[1], mem[0]), 0.0)) or 0.0
        L_px = math.dist(*mem)
        sizing[mem] = min_section_for_force(f, L_px, p)
    return sizing


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS PANEL
# ══════════════════════════════════════════════════════════════════════════════

class Slider:
    H = 6; KNOB_R = 8

    def __init__(self, label, unit, attr, lo, hi, fmt=".1f", log=False):
        self.label = label; self.unit = unit; self.attr = attr
        self.lo = lo; self.hi = hi; self.fmt = fmt; self.log = log
        self.rect = pygame.Rect(0,0,0,0); self.dragging = False

    def _to_t(self, v):
        if self.log:
            return (math.log(v)-math.log(self.lo)) / (math.log(self.hi)-math.log(self.lo))
        return (v-self.lo)/(self.hi-self.lo)

    def _from_t(self, t):
        t = max(0.0, min(1.0, t))
        if self.log:
            return math.exp(math.log(self.lo)+t*(math.log(self.hi)-math.log(self.lo)))
        return self.lo+t*(self.hi-self.lo)

    def draw(self, surf, font_sm, font_tiny, p, x, y, w):
        v = getattr(p, self.attr)
        t = self._to_t(v)
        surf.blit(font_tiny.render(self.label, True, TEXT_COL), (x, y))
        vs = font_tiny.render(format(v, self.fmt)+" "+self.unit, True, WHITE)
        surf.blit(vs, (x+w-vs.get_width(), y))
        ty = y+18
        pygame.draw.rect(surf, SLIDER_TRACK, pygame.Rect(x, ty, w, self.H), border_radius=3)
        fw = int(t*w)
        if fw > 0: pygame.draw.rect(surf, SLIDER_FILL, pygame.Rect(x, ty, fw, self.H), border_radius=3)
        kx = x+int(t*w); ky = ty+self.H//2
        pygame.draw.circle(surf, SLIDER_KNOB, (kx, ky), self.KNOB_R)
        pygame.draw.circle(surf, SLIDER_FILL,  (kx, ky), self.KNOB_R, 2)
        self.rect = pygame.Rect(x-self.KNOB_R, ty-self.KNOB_R,
                                w+self.KNOB_R*2, self.H+self.KNOB_R*2)
        return y+38

    def handle_event(self, event, p):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos): self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1: self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            rx = self.rect.x+self.KNOB_R; rw = self.rect.w-self.KNOB_R*2
            setattr(p, self.attr, self._from_t((event.pos[0]-rx)/rw))
            p.invalidate_cache()
            return True
        return False

    def is_active(self): return self.dragging


class SettingsPanel:
    PAD = 14
    def __init__(self):
        self.open = False; self._anim_x = -SETTINGS_W
        self.groups = [
            ("DEFAULT LOAD", [Slider("Default load (new nodes)","N","applied_load",10,2000,".0f")]),
            ("MATERIAL", [
                Slider("Young's modulus E","GPa","E",10,400,".0f"),
                Slider("Yield strength sy","MPa","yield_mpa",50,1000,".0f"),
            ]),
            ("CROSS-SECTION", [
                Slider("Outer radius","mm","outer_r_mm",2,50,".1f"),
                Slider("Wall thickness","mm","wall_mm",0.5,10,".1f"),
            ]),
            ("SOLVER", [
                Slider("Safety factor","x","safety_factor",1.0,4.0,".2f"),
                Slider("Pixels per metre","px/m","pixels_per_m",50,500,".0f"),
            ]),
        ]

    def toggle(self): self.open = not self.open

    def update_anim(self):
        target = 0 if self.open else -SETTINGS_W
        self._anim_x += (target-self._anim_x)*0.18
        if abs(self._anim_x-target) < 0.5: self._anim_x = target

    def draw(self, surf, fonts, p):
        self.update_anim()
        ox = int(self._anim_x)
        if ox <= -SETTINGS_W: return
        sw = SETTINGS_W; sh = HEIGHT-PANEL_H
        ps = pygame.Surface((sw,sh), pygame.SRCALPHA); ps.fill((*SETTINGS_BG,240))
        surf.blit(ps, (ox,0))
        pygame.draw.line(surf, DIVIDER, (ox+sw,0), (ox+sw,sh), 1)
        font_sm, font_md, font_tiny = fonts
        py = self.PAD
        hs = pygame.Surface((sw,36), pygame.SRCALPHA); hs.fill((*SETTINGS_HDR,255))
        surf.blit(hs, (ox,py-self.PAD))
        surf.blit(font_md.render("  PARAMETERS", True, HIGHLIGHT), (ox+self.PAD,py))
        py += 36
        surf.blit(font_tiny.render("Changes apply on next SPACE solve", True, DIM), (ox+self.PAD,py))
        py += 20
        iw = sw-self.PAD*2
        for gname, sliders in self.groups:
            pygame.draw.line(surf, DIVIDER, (ox,py+4), (ox+sw,py+4), 1)
            gh = font_tiny.render(gname, True, (100,120,160))
            pygame.draw.rect(surf, TAG_BG, pygame.Rect(ox+self.PAD-2,py-1,
                                                        gh.get_width()+8,gh.get_height()+4),
                             border_radius=3)
            surf.blit(gh, (ox+self.PAD+2,py)); py += 20
            for sl in sliders: py = sl.draw(surf, font_sm, font_tiny, p, ox+self.PAD, py, iw)
            py += 6
        pygame.draw.line(surf, DIVIDER, (ox,py), (ox+sw,py), 1); py += 8
        for line in [
            f"Area  = {p.area*1e6:.3f} mm²",
            f"I     = {p.I*1e12:.4f} mm⁴",
            f"T_MAX = {p.T_MAX:.1f} N",
            f"L_ref = {p.L_MAX_PX/p.pixels_per_m*100:.1f} cm",
        ]:
            surf.blit(font_tiny.render(line, True, (130,155,195)), (ox+self.PAD,py)); py += 15
        py += 4
        pygame.draw.line(surf, DIVIDER, (ox,py), (ox+sw,py), 1); py += 8
        surf.blit(font_tiny.render("Buckling checked post-solve", True, (180,120,60)), (ox+self.PAD,py)); py += 14
        surf.blit(font_tiny.render("L_max does NOT limit A* search", True, (180,120,60)), (ox+self.PAD,py)); py += 18
        pygame.draw.line(surf, DIVIDER, (ox,py), (ox+sw,py), 1); py += 8
        surf.blit(font_tiny.render("COLOUR KEY", True, (100,120,160)), (ox+self.PAD,py)); py += 14
        for col, lbl in [
            (MBR_TENSION,  "BLUE   = Tension (+ve)"),
            (MBR_COMPRESS, "RED    = Compression (-ve)"),
            (MBR_BUCKLE,   "PURPLE = Buckled"),
            (MBR_UNLOADED, "GREY   = Unloaded"),
            (NOLOAD_COL,   "DIAMOND= No-load node"),
        ]:
            pygame.draw.rect(surf, col, pygame.Rect(ox+self.PAD,py+2,10,10))
            surf.blit(font_tiny.render(lbl, True, (160,175,210)), (ox+self.PAD+14,py)); py += 14

    def handle_event(self, event, p):
        changed = False
        for _, sliders in self.groups:
            for sl in sliders:
                if sl.handle_event(event, p): changed = True
        return changed

    def any_active(self):
        for _, sliders in self.groups:
            for sl in sliders:
                if sl.is_active(): return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
# LOAD NODE EDITOR
# ══════════════════════════════════════════════════════════════════════════════

class LoadEditor:
    BTN_W  = 28
    BTN_H  = 20
    DIAL_R = 30

    def __init__(self):
        self._btn_minus     = pygame.Rect(0,0,0,0)
        self._btn_plus      = pygame.Rect(0,0,0,0)
        self._btn_minus_big = pygame.Rect(0,0,0,0)
        self._btn_plus_big  = pygame.Rect(0,0,0,0)
        self._hover_minus     = False
        self._hover_plus      = False
        self._hover_minus_big = False
        self._hover_plus_big  = False

    def draw(self, surf, node_screen_pos, angle_deg, magnitude, font_tiny, font_sm, mp):
        """node_screen_pos is already in screen space."""
        nx, ny = node_screen_pos
        R  = self.DIAL_R
        CX = nx + 70
        CY = ny

        if CX + R + 120 > WIDTH:           CX = nx - 70
        if CY - R - 50  < 0:              CY = R + 50
        if CY + R + 80  > HEIGHT-PANEL_H: CY = HEIGHT - PANEL_H - R - 80

        bg = pygame.Surface((R*2+6, R*2+6), pygame.SRCALPHA)
        pygame.draw.circle(bg, (12,18,40,220), (R+3,R+3), R+3)
        surf.blit(bg, (CX-R-3, CY-R-3))
        pygame.draw.circle(surf, DIAL_RING, (CX,CY), R, 2)

        for tick_deg in range(0, 360, 30):
            tr = math.radians(tick_deg); is_major = (tick_deg % 90 == 0)
            inner = R - (8 if is_major else 4)
            pygame.draw.line(surf,
                (150,170,210) if is_major else (55,70,105),
                (int(CX+math.cos(tr)*inner), int(CY+math.sin(tr)*inner)),
                (int(CX+math.cos(tr)*R),     int(CY+math.sin(tr)*R)),
                2 if is_major else 1)

        for label, deg in [("E",0),("S",90),("W",180),("N",270)]:
            lr = math.radians(deg)
            lx = int(CX+math.cos(lr)*(R-13)); ly = int(CY+math.sin(lr)*(R-13))
            ls = font_tiny.render(label, True, (100,125,170))
            surf.blit(ls, (lx-ls.get_width()//2, ly-ls.get_height()//2))

        ar = math.radians(angle_deg)
        ex = int(CX+math.cos(ar)*(R-5)); ey = int(CY+math.sin(ar)*(R-5))
        pygame.draw.line(surf, DIAL_NEEDLE, (CX,CY), (ex,ey), 2)
        pygame.draw.circle(surf, DIAL_NEEDLE, (ex,ey), 4)
        pygame.draw.circle(surf, BG, (CX,CY), 3)

        ds = font_sm.render(f"{angle_deg%360:.0f}°", True, DIAL_TEXT)
        surf.blit(ds, (CX-ds.get_width()//2, CY+R+4))
        pygame.draw.line(surf, (45,60,95), node_screen_pos, (CX,CY), 1)

        ex_y  = CY + R + 22
        ex_cx = CX
        mag_str = f"{magnitude:.0f} N"
        ms = font_sm.render(mag_str, True, LOAD_COL)
        BW = self.BTN_W; BH = self.BTN_H
        total_w = BW + 6 + ms.get_width() + 6 + BW
        start_x = ex_cx - total_w // 2
        row_y   = ex_y + 14

        self._btn_minus = pygame.Rect(start_x, row_y, BW, BH)
        self._btn_plus  = pygame.Rect(start_x + BW+6+ms.get_width()+6, row_y, BW, BH)
        self._hover_minus = self._btn_minus.collidepoint(mp)
        self._hover_plus  = self._btn_plus.collidepoint(mp)

        for rect, lbl, hov in [
            (self._btn_minus, "-10", self._hover_minus),
            (self._btn_plus,  "+10", self._hover_plus),
        ]:
            pygame.draw.rect(surf, BTN_HOV if hov else BTN_BG, rect, border_radius=4)
            pygame.draw.rect(surf, (60,80,130), rect, 1, border_radius=4)
            ls2 = font_tiny.render(lbl, True, WHITE)
            surf.blit(ls2, (rect.x+rect.w//2-ls2.get_width()//2,
                            rect.y+rect.h//2-ls2.get_height()//2))

        surf.blit(ms, (start_x+BW+6, row_y+BH//2-ms.get_height()//2))

        row_y2 = row_y + BH + 4
        BW2 = 38
        sx2 = ex_cx - (BW2*2+6) // 2
        self._btn_minus_big = pygame.Rect(sx2,       row_y2, BW2, BH)
        self._btn_plus_big  = pygame.Rect(sx2+BW2+6, row_y2, BW2, BH)
        self._hover_minus_big = self._btn_minus_big.collidepoint(mp)
        self._hover_plus_big  = self._btn_plus_big.collidepoint(mp)

        for rect, lbl, hov in [
            (self._btn_minus_big, "-100", self._hover_minus_big),
            (self._btn_plus_big,  "+100", self._hover_plus_big),
        ]:
            pygame.draw.rect(surf, BTN_HOV if hov else BTN_BG, rect, border_radius=4)
            pygame.draw.rect(surf, (60,80,130), rect, 1, border_radius=4)
            ls3 = font_tiny.render(lbl, True, (200,210,240))
            surf.blit(ls3, (rect.x+rect.w//2-ls3.get_width()//2,
                            rect.y+rect.h//2-ls3.get_height()//2))

        hint = font_tiny.render("[/] ±10  {/} ±100  scroll ±10  <>^v angle", True, DIM)
        surf.blit(hint, (ex_cx-hint.get_width()//2, row_y2+BH+4))

    def handle_click(self, pos):
        if self._btn_minus.collidepoint(pos):     return -10.0
        if self._btn_plus.collidepoint(pos):       return  10.0
        if self._btn_minus_big.collidepoint(pos): return -100.0
        if self._btn_plus_big.collidepoint(pos):   return  100.0
        return None


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(members):
    g = {}
    for (a, b) in members:
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set()).add(a)
    return g

def node_degrees(members):
    deg = {}
    for (a, b) in members:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return deg

def is_connected(members, anchors, loads):
    if not loads or not anchors: return False
    g = build_graph(members)
    anchor_set = set(anchors)
    for load in loads:
        if load not in g: return False
        visited, q, found = {load}, deque([load]), False
        while q:
            cur = q.popleft()
            if cur in anchor_set: found = True; break
            for nb in g.get(cur, ()):
                if nb not in visited: visited.add(nb); q.append(nb)
        if not found: return False
    return True

def steiner_lower_bound(members_list, required_nodes, anchors):
    if not required_nodes or not anchors:
        return 0.0
    graph: dict = {}
    for (a, b) in members_list:
        d = math.dist(a, b)
        graph.setdefault(a, {})[b] = d
        graph.setdefault(b, {})[a] = d

    dist = {}
    heap = []
    for a in anchors:
        dist[a] = 0.0
        heapq.heappush(heap, (0.0, a))

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")): continue
        for v, w in graph.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    h = 0.0
    for nd in required_nodes:
        if dist.get(nd, float("inf")) < float("inf"):
            continue
        h += min(math.dist(nd, a) for a in anchors)
    return h


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def maxwell_check(members, anchors, node_set):
    m = len(members); r = len(anchors) * 2; j = len(node_set)
    diff = m + r - 2 * j
    if diff == 0:  return 'ok'
    if diff > 0:   return 'over'
    return 'under'

def solve_truss(members, anchors, loads, p, load_angles, load_magnitudes):
    if not members: return False, {}, set()
    node_set = set(anchors) | set(loads)
    for a, b in members: node_set.add(a); node_set.add(b)
    if maxwell_check(members, anchors, node_set) == 'under':
        return False, {}, set()

    nodes = list(node_set); nidx = {n: i for i, n in enumerate(nodes)}
    j, m, r = len(nodes), len(members), len(anchors) * 2
    A = np.zeros((2*j, m+r)); b_vec = np.zeros(2*j)

    for ci, (n1, n2) in enumerate(members):
        i1, i2 = nidx[n1], nidx[n2]
        dx, dy = n2[0]-n1[0], n2[1]-n1[1]; L = math.hypot(dx, dy)
        if L < 1e-9: return False, {}, set()
        cx, cy = dx/L, dy/L
        A[2*i1, ci] = -cx; A[2*i1+1, ci] = -cy
        A[2*i2, ci] =  cx; A[2*i2+1, ci] =  cy

    rc = m
    for anc in anchors:
        idx = nidx[anc]; A[2*idx, rc] = 1.0; A[2*idx+1, rc+1] = 1.0; rc += 2

    for ld in loads:
        angle_deg = load_angles.get(ld, DEFAULT_LOAD_ANGLE)
        mag       = load_magnitudes.get(ld, p.applied_load)
        ar        = math.radians(angle_deg)
        idx       = nidx[ld]
        b_vec[2*idx  ] += mag * math.cos(ar)
        b_vec[2*idx+1] += mag * math.sin(ar)

    try:
        x, _, rank, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        if rank < min(A.shape): return False, {}, set()
        if np.linalg.norm(A @ x - b_vec) > 1e-3: return False, {}, set()
        forces = {members[i]: float(x[i]) for i in range(m)}
        if max(abs(f) for f in forces.values()) > p.T_MAX:
            return False, forces, set()
        buckled = set()
        for mem, f in forces.items():
            if f < 0:
                pcr = p.euler_pcr(math.dist(*mem)) / p.safety_factor
                if abs(f) > pcr: buckled.add(mem)
        return len(buckled) == 0, forces, buckled
    except Exception:
        return False, {}, set()


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED A* SEARCH
# ══════════════════════════════════════════════════════════════════════════════

class TrussStateV4:
    __slots__ = ("members_fs","member_hash","frontier_fs","g","h","f","n_members")

    def __init__(self, members_fs, member_hash, frontier_fs, g, anchors_fs, required_fs):
        self.members_fs  = members_fs
        self.member_hash = member_hash
        self.frontier_fs = frontier_fs
        self.n_members   = len(members_fs)
        self.g = g
        self.h = steiner_lower_bound(members_fs, required_fs, anchors_fs)
        self.f = g + self.h

    def __lt__(self, other): return self.f < other.f


def _edge_key(a, b):
    return (a, b) if a <= b else (b, a)

def run_astar(anchors, loads, noloads, p, load_angles, load_magnitudes,
              fixed_members=None, blacklist=None):
    anchors_fs  = frozenset(anchors)
    loads_fs    = frozenset(loads)
    required_fs = loads_fs | frozenset(noloads)
    all_nodes   = list(anchors_fs | required_fs)

    fixed       = list(fixed_members) if fixed_members else []
    fixed_fs    = frozenset(map(lambda m: _edge_key(*m), fixed))
    bl          = set(blacklist) if blacklist else set()

    log = [f"A* v4 | {len(loads)} load, {len(noloads)} no-load, {len(anchors)} anchor"]

    if not anchors or not loads:
        log.append("  ✗ Need anchors and load nodes.")
        return list(fixed_fs), {}, set(), log, float("inf")

    init_hash = 0
    for edge in fixed_fs: init_hash ^= zob(edge)

    def node_priority(n):
        return min((math.dist(n, r) for r in required_fs), default=0.0)
    all_nodes_sorted = sorted(all_nodes, key=node_priority)

    start = TrussStateV4(fixed_fs, init_hash, required_fs,
                         sum(math.dist(*e) for e in fixed_fs),
                         anchors_fs, required_fs)
    heap    = [(start.f, 0, start)]
    counter = 0
    explored: dict = {}
    best         = None
    best_g       = float("inf")
    states_tried = 0

    while heap:
        _, _, cur = heapq.heappop(heap)
        states_tried += 1

        vis_key = (cur.member_hash, cur.n_members)
        prev_g  = explored.get(vis_key, float("inf"))
        if cur.g >= prev_g: continue
        explored[vis_key] = cur.g
        if cur.g >= best_g: continue

        if is_connected(list(cur.members_fs), anchors_fs, required_fs):
            cand_key = frozenset(_edge_key(*e) for e in cur.members_fs)
            if cand_key not in bl:
                node_set = set(anchors)
                for a, b in cur.members_fs: node_set.add(a); node_set.add(b)
                node_set.update(loads)
                if maxwell_check(list(cur.members_fs), anchors_fs, node_set) != 'under':
                    ok, forces, buckled = solve_truss(
                        list(cur.members_fs), anchors_fs, loads_fs, p,
                        load_angles, load_magnitudes)
                    if forces:
                        if ok and cur.g < best_g:
                            best_g = cur.g
                            best   = (cur.g, cur.members_fs, forces, buckled)
                            log.append(f"  ✓ {len(cur.members_fs)} mbr  "
                                       f"{cur.g/p.pixels_per_m*100:.1f}cm")
                            continue
                        elif not ok and buckled and cur.g < best_g:
                            log.append(f"  ⚠ buckled {len(cur.members_fs)} mbr")
                            return list(cur.members_fs), forces, buckled, log, cur.g

        deg     = node_degrees(list(cur.members_fs))
        special = anchors_fs | required_fs

        g_cur   = build_graph(list(cur.members_fs))
        reachable = set(anchors_fs)
        bfs_q = deque(anchors_fs)
        while bfs_q:
            u = bfs_q.popleft()
            for v in g_cur.get(u, ()):
                if v not in reachable: reachable.add(v); bfs_q.append(v)

        for fn in cur.frontier_fs:
            if fn not in special and deg.get(fn, 0) >= MAX_DEGREE: continue
            for other in all_nodes_sorted:
                if other == fn: continue
                ek = _edge_key(fn, other)
                if ek in cur.members_fs: continue
                if other not in special and deg.get(other, 0) >= MAX_DEGREE: continue
                if fn in reachable and other in reachable:
                    if fn not in required_fs and other not in required_fs: continue
                new_g = cur.g + math.dist(fn, other)
                if new_g >= best_g: continue
                new_hash = cur.member_hash ^ zob(ek)
                new_n    = cur.n_members + 1
                if new_g >= explored.get((new_hash, new_n), float("inf")): continue
                ns = TrussStateV4(cur.members_fs | {ek}, new_hash,
                                  cur.frontier_fs | {other}, new_g,
                                  anchors_fs, required_fs)
                if ns.f < best_g:
                    counter += 1
                    heapq.heappush(heap, (ns.f, counter, ns))

        if len(heap) > BEAM_WIDTH * 4:
            heap = heapq.nsmallest(BEAM_WIDTH, heap)
            heapq.heapify(heap)

        if states_tried >= MAX_STATES:
            log.append(f"  ! State cap reached."); break

    if best:
        _, mfs, forces, buckled = best
        log.append(f"  * Optimal: {best_g/p.pixels_per_m*100:.1f}cm, "
                   f"{len(mfs)} members. States: {states_tried}")
        return list(mfs), forces, buckled, log, best_g
    else:
        log.append(f"  ✗ No valid truss. States: {states_tried}")
        return list(fixed_fs), {}, set(), log, float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS  (all world-space coords; vp = Viewport passed in)
# ══════════════════════════════════════════════════════════════════════════════

def lerp_col(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i]+(c2[i]-c1[i])*t) for i in range(3))

def draw_canvas_area(surf, vp: Viewport, settings_open):
    """Draw grid lines in screen space, respecting pan/zoom."""
    sp     = 40  # world-space grid spacing
    x0_px  = SETTINGS_W if settings_open else 0
    # Grid step in screen space
    sp_s   = sp * vp.zoom
    if sp_s < 6: return   # too dense to draw

    # Starting world coord just left/above screen
    wx_start = (x0_px - vp.pan_x) / vp.zoom
    wy_start = (0     - vp.pan_y) / vp.zoom
    wx_end   = (WIDTH  - vp.pan_x) / vp.zoom
    wy_end   = ((HEIGHT - PANEL_H) - vp.pan_y) / vp.zoom

    # Snap to grid
    wx0 = math.floor(wx_start / sp) * sp
    wy0 = math.floor(wy_start / sp) * sp

    wx = wx0
    while wx <= wx_end:
        sx, _ = vp.w2s(wx, 0)
        if sx >= x0_px:
            pygame.draw.line(surf, GRID_COL, (int(sx), 0), (int(sx), HEIGHT-PANEL_H))
        wx += sp

    wy = wy0
    while wy <= wy_end:
        _, sy = vp.w2s(0, wy)
        if 0 <= sy <= HEIGHT-PANEL_H:
            pygame.draw.line(surf, GRID_COL, (x0_px, int(sy)), (WIDTH, int(sy)))
        wy += sp


def draw_member_line(surf, vp: Viewport, a, b, force=None, max_force=1.0,
                     manual=False, buckled=False,
                     label_mode="both", font_tiny=None, p=None, sizing=None):
    sa = vp.w2si(*a)
    sb = vp.w2si(*b)
    L_px = math.dist(a, b)
    pcr  = (p.euler_pcr(L_px) / p.safety_factor) if p else float("inf")

    if buckled:
        color, w = MBR_BUCKLE, max(2, int(4 * vp.zoom))
    elif force is not None and max_force > 1e-9:
        t = min(1.0, abs(force)/max_force)
        if force > 0:
            color = lerp_col((20,60,140), MBR_TENSION, 0.4+0.6*t)
        else:
            t_b = min(1.0, abs(force)/pcr) if pcr < float("inf") else 0.0
            base = lerp_col((80,20,20), MBR_COMPRESS, 0.4+0.6*t)
            color = lerp_col(base, MBR_BUCKLE, t_b*0.45)
        w = max(2, int((2+4*t) * max(0.5, min(2.0, vp.zoom))))
    else:
        color, w = MBR_UNLOADED, max(1, int(2 * max(0.5, min(2.0, vp.zoom))))

    pygame.draw.line(surf, color, sa, sb, w)
    nr = max(2, int(4 * max(0.4, min(2.0, vp.zoom))))
    pygame.draw.circle(surf, WHITE, sa, nr)
    pygame.draw.circle(surf, WHITE, sb, nr)

    # Labels — only draw when zoom is large enough to be readable
    if label_mode != "none" and font_tiny and vp.zoom >= 0.5:
        mid_w = ((a[0]+b[0])/2, (a[1]+b[1])/2)
        mid_s = vp.w2s(*mid_w)
        dx = b[0]-a[0]; dy = b[1]-a[1]; Lv = math.hypot(dx,dy) or 1
        offset = 14
        ox2 = int(mid_s[0]+(-dy/Lv)*offset)
        oy2 = int(mid_s[1]+( dx/Lv)*offset)
        rows = []
        if label_mode in ("length","both"):
            cm = L_px/(p.pixels_per_m if p else 200)*100
            rows.append((f"{cm:.1f}cm", lerp_col(DIM,WHITE,0.55)))
        if label_mode in ("force","both") and force is not None:
            kind = "T" if force > 0 else "C"
            fcol = MBR_BUCKLE if buckled else (MBR_TENSION if force > 0 else MBR_COMPRESS)
            rows.append((f"{abs(force):.1f}N {kind}", fcol))
        if label_mode == "sizing" and sizing is not None:
            dom_col = MBR_COMPRESS if sizing["dominated"]=="buckling" else MBR_TENSION
            rows.append((f"r={sizing['r_o_mm']:.1f}mm t={sizing['t_mm']:.1f}mm", dom_col))
            rows.append((f"V={sizing['volume_cm3']:.3f}cm³", (180,190,210)))
        for li, (txt, col) in enumerate(rows):
            s = font_tiny.render(txt, True, col)
            lx = ox2-s.get_width()//2; ly = oy2-s.get_height()//2+li*13
            pad = 2
            bg2 = pygame.Surface((s.get_width()+pad*2, s.get_height()+pad*2), pygame.SRCALPHA)
            bg2.fill((8,10,18,185)); surf.blit(bg2, (lx-pad,ly-pad)); surf.blit(s, (lx,ly))

    if buckled:
        mid2 = vp.w2si((a[0]+b[0])/2, (a[1]+b[1])/2)
        dx = b[0]-a[0]; dy = b[1]-a[1]; Lv = math.hypot(dx,dy) or 1
        perp = (-dy/Lv*9, dx/Lv*9)
        sca = vp.w2s(*a); scb = vp.w2s(*b)
        pts = [sca,
               (mid2[0]+perp[0], mid2[1]+perp[1]),
               (mid2[0]-perp[0], mid2[1]-perp[1]),
               scb]
        pygame.draw.lines(surf, MBR_BUCKLE, False,
                          [(int(q[0]),int(q[1])) for q in pts], 2)


def draw_node(surf, vp: Viewport, wpos, kind, hover=False, selected=False):
    sx, sy = vp.w2si(*wpos)
    ring = SEL_COL if selected else (HOVER_COL if hover else WHITE)
    z = vp.zoom

    if kind == "anchor":
        s = max(7, int(13 * max(0.5, min(2.0, z))))
        pts = [(sx, sy-s),(sx-s, sy+s),(sx+s, sy+s)]
        pygame.draw.polygon(surf, ANCHOR_COL, pts)
        pygame.draw.polygon(surf, ring, pts, 2)
        step = max(3, int(6 * max(0.5, min(2.0, z))))
        for i in range(-2, 4):
            bx = sx-s+i*step
            pygame.draw.line(surf, ANCHOR_COL, (bx, sy+s), (bx-step, sy+s+max(4,step)), 2)

    elif kind == "load":
        r = max(5, int(10 * max(0.5, min(2.0, z))))
        pygame.draw.circle(surf, LOAD_COL, (sx,sy), r)
        pygame.draw.circle(surf, ring, (sx,sy), r, 2)

    elif kind == "noload":
        s = max(5, int(10 * max(0.5, min(2.0, z))))
        pts = [(sx, sy-s),(sx+s, sy),(sx, sy+s),(sx-s, sy)]
        pygame.draw.polygon(surf, BG, pts)
        pygame.draw.polygon(surf, NOLOAD_COL, pts, 2)
        if selected or hover:
            pygame.draw.polygon(surf, ring, pts, 2)


def draw_load_arrow(surf, vp: Viewport, wpos, angle_deg, magnitude, p, font_tiny):
    wx, wy = wpos
    scale  = max(0.5, min(2.5, magnitude / max(p.applied_load, 1.0)))
    length = int(20 + 16 * scale)  # world-space length
    ar = math.radians(angle_deg)
    dx = math.cos(ar)*length; dy = math.sin(ar)*length
    # tail and tip in world space
    tail_w = (wx-dx, wy-dy)
    tip_w  = (wx,    wy   )
    tail_s = vp.w2si(*tail_w)
    tip_s  = vp.w2si(*tip_w)

    col = lerp_col(LOAD_COL, (255,255,100), max(0.0,
          min(1.0, magnitude/max(p.applied_load,1.0)*0.5+0.5) - 1.0))
    pygame.draw.line(surf, col, tail_s, tip_s, max(1, int(3*max(0.5,min(2.0,vp.zoom)))))

    # arrowhead
    screen_len = math.dist(tail_s, tip_s)
    if screen_len > 4:
        ahlen = min(12, length)
        norm_dx = dx/length; norm_dy = dy/length
        perp_x  = -norm_dy * 6; perp_y = norm_dx * 6
        p1 = vp.w2si(wx-norm_dx*ahlen + perp_x, wy-norm_dy*ahlen + perp_y)
        p2 = vp.w2si(wx-norm_dx*ahlen - perp_x, wy-norm_dy*ahlen - perp_y)
        pygame.draw.polygon(surf, col, [tip_s, p1, p2])

    if font_tiny and vp.zoom >= 0.5:
        ls = font_tiny.render(f"{magnitude:.0f}N", True, lerp_col(LOAD_COL,WHITE,0.4))
        surf.blit(ls, (tail_s[0]-ls.get_width()//2-2, tail_s[1]-14))


def draw_force_table(surf, fonts, members, forces, buckled_set,
                     manual_indices, p, sizing_map=None):
    if not forces: return
    font_sm, font_md, font_tiny = fonts
    rows = []
    for i, (a, b) in enumerate(members):
        f = forces.get((a,b), forces.get((b,a), None))
        if f is None: continue
        L_px = math.dist(a, b); L_cm = L_px/p.pixels_per_m*100
        bk   = (a,b) in buckled_set or (b,a) in buckled_set
        mn   = i in manual_indices
        sz   = (sizing_map.get((a,b), sizing_map.get((b,a),None))) if sizing_map else None
        rows.append((abs(f), f, L_cm, bk, mn, i+1, sz))
    rows.sort(key=lambda r: r[0], reverse=True)

    PAD, ROW_H, BAR_W, COL_W = 8, 18, 70, 195
    pw = COL_W + BAR_W + PAD*3
    ph = min(PAD*2+22+len(rows)*ROW_H+22, HEIGHT-PANEL_H-20)
    px = WIDTH-pw-8; py = 8

    bg = pygame.Surface((pw,ph), pygame.SRCALPHA); bg.fill((12,15,28,215))
    surf.blit(bg, (px,py))
    pygame.draw.rect(surf, (45,55,85), (px,py,pw,ph), 1)
    surf.blit(font_sm.render("MEMBER FORCES", True, (175,192,222)), (px+PAD, py+PAD))
    pygame.draw.line(surf, (40,52,82), (px,py+22+PAD), (px+pw,py+22+PAD), 1)

    max_abs = max(r[0] for r in rows) if rows else 1.0
    vis = (ph-22-PAD*2-22)//ROW_H
    for ri, (abs_f, f, L_cm, bk, mn, midx, sz) in enumerate(rows[:vis]):
        ry = py+24+PAD+ri*ROW_H
        if ri % 2 == 0:
            rb = pygame.Surface((pw-2,ROW_H), pygame.SRCALPHA); rb.fill((255,255,255,7))
            surf.blit(rb, (px+1,ry))
        if bk:    tc, tcol = "B!", MBR_BUCKLE
        elif f>0: tc, tcol = "T",  MBR_TENSION
        else:     tc, tcol = "C",  MBR_COMPRESS
        if mn and not bk: tcol = lerp_col(tcol, (180,170,255), 0.35)
        surf.blit(font_tiny.render(f"#{midx:02d}", True, (110,130,160)), (px+PAD,ry+2))
        surf.blit(font_tiny.render(tc, True, tcol), (px+PAD+28,ry+2))
        surf.blit(font_tiny.render(f"{abs_f:6.1f}N", True, tcol), (px+PAD+42,ry+2))
        surf.blit(font_tiny.render(f"{L_cm:5.1f}cm", True, (95,115,150)), (px+PAD+105,ry+2))
        if sz:
            surf.blit(font_tiny.render(f"r{sz['r_o_mm']:.1f}", True, (85,105,140)),
                      (px+PAD+152,ry+2))
        bx2 = px+COL_W+PAD; bm = BAR_W-4; bl2 = int(bm*abs_f/max_abs)
        pygame.draw.rect(surf, (28,36,56), (bx2,ry+4,bm,ROW_H-8))
        if bl2 > 0: pygame.draw.rect(surf, tcol, (bx2,ry+4,bl2,ROW_H-8))
    if len(rows) > vis:
        surf.blit(font_tiny.render(f"...{len(rows)-vis} more", True, DIM),
                  (px+PAD, py+ph-16))
    fy = py+ph-14
    for txt, col, ox2 in [("T=tension",MBR_TENSION,PAD),
                           ("C=compress",MBR_COMPRESS,70),
                           ("B!=buckled",MBR_BUCKLE,138)]:
        surf.blit(font_tiny.render(txt, True, col), (px+ox2, fy))


def draw_stress_legend(surf, font_sm, font_tiny):
    lx, ly = 14, HEIGHT-PANEL_H-68
    surf.blit(font_sm.render("tension",     True, MBR_TENSION),  (lx,    ly-16))
    surf.blit(font_sm.render("compression", True, MBR_COMPRESS), (lx+54, ly-16))
    for i in range(50):
        c = lerp_col((20,60,140), MBR_TENSION, i/49)
        pygame.draw.line(surf, c, (lx+i,ly), (lx+i,ly+8))
    for i in range(50):
        c = lerp_col((80,20,20), MBR_COMPRESS, i/49)
        pygame.draw.line(surf, c, (lx+50+i,ly), (lx+50+i,ly+8))
    pygame.draw.line(surf, MBR_BUCKLE, (lx,ly+18), (lx+100,ly+18), 3)
    surf.blit(font_sm.render("buckled", True, MBR_BUCKLE), (lx+4,ly+23))


def draw_buckle_banner(surf, fonts, n_buckled, attempt):
    font_sm, font_md, font_tiny = fonts
    alpha  = int(200+55*math.sin(time.time()*8))
    banner = pygame.Surface((WIDTH,36), pygame.SRCALPHA)
    banner.fill((180,60,0, min(255,alpha)))
    surf.blit(banner, (0,0))
    msg = (f"⚠  BUCKLED (attempt {attempt}) — "
           f"{n_buckled} member(s) buckle — searching next topology…")
    s = font_md.render(msg, True, (255,220,100))
    surf.blit(s, (WIDTH//2-s.get_width()//2, 8))


def draw_zoom_indicator(surf, vp: Viewport, font_tiny):
    """Small zoom % badge in the bottom-left of the canvas."""
    if abs(vp.zoom - 1.0) < 0.01: return
    txt = f"  {vp.zoom*100:.0f}%  "
    s = font_tiny.render(txt, True, WHITE)
    bx, by = 14, HEIGHT-PANEL_H-20
    pygame.draw.rect(surf, (20,28,50,200), (bx-2, by-2, s.get_width()+4, s.get_height()+4),
                     border_radius=3)
    surf.blit(s, (bx, by))


def draw_status_bar(surf, fonts, mode, status, solved, label_mode,
                    n_members, total_len, p, sel_load, load_angles, load_magnitudes, vp):
    font_sm, font_md, font_tiny = fonts
    bar = pygame.Surface((WIDTH,PANEL_H)); bar.fill(PANEL_COL)
    surf.blit(bar, (0,HEIGHT-PANEL_H))
    pygame.draw.line(surf, DIVIDER, (0,HEIGHT-PANEL_H), (WIDTH,HEIGHT-PANEL_H), 2)
    mc = {"ANCHOR":ANCHOR_COL,"LOAD":LOAD_COL,"MANUAL":MBR_MANUAL,"NOLOAD":NOLOAD_COL}
    ml = {"ANCHOR":"ANCHOR [A]","LOAD":"LOAD [L]","MANUAL":"MANUAL [M]","NOLOAD":"NO-LOAD [N]"}
    len_str = (f"  |  {total_len/p.pixels_per_m*100:.1f}cm" if total_len < float("inf") else "")
    zoom_str = f"  zoom:{vp.zoom*100:.0f}%"
    mode_str = ml.get(mode, mode)
    if mode == "LOAD" and sel_load is not None:
        ang = load_angles.get(sel_load, DEFAULT_LOAD_ANGLE) % 360
        mag = load_magnitudes.get(sel_load, p.applied_load)
        mode_str += f"  (sel: {ang:.0f}°  {mag:.0f}N)"
    lines = [
        (f"MODE: {mode_str}", mc.get(mode,WHITE), (14,HEIGHT-PANEL_H+6)),
        ("A=anchor  L=load  N=no-load  M=manual  S=settings  SPACE=solve  F=labels  R=reset  C=clear  H=reset view  Q=quit",
         DIM, (14,HEIGHT-PANEL_H+27)),
        (status, HIGHLIGHT if solved else TEXT_COL, (14,HEIGHT-PANEL_H+50)),
        (f"default={p.applied_load:.0f}N  E={p.E:.0f}GPa  "
         f"OD={p.outer_r_mm*2:.1f}mm  t={p.wall_mm:.1f}mm  "
         f"T_MAX={p.T_MAX:.1f}N  SF={p.safety_factor:.1f}{len_str}{zoom_str}  labels:{label_mode}",
         DIM, (14,HEIGHT-PANEL_H+74)),
        ("Wheel=zoom  Mid/Right-drag=pan  +/-=zoom  H=reset view  "
         "In LOAD: [/]±10N  {/}±100N  Ctrl+scroll=angle",
         (60,80,120), (14,HEIGHT-PANEL_H+92)),
    ]
    if n_members:
        lines.append((f"{n_members} members", WHITE, (WIDTH-150,HEIGHT-PANEL_H+6)))
    for txt, col, pos in lines:
        surf.blit(font_sm.render(txt, True, col), pos)


def nearest_node(screen_pos, nodes, vp: Viewport):
    """Find closest node to a screen-space position, using world-space snap radius."""
    wx, wy = vp.s2w(*screen_pos)
    snap_r = vp.snap_radius_world()
    best, best_d = None, snap_r
    for n in nodes:
        d = math.dist((wx, wy), n)
        if d < best_d: best, best_d = n, d
    return best


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Truss Builder v5 — zoom + pan")
    clock = pygame.time.Clock()
    try:
        font_sm   = pygame.font.SysFont("Consolas", 15)
        font_md   = pygame.font.SysFont("Consolas", 17, bold=True)
        font_tiny = pygame.font.SysFont("Consolas", 11)
    except Exception:
        font_sm   = pygame.font.SysFont(None, 15)
        font_md   = pygame.font.SysFont(None, 17)
        font_tiny = pygame.font.SysFont(None, 11)
    fonts = (font_sm, font_md, font_tiny)

    p        = Params()
    settings = SettingsPanel()
    editor   = LoadEditor()
    vp       = Viewport()

    # Node data — all in world space (same pixel units as v3/v4)
    anchors         = []
    loads           = []
    noloads         = []
    load_angles     = {}
    load_magnitudes = {}

    members        = []
    forces         = {}
    buckled_set    = set()
    manual_indices = set()
    member_sizing  = {}

    mode       = "ANCHOR"
    status     = "Place anchors [A], loads [L], no-load [N], then SPACE. Wheel=zoom, mid-drag=pan."
    solved     = False
    sel_node   = None   # MANUAL first-click (world coords)
    sel_load   = None   # selected load node (world coords)
    total_len  = float("inf")
    label_mode = "both"

    # Right-click pan state (separate from middle-button pan)
    _rclick_pan     = False
    _rclick_pan_pos = (0, 0)  # screen pos where right-click started
    _rclick_moved   = False   # did the mouse move before release?

    def all_placed_nodes():
        return list(set(anchors) | set(loads) | set(noloads))

    def get_fixed():
        return [members[i] for i in sorted(manual_indices)]

    def clamp_mag(v):
        return max(MIN_LOAD_N, min(MAX_LOAD_N, v))

    def change_magnitude(node, delta):
        nonlocal forces, buckled_set, solved, total_len, member_sizing
        if node is None: return
        load_magnitudes[node] = clamp_mag(load_magnitudes.get(node, p.applied_load) + delta)
        forces.clear(); buckled_set.clear(); member_sizing.clear()
        solved = False; total_len = float("inf")

    def rotate_angle(node, delta_deg):
        nonlocal forces, buckled_set, solved, total_len, member_sizing
        if node is None: return
        load_angles[node] = load_angles.get(node, DEFAULT_LOAD_ANGLE) + delta_deg
        forces.clear(); buckled_set.clear(); member_sizing.clear()
        solved = False; total_len = float("inf")

    def apply_result(src_members, src_forces, src_buckled, src_len, fixed):
        nonlocal members, forces, buckled_set, manual_indices, total_len
        members.clear(); forces.clear(); buckled_set.clear(); manual_indices.clear()
        for mm in fixed:
            manual_indices.add(len(members)); members.append(mm)
        for m in src_members:
            norm = tuple(sorted([m[0], m[1]]))
            if not any(tuple(sorted([e[0],e[1]])) == norm for e in members):
                members.append(m)
        forces = {}
        for m, f in src_forces.items():
            if m in members: forces[m] = f
            elif (m[1],m[0]) in members: forces[(m[1],m[0])] = f
        buckled_set = set()
        for m in src_buckled:
            if m in members: buckled_set.add(m)
            elif (m[1],m[0]) in members: buckled_set.add((m[1],m[0]))
        total_len = src_len

    def do_solve():
        nonlocal solved, member_sizing, status
        if len(anchors) < 1 or len(loads) < 1:
            status = "Need >=1 anchor and >=1 load node."; return

        p.invalidate_cache()
        blacklist = set(); attempt = 0; fixed = get_fixed()

        while True:
            attempt += 1
            win.fill(BG)
            win.blit(font_md.render(f"A* Solving… attempt {attempt}", True, HIGHLIGHT),
                     (WIDTH//2-120, HEIGHT//2-24))
            win.blit(font_sm.render(p.summary(), True, DIM),
                     (WIDTH//2-220, HEIGHT//2+10))
            if noloads:
                win.blit(font_sm.render(
                    f"{len(noloads)} no-load node(s) as waypoints",
                    True, NOLOAD_COL), (WIDTH//2-130, HEIGHT//2+30))
            if blacklist:
                win.blit(font_sm.render(
                    f"Blacklisted {len(blacklist)} buckled topology(s)",
                    True, (200,120,60)), (WIDTH//2-150, HEIGHT//2+50))
            pygame.display.flip(); pygame.event.pump()

            t_start = time.time()

            new_m, new_f, new_b, log, opt_len = run_astar(
                anchors, loads, noloads, p,
                load_angles, load_magnitudes,
                fixed_members=fixed, blacklist=blacklist)
            t_end = time.time()     # ← AND THIS LINE HERE
            print(f"Attempt {attempt} | Solve time: {t_end - t_start:.2f}s | "
                    f"Members: {len(new_m)} | Nodes: {len(anchors)} anchors, "
                    f"{len(loads)} loads")

            for line in log: print(line)

            if not new_f:
                apply_result(new_m, new_f, new_b, opt_len, fixed)
                solved = False; member_sizing = {}
                status = "No valid truss — add intermediate nodes or adjust parameters."
                return

            if not new_b:
                apply_result(new_m, new_f, new_b, opt_len, fixed)
                solved = True; member_sizing = compute_member_sizing(members, forces, p)
                max_f  = max(abs(f) for f in new_f.values()) if new_f else 0
                pct    = 100 * max_f / p.T_MAX
                total_vol = sum(sz["volume_cm3"] for sz in member_sizing.values())
                status = (f"OK  {opt_len/p.pixels_per_m*100:.1f}cm  "
                          f"{len(members)} members  peak {pct:.0f}% T_MAX  "
                          f"min-vol={total_vol:.3f}cm³")
                return
            else:
                apply_result(new_m, new_f, new_b, opt_len, fixed)
                solved = False; n_b = len(new_b)
                status = f"Attempt {attempt}: {n_b} member(s) buckle — re-searching…"

                t_start = time.time()
                while time.time()-t_start < BUCKLE_PREVIEW_SEC:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT: pygame.quit(); sys.exit()
                        if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE):
                            pygame.quit(); sys.exit()
                    win.fill(BG); draw_canvas_area(win, vp, settings.open)
                    max_fv = max((abs(f) for f in forces.values()), default=1.0)
                    for i, (a, b) in enumerate(members):
                        f2 = forces.get((a,b), forces.get((b,a), None))
                        draw_member_line(win, vp, a, b, force=f2, max_force=max_fv,
                                         manual=(i in manual_indices),
                                         buckled=((a,b) in buckled_set or (b,a) in buckled_set),
                                         label_mode=label_mode, font_tiny=font_tiny, p=p)
                    for a in anchors: draw_node(win, vp, a, "anchor")
                    for ld in loads:
                        draw_node(win, vp, ld, "load")
                        draw_load_arrow(win, vp, ld,
                                        load_angles.get(ld, DEFAULT_LOAD_ANGLE),
                                        load_magnitudes.get(ld, p.applied_load),
                                        p, font_tiny)
                    for nl in noloads: draw_node(win, vp, nl, "noload")
                    draw_buckle_banner(win, fonts, n_b, attempt)
                    draw_force_table(win, fonts, members, forces, buckled_set, manual_indices, p)
                    draw_stress_legend(win, font_sm, font_tiny)
                    draw_status_bar(win, fonts, mode, status, False, label_mode,
                                    len(members), opt_len, p, sel_load,
                                    load_angles, load_magnitudes, vp)
                    pygame.display.flip(); clock.tick(60)

                blacklist.add(frozenset(tuple(sorted([m[0],m[1]])) for m in new_m))
                if attempt >= 8:
                    status = f"Gave up after {attempt} attempts. Adjust parameters or add nodes."
                    member_sizing = {}; return

    # ── canvas clip surface for scissoring ────────────────────────────────────
    def canvas_x0():
        return SETTINGS_W if (settings.open and settings._anim_x > -SETTINGS_W*0.5) else 0

    # ── Main loop ──────────────────────────────────────────────────────────────
    running = True
    while running:
        mp         = pygame.mouse.get_pos()
        all_nodes  = all_placed_nodes()
        hover_node = nearest_node(mp, all_nodes, vp)

        # ── Are we over the canvas (not settings or status bar)? ──────────────
        cx0 = canvas_x0()
        in_canvas = (mp[0] >= cx0 and mp[1] < HEIGHT-PANEL_H)

        win.fill(BG)

        # Clip drawing to canvas area
        canvas_surf = win.subsurface(
            pygame.Rect(cx0, 0, WIDTH-cx0, HEIGHT-PANEL_H))
        # We still draw on win but respect clip via subsurface

        draw_canvas_area(win, vp, settings.open and settings._anim_x > -SETTINGS_W*0.5)

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

            # ── Settings panel ──────────────────────────────────────────────
            if settings.open: settings.handle_event(event, p)

            # ── Middle mouse button: pan ─────────────────────────────────────
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2 and in_canvas:
                vp.start_pan(*mp)

            if event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                vp.stop_pan()

            # ── Mouse motion: middle-drag pan ────────────────────────────────
            if event.type == pygame.MOUSEMOTION:
                if vp.is_panning:
                    vp.update_pan(*mp)
                if _rclick_pan:
                    dx = mp[0]-_rclick_pan_pos[0]; dy = mp[1]-_rclick_pan_pos[1]
                    if abs(dx)+abs(dy) > 4:
                        _rclick_moved = True
                    if _rclick_moved:
                        vp.update_pan(*mp)

            # ── Mouse wheel: zoom centred on cursor ──────────────────────────
            if event.type == pygame.MOUSEWHEEL and in_canvas:
                mods = pygame.key.get_mods()
                if mode == "LOAD" and sel_load is not None:
                    # Ctrl+wheel → angle, plain wheel → magnitude (LOAD mode)
                    ctrl = bool(mods & pygame.KMOD_CTRL)
                    if ctrl:
                        rotate_angle(sel_load, event.y * 5.0)
                    else:
                        # Alt held → zoom even in LOAD mode
                        if bool(mods & pygame.KMOD_ALT):
                            vp.zoom_at(mp[0], mp[1], ZOOM_STEP ** event.y)
                        else:
                            change_magnitude(sel_load, event.y * 10.0)
                            status = (f"Load: {load_magnitudes.get(sel_load,p.applied_load):.0f}N  "
                                      f"{load_angles.get(sel_load,DEFAULT_LOAD_ANGLE)%360:.0f}°")
                else:
                    vp.zoom_at(mp[0], mp[1], ZOOM_STEP ** event.y)

            # ── Key events ───────────────────────────────────────────────────
            if event.type == pygame.KEYDOWN:
                k = event.key; mods = pygame.key.get_mods()
                shift = bool(mods & pygame.KMOD_SHIFT)

                if k in (pygame.K_q, pygame.K_ESCAPE): running = False
                elif k == pygame.K_a: mode = "ANCHOR"; sel_node = None; sel_load = None
                elif k == pygame.K_l: mode = "LOAD";   sel_node = None
                elif k == pygame.K_n: mode = "NOLOAD"; sel_node = None; sel_load = None
                elif k == pygame.K_m: mode = "MANUAL"; sel_node = None; sel_load = None
                elif k == pygame.K_s: settings.toggle()
                elif k == pygame.K_f:
                    idx = LABEL_CYCLE.index(label_mode)
                    label_mode = LABEL_CYCLE[(idx+1) % len(LABEL_CYCLE)]
                    status = f"Labels: {label_mode.upper()}"

                # ── Viewport reset ───────────────────────────────────────────
                elif k in (pygame.K_h, pygame.K_HOME):
                    vp.reset(); status = "Viewport reset."

                # ── Zoom keys (+/-) ──────────────────────────────────────────
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    vp.zoom_step(+1, mp[0], mp[1])
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    vp.zoom_step(-1, mp[0], mp[1])

                elif k == pygame.K_c:
                    anchors.clear(); loads.clear(); noloads.clear()
                    load_angles.clear(); load_magnitudes.clear()
                    members.clear(); forces.clear()
                    buckled_set.clear(); manual_indices.clear(); member_sizing.clear()
                    sel_node = None; sel_load = None
                    solved = False; total_len = float("inf"); status = "Cleared."

                elif k == pygame.K_r:
                    fixed = get_fixed()
                    members.clear(); forces.clear()
                    buckled_set.clear(); manual_indices.clear(); member_sizing.clear()
                    for mm in fixed: manual_indices.add(len(members)); members.append(mm)
                    solved = False; total_len = float("inf"); status = "Auto members cleared."

                elif k == pygame.K_LEFTBRACKET and mode == "LOAD" and sel_load is not None:
                    change_magnitude(sel_load, -100.0 if shift else -10.0)
                    status = f"Load: {load_magnitudes.get(sel_load,p.applied_load):.0f}N"
                elif k == pygame.K_RIGHTBRACKET and mode == "LOAD" and sel_load is not None:
                    change_magnitude(sel_load, 100.0 if shift else 10.0)
                    status = f"Load: {load_magnitudes.get(sel_load,p.applied_load):.0f}N"

                elif (mode == "LOAD" and sel_load is not None
                      and k in (pygame.K_LEFT,pygame.K_RIGHT,pygame.K_UP,pygame.K_DOWN)):
                    step = (15.0 if k in (pygame.K_UP,pygame.K_DOWN) else 1.0)
                    if shift: step *= 10
                    if k in (pygame.K_LEFT,pygame.K_DOWN): step = -step
                    rotate_angle(sel_load, step)
                    status = f"Angle: {load_angles[sel_load]%360:.0f}°"

                elif k == pygame.K_SPACE:
                    do_solve()

            # ── Mouse button down ────────────────────────────────────────────
            if event.type == pygame.MOUSEBUTTONDOWN and not settings.any_active():
                in_settings_panel = (settings.open and mp[0] < SETTINGS_W
                                     and settings._anim_x > -SETTINGS_W*0.5)
                in_panel = (mp[1] >= HEIGHT-PANEL_H)

                if in_settings_panel or in_panel:
                    pass   # handled elsewhere

                elif event.button == 3:
                    # Start right-click: might be pan or node-remove
                    _rclick_pan     = True
                    _rclick_pan_pos = mp
                    _rclick_moved   = False
                    vp.start_pan(*mp)   # tentatively start pan

                elif event.button == 1 and in_canvas:
                    # Convert click to world space
                    wx, wy = vp.s2w(*mp)
                    wmp = (wx, wy)   # world-space mouse position (float)
                    # Snap to integer pixel grid for node placement
                    wmp_i = (int(round(wx)), int(round(wy)))

                    if mode == "ANCHOR":
                        if wmp_i not in loads and wmp_i not in anchors and wmp_i not in noloads:
                            anchors.append(wmp_i)
                            forces.clear(); buckled_set.clear(); member_sizing.clear()
                            solved = False; total_len = float("inf")

                    elif mode == "LOAD":
                        if sel_load is not None:
                            delta = editor.handle_click(mp)
                            if delta is not None:
                                change_magnitude(sel_load, delta)
                                status = f"Load: {load_magnitudes.get(sel_load,p.applied_load):.0f}N"
                                continue

                        clicked = nearest_node(mp, loads, vp)
                        if clicked:
                            sel_load = clicked
                            mag = load_magnitudes.get(sel_load, p.applied_load)
                            ang = load_angles.get(sel_load, DEFAULT_LOAD_ANGLE) % 360
                            status = f"Selected: {mag:.0f}N  {ang:.0f}°  [/] ±10N  scroll=mag"
                        else:
                            if wmp_i not in anchors and wmp_i not in noloads:
                                loads.append(wmp_i)
                                load_angles[wmp_i]     = DEFAULT_LOAD_ANGLE
                                load_magnitudes[wmp_i] = p.applied_load
                                sel_load = wmp_i
                                forces.clear(); buckled_set.clear(); member_sizing.clear()
                                solved = False; total_len = float("inf")
                                status = f"Load placed: {p.applied_load:.0f}N  90°"

                    elif mode == "NOLOAD":
                        if wmp_i not in anchors and wmp_i not in loads and wmp_i not in noloads:
                            noloads.append(wmp_i)
                            forces.clear(); buckled_set.clear(); member_sizing.clear()
                            solved = False; total_len = float("inf")
                            status = "No-load node placed."

                    elif mode == "MANUAL":
                        clicked = nearest_node(mp, all_nodes, vp)
                        if clicked:
                            if sel_node is None:
                                sel_node = clicked; status = "Click second node to connect."
                            elif clicked != sel_node:
                                pair = (sel_node, clicked); rev = (clicked, sel_node)
                                if pair not in members and rev not in members:
                                    members.append(pair)
                                    manual_indices.add(len(members)-1)
                                    forces.clear(); buckled_set.clear(); member_sizing.clear()
                                    solved = False; total_len = float("inf")
                                    status = "Manual member added."
                                else:
                                    status = "Already exists."
                                sel_node = None
                            else:
                                sel_node = None; status = "Deselected."

            # ── Mouse button up ──────────────────────────────────────────────
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    vp.stop_pan()
                    if not _rclick_moved:
                        # Treat as a right-click: remove node
                        target = nearest_node(mp, all_nodes, vp)
                        if target:
                            if target in anchors: anchors.remove(target)
                            if target in loads:
                                loads.remove(target)
                                load_angles.pop(target, None)
                                load_magnitudes.pop(target, None)
                                if sel_load == target: sel_load = None
                            if target in noloads: noloads.remove(target)
                            new_m2, new_mm = [], set()
                            for i, m in enumerate(members):
                                if target not in m:
                                    if i in manual_indices: new_mm.add(len(new_m2))
                                    new_m2.append(m)
                            members[:] = new_m2
                            manual_indices.clear(); manual_indices.update(new_mm)
                            forces.clear(); buckled_set.clear(); member_sizing.clear()
                            solved = False; total_len = float("inf"); status = "Node removed."
                    _rclick_pan = False

        # ── Draw members ───────────────────────────────────────────────────────
        max_f = max((abs(f) for f in forces.values()), default=1.0)
        for i, (a, b) in enumerate(members):
            f  = forces.get((a,b), forces.get((b,a), None))
            sz = member_sizing.get((a,b), member_sizing.get((b,a), None))
            draw_member_line(win, vp, a, b, force=f, max_force=max_f,
                             manual=(i in manual_indices),
                             buckled=((a,b) in buckled_set or (b,a) in buckled_set),
                             label_mode=label_mode, font_tiny=font_tiny, p=p, sizing=sz)

        # Manual preview line
        if mode == "MANUAL" and sel_node:
            wx, wy = vp.s2w(*mp)
            Lpx    = math.dist(sel_node, (wx, wy))
            col    = (255,180,0) if Lpx > p.L_MAX_PX else SEL_COL
            pygame.draw.line(win, col, vp.w2si(*sel_node), mp, 1)
            warn = " (may buckle)" if Lpx > p.L_MAX_PX else ""
            win.blit(font_sm.render(f"{Lpx/p.pixels_per_m*100:.1f}cm{warn}", True, col),
                     (mp[0]+12, mp[1]-18))

        # ── Draw nodes ─────────────────────────────────────────────────────────
        for a in anchors:
            draw_node(win, vp, a, "anchor",
                      hover=(a==hover_node and mode in ("MANUAL","ANCHOR")),
                      selected=(a==sel_node))
        for ld in loads:
            is_sel = (ld == sel_load)
            draw_node(win, vp, ld, "load",
                      hover=(ld==hover_node and mode in ("MANUAL","LOAD")),
                      selected=is_sel)
            draw_load_arrow(win, vp, ld,
                            load_angles.get(ld, DEFAULT_LOAD_ANGLE),
                            load_magnitudes.get(ld, p.applied_load),
                            p, font_tiny)
            if is_sel and mode == "LOAD":
                editor.draw(win,
                            vp.w2si(*ld),   # screen-space position for the dial
                            load_angles.get(ld, DEFAULT_LOAD_ANGLE),
                            load_magnitudes.get(ld, p.applied_load),
                            font_tiny, font_sm, mp)
        for nl in noloads:
            draw_node(win, vp, nl, "noload",
                      hover=(nl==hover_node and mode in ("MANUAL","NOLOAD")),
                      selected=(nl==sel_node))

        # ── Overlay UI (screen-space, not affected by viewport) ────────────────
        draw_force_table(win, fonts, members, forces, buckled_set, manual_indices, p,
                         sizing_map=member_sizing if label_mode=="sizing" else None)
        if solved and member_sizing and label_mode == "sizing":
            total_vol = sum(sz["volume_cm3"] for sz in member_sizing.values())
            sv = font_sm.render(f"Total min material volume: {total_vol:.4f} cm³",
                                True, (180,220,160))
            win.blit(sv, (14, 8))

        draw_stress_legend(win, font_sm, font_tiny)
        draw_zoom_indicator(win, vp, font_tiny)
        settings.draw(win, fonts, p)
        draw_status_bar(win, fonts, mode, status, solved, label_mode,
                        len(members), total_len, p, sel_load,
                        load_angles, load_magnitudes, vp)

        # Pan cursor hint
        if vp.is_panning or _rclick_pan:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
