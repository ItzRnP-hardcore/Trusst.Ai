"""
AI Truss Builder — A* + Live Settings Panel  (v2)
==================================================
Key changes from v1
-------------------
  1. A* search is NO LONGER pruned by L_max / Euler buckling.
     Members of any length are explored.  The solver only prunes
     on connectivity and node-degree limits.

  2. POST-SOLVE BUCKLING LOOP
     After A* finds a geometry solution the statics solver checks
     buckling.  If any member buckles the buckled solution is SHOWN
     briefly (purple members), then the search restarts with that
     member-set BLACKLISTED so A* finds the next-best topology.
     This repeats until a non-buckling solution is found or the
     search is exhausted.

  3. PER-MEMBER MINIMUM SIZING  (volume optimisation)
     After a valid solution is found, each member is individually
     sized to the MINIMUM outer radius + wall thickness (keeping the
     t/r ratio of the global cross-section) that simultaneously
     satisfies:
       • Tensile / compressive yield   |F| <= sigma_y * A
       • Euler buckling (SF applied)   |F| <= P_cr / SF
     The resulting minimum area, radius, and volume are stored and
     shown in the per-member labels (toggle with F key).

Per-node load angle editing
----------------------------
  In LOAD mode, LEFT-CLICK an existing load node to SELECT it.
  Then use ANY of these to set its force direction:
    Mouse-wheel         rotate ±5° per tick
    Left / Right        rotate ±1°
    Up / Down           rotate ±15°
    Hold Shift          ×10 multiplier on arrow keys
    Click elsewhere     place a new load node (deselects first)
  A live angle dial and degree readout are drawn next to the
  selected node.

Controls
---------
  A           Anchor mode           L  Load mode
  M           Manual member mode    S  Settings panel
  SPACE       Run A* solver         F  Cycle force labels
  R           Reset auto members    C  Clear all
  Q / Esc     Quit                  Right-click  Remove node

Colour convention
-----------------
  BLUE   = Tension   (+ve force)
  RED    = Compression (-ve force)
  PURPLE = Buckled   (compression exceeds Euler P_cr / SF)
  GREY   = Unloaded  (manual member, no solve yet)
"""

import pygame
import numpy as np
import math, heapq, sys, time
from collections import deque

# ── Window ─────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT  = 1100, 720
PANEL_H        = 100
SETTINGS_W     = 310
NODE_SNAP      = 22

# ── Palette ────────────────────────────────────────────────────────────────────
BG           = (10,  12,  20)
GRID_COL     = (20,  25,  40)
WHITE        = (228, 238, 255)
DIM          = ( 80, 100, 135)
ANCHOR_COL   = (255,  65,  65)
LOAD_COL     = ( 50, 215, 135)

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

BUCKLE_FLASH = (255, 140,   0)   # orange banner for buckled-preview

# ── Search limits ──────────────────────────────────────────────────────────────
MAX_STATES   = 120_000
MAX_DEGREE   = 4
LABEL_CYCLE  = ["both", "force", "length", "sizing", "none"]

DEFAULT_LOAD_ANGLE = 90.0   # straight down in pygame coords

# How long (seconds) to show the buckled solution before continuing
BUCKLE_PREVIEW_SEC = 1.8


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

    @property
    def outer_r(self):  return self.outer_r_mm * 1e-3
    @property
    def inner_r(self):  return max(0.0, self.outer_r - self.wall_mm * 1e-3)
    @property
    def t_r_ratio(self):
        """wall-thickness / outer-radius ratio — preserved in per-member sizing."""
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

    # NOTE: L_MAX_PX is kept for display / info only — NOT used to prune A*
    @property
    def L_MAX_PX(self):
        p_cr_min = self.T_MAX / self.safety_factor
        if p_cr_min <= 0: return float("inf")
        L_m = (math.pi / self.K_factor) * math.sqrt(self.E_pa * self.I / p_cr_min)
        return L_m * self.pixels_per_m

    def euler_pcr(self, length_px):
        Lm = length_px / self.pixels_per_m
        if Lm < 1e-9: return float("inf")
        return (math.pi**2 * self.E_pa * self.I) / (self.K_factor * Lm)**2

    def summary(self):
        return (f"E={self.E:.0f}GPa  sy={self.yield_mpa:.0f}MPa  "
                f"OD={self.outer_r_mm*2:.1f}mm  t={self.wall_mm:.1f}mm  "
                f"A={self.area*1e6:.2f}mm²  T_MAX={self.T_MAX:.1f}N")


# ══════════════════════════════════════════════════════════════════════════════
# PER-MEMBER MINIMUM SIZING
# ══════════════════════════════════════════════════════════════════════════════

def min_section_for_force(force_N, length_px, p):
    """
    Given a scalar axial force |force_N| and member length, find the
    minimum hollow circular cross-section (outer radius r_o, wall t = k*r_o)
    that satisfies BOTH:
      (a) Yield:    |F| <= sigma_y * A(r_o)          → A >= |F|/sigma_y
      (b) Buckling: |F| <= P_cr(r_o,L) / SF          → I >= |F|*SF*(KL)²/π²E
          (only if force is compressive)

    Wall thickness is kept at the same t/r ratio as the global section.
    Returns dict with r_o_mm, t_mm, area_mm2, volume_cm3, dominated_by.
    """
    abs_f  = abs(force_N)
    sy     = p.yield_mpa * 1e6
    E_pa   = p.E_pa
    K      = p.K_factor
    SF     = p.safety_factor
    k      = p.t_r_ratio          # t/r_o
    Lm     = length_px / p.pixels_per_m

    # --- (a) yield constraint: A = pi*(r_o^2 - r_i^2) = pi*r_o^2*(1-(1-k)^2) >= F/sy
    A_min  = abs_f / sy if sy > 0 else 0.0
    r_yield = math.sqrt(A_min / (math.pi * (1 - (1 - k)**2))) if A_min > 0 else 0.0

    # --- (b) buckling constraint (compression only)
    r_buckle = 0.0
    if force_N < 0 and Lm > 1e-9:
        # P_cr = pi^2*E*I / (K*L)^2  and  I = pi/4 * r_o^4 * (1-(1-k)^4)
        # Require P_cr/SF >= |F|  =>  I >= |F|*SF*(K*L)^2 / (pi^2*E)
        I_min    = abs_f * SF * (K * Lm)**2 / (math.pi**2 * E_pa)
        coeff    = math.pi / 4 * (1 - (1 - k)**4)
        r_buckle = (I_min / coeff) ** 0.25 if coeff > 0 else 0.0

    r_o = max(r_yield, r_buckle, 1e-4)   # at least 0.1 mm
    t   = k * r_o
    r_i = r_o - t
    A   = math.pi * (r_o**2 - r_i**2)
    vol = A * Lm * 1e6   # cm³  (A in m², L in m → m³ → ×1e6 cm³)

    dominated = "buckling" if r_buckle >= r_yield else "yield"
    return {
        "r_o_mm":     r_o * 1e3,
        "t_mm":       t   * 1e3,
        "area_mm2":   A   * 1e6,
        "volume_cm3": vol,
        "dominated":  dominated,
    }


def compute_member_sizing(members, forces, p):
    """Returns dict: member -> sizing-dict from min_section_for_force."""
    sizing = {}
    for mem in members:
        f = forces.get(mem, forces.get((mem[1], mem[0]), None))
        if f is None:
            f = 0.0
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
            return ((math.log(v)-math.log(self.lo)) /
                    (math.log(self.hi)-math.log(self.lo)))
        return (v-self.lo)/(self.hi-self.lo)

    def _from_t(self, t):
        t = max(0.0,min(1.0,t))
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
        if fw>0: pygame.draw.rect(surf, SLIDER_FILL, pygame.Rect(x, ty, fw, self.H), border_radius=3)
        kx=x+int(t*w); ky=ty+self.H//2
        pygame.draw.circle(surf, SLIDER_KNOB, (kx, ky), self.KNOB_R)
        pygame.draw.circle(surf, SLIDER_FILL,  (kx, ky), self.KNOB_R, 2)
        self.rect = pygame.Rect(x-self.KNOB_R, ty-self.KNOB_R, w+self.KNOB_R*2, self.H+self.KNOB_R*2)
        return y+38

    def handle_event(self, event, p):
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            if self.rect.collidepoint(event.pos): self.dragging=True
        if event.type==pygame.MOUSEBUTTONUP and event.button==1: self.dragging=False
        if event.type==pygame.MOUSEMOTION and self.dragging:
            rx=self.rect.x+self.KNOB_R; rw=self.rect.w-self.KNOB_R*2
            setattr(p, self.attr, self._from_t((event.pos[0]-rx)/rw))
            return True
        return False

    def is_active(self): return self.dragging


class SettingsPanel:
    PAD = 14
    def __init__(self):
        self.open = False; self._anim_x = -SETTINGS_W
        self.groups = [
            ("LOAD",  [Slider("Applied load per node","N","applied_load",10,2000,".0f")]),
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
        if abs(self._anim_x-target)<0.5: self._anim_x=target

    def draw(self, surf, fonts, p):
        self.update_anim()
        ox=int(self._anim_x)
        if ox<=-SETTINGS_W: return
        sw=SETTINGS_W; sh=HEIGHT-PANEL_H
        ps=pygame.Surface((sw,sh),pygame.SRCALPHA); ps.fill((*SETTINGS_BG,240))
        surf.blit(ps,(ox,0))
        pygame.draw.line(surf,DIVIDER,(ox+sw,0),(ox+sw,sh),1)
        font_sm,font_md,font_tiny=fonts
        py=self.PAD
        hs=pygame.Surface((sw,36),pygame.SRCALPHA); hs.fill((*SETTINGS_HDR,255))
        surf.blit(hs,(ox,py-self.PAD))
        surf.blit(font_md.render("  PARAMETERS",True,HIGHLIGHT),(ox+self.PAD,py))
        py+=36
        surf.blit(font_tiny.render("Changes apply on next SPACE solve",True,DIM),(ox+self.PAD,py))
        py+=20
        iw=sw-self.PAD*2
        for gname,sliders in self.groups:
            pygame.draw.line(surf,DIVIDER,(ox,py+4),(ox+sw,py+4),1)
            gh=font_tiny.render(gname,True,(100,120,160))
            pygame.draw.rect(surf,TAG_BG,pygame.Rect(ox+self.PAD-2,py-1,gh.get_width()+8,gh.get_height()+4),border_radius=3)
            surf.blit(gh,(ox+self.PAD+2,py)); py+=20
            for sl in sliders: py=sl.draw(surf,font_sm,font_tiny,p,ox+self.PAD,py,iw)
            py+=6
        pygame.draw.line(surf,DIVIDER,(ox,py),(ox+sw,py),1); py+=8
        for line in [
            f"Area   = {p.area*1e6:.3f} mm²",
            f"I      = {p.I*1e12:.4f} mm⁴",
            f"T_MAX  = {p.T_MAX:.1f} N",
            f"L_ref  = {p.L_MAX_PX/p.pixels_per_m*100:.1f} cm  (info only)",
        ]:
            surf.blit(font_tiny.render(line,True,(130,155,195)),(ox+self.PAD,py)); py+=15
        py+=6
        pygame.draw.line(surf,DIVIDER,(ox,py),(ox+sw,py),1); py+=8
        surf.blit(font_tiny.render("NOTE: L_max no longer limits A* search",True,(180,120,60)),(ox+self.PAD,py)); py+=14
        surf.blit(font_tiny.render("Buckling checked post-solve",True,(180,120,60)),(ox+self.PAD,py)); py+=20
        pygame.draw.line(surf,DIVIDER,(ox,py),(ox+sw,py),1); py+=8
        surf.blit(font_tiny.render("COLOUR KEY",True,(100,120,160)),(ox+self.PAD,py)); py+=14
        for col,lbl in [
            (MBR_TENSION, "BLUE   = Tension (+ve)"),
            (MBR_COMPRESS,"RED    = Compression (-ve)"),
            (MBR_BUCKLE,  "PURPLE = Buckled"),
            (MBR_UNLOADED,"GREY   = Unloaded"),
        ]:
            pygame.draw.rect(surf,col,pygame.Rect(ox+self.PAD,py+2,10,10))
            surf.blit(font_tiny.render(lbl,True,(160,175,210)),(ox+self.PAD+14,py)); py+=14

    def handle_event(self, event, p):
        changed=False
        for _,sliders in self.groups:
            for sl in sliders:
                if sl.handle_event(event,p): changed=True
        return changed

    def any_active(self):
        for _,sliders in self.groups:
            for sl in sliders:
                if sl.is_active(): return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
# ANGLE DIAL
# ══════════════════════════════════════════════════════════════════════════════

def draw_angle_dial(surf, node_pos, angle_deg, font_tiny, font_sm):
    nx,ny=node_pos; R=30; CX=nx+58; CY=ny
    if CX+R+65>WIDTH: CX=nx-58
    if CY-R-26<0: CY=ny+62
    if CY+R+36>HEIGHT-PANEL_H: CY=ny-62
    bg=pygame.Surface((R*2+6,R*2+6),pygame.SRCALPHA)
    pygame.draw.circle(bg,(12,18,40,215),(R+3,R+3),R+3)
    surf.blit(bg,(CX-R-3,CY-R-3))
    pygame.draw.circle(surf,DIAL_RING,(CX,CY),R,2)
    for tick_deg in range(0,360,30):
        tr=math.radians(tick_deg); is_major=(tick_deg%90==0)
        inner=R-(8 if is_major else 4)
        tx1=int(CX+math.cos(tr)*inner); ty1=int(CY+math.sin(tr)*inner)
        tx2=int(CX+math.cos(tr)*R);    ty2=int(CY+math.sin(tr)*R)
        col=(150,170,210) if is_major else (55,70,105)
        pygame.draw.line(surf,col,(tx1,ty1),(tx2,ty2),2 if is_major else 1)
    for label,deg in [("E",0),("S",90),("W",180),("N",270)]:
        lr=math.radians(deg)
        lx=int(CX+math.cos(lr)*(R-13)); ly=int(CY+math.sin(lr)*(R-13))
        ls=font_tiny.render(label,True,(100,125,170))
        surf.blit(ls,(lx-ls.get_width()//2,ly-ls.get_height()//2))
    ar=math.radians(angle_deg)
    ex=int(CX+math.cos(ar)*(R-5)); ey=int(CY+math.sin(ar)*(R-5))
    pygame.draw.line(surf,DIAL_NEEDLE,(CX,CY),(ex,ey),2)
    pygame.draw.circle(surf,DIAL_NEEDLE,(ex,ey),4)
    pygame.draw.circle(surf,BG,(CX,CY),3)
    ds=font_sm.render(f"{angle_deg%360:.0f} deg",True,DIAL_TEXT)
    surf.blit(ds,(CX-ds.get_width()//2,CY+R+5))
    hint=font_tiny.render("<> +/-1  ^v +/-15  scroll +/-5",True,DIM)
    surf.blit(hint,(CX-hint.get_width()//2,CY+R+21))
    pygame.draw.line(surf,(45,60,95),node_pos,(CX,CY),1)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(members):
    g={}
    for (a,b) in members:
        g.setdefault(a,set()).add(b); g.setdefault(b,set()).add(a)
    return g

def node_degrees(members):
    deg={}
    for (a,b) in members:
        deg[a]=deg.get(a,0)+1; deg[b]=deg.get(b,0)+1
    return deg

def is_connected(members, anchors, loads):
    if not loads or not anchors: return False
    g=build_graph(members); anchor_set=set(anchors)
    for load in loads:
        if load not in g: return False
        visited,q,found={load},deque([load]),False
        while q:
            cur=q.popleft()
            if cur in anchor_set: found=True; break
            for nb in g.get(cur,()):
                if nb not in visited: visited.add(nb); q.append(nb)
        if not found: return False
    return True

def dijkstra_dist(graph, start, targets):
    if start in targets: return 0.0
    dist={start:0.0}; heap=[(0.0,start)]
    while heap:
        d,u=heapq.heappop(heap)
        if u in targets: return d
        if d>dist.get(u,float("inf")): continue
        for v in graph.get(u,()):
            nd=d+math.dist(u,v)
            if nd<dist.get(v,float("inf")):
                dist[v]=nd; heapq.heappush(heap,(nd,v))
    return float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def solve_truss(members, anchors, loads, p, load_angles):
    """
    Method of joints.  Returns (ok, forces, buckled_set).
    ok=True only when statically determinate, within yield, and no buckling.
    Forces dict uses the member tuples as keys (as passed in).
    """
    if not members: return False, {}, set()
    node_set=set(anchors)|set(loads)
    for a,b in members: node_set.add(a); node_set.add(b)
    nodes=list(node_set); nidx={n:i for i,n in enumerate(nodes)}
    j,m,r=len(nodes),len(members),len(anchors)*2
    A=np.zeros((2*j,m+r)); b_vec=np.zeros(2*j)
    for ci,(n1,n2) in enumerate(members):
        i1,i2=nidx[n1],nidx[n2]
        dx,dy=n2[0]-n1[0],n2[1]-n1[1]; L=math.hypot(dx,dy)
        if L<1e-9: return False,{},set()
        cx,cy=dx/L,dy/L
        A[2*i1,ci]=-cx; A[2*i1+1,ci]=-cy
        A[2*i2,ci]= cx; A[2*i2+1,ci]= cy
    rc=m
    for anc in anchors:
        idx=nidx[anc]; A[2*idx,rc]=1.0; A[2*idx+1,rc+1]=1.0; rc+=2
    for ld in loads:
        angle_deg=load_angles.get(ld,DEFAULT_LOAD_ANGLE)
        ar=math.radians(angle_deg)
        idx=nidx[ld]
        b_vec[2*idx  ]+=p.applied_load*math.cos(ar)
        b_vec[2*idx+1]+=p.applied_load*math.sin(ar)
    try:
        x,_,rank,_=np.linalg.lstsq(A,b_vec,rcond=None)
        if rank<min(A.shape): return False,{},set()
        if np.linalg.norm(A@x-b_vec)>1e-3: return False,{},set()
        forces={members[i]:float(x[i]) for i in range(m)}
        if max(abs(f) for f in forces.values())>p.T_MAX:
            return False,forces,set()   # yield exceeded — return forces for display
        buckled=set()
        for mem,f in forces.items():
            if f<0:
                pcr=p.euler_pcr(math.dist(*mem))/p.safety_factor
                if abs(f)>pcr: buckled.add(mem)
        return len(buckled)==0, forces, buckled
    except Exception:
        return False,{},set()


# ══════════════════════════════════════════════════════════════════════════════
# A* SEARCH  (no L_max pruning — buckling handled post-solve)
# ══════════════════════════════════════════════════════════════════════════════

def _total_length(members):
    return sum(math.dist(a,b) for a,b in members)

def _heuristic(members_iter, loads, anchors):
    ml = list(members_iter)
    graph = build_graph(ml)
    anchor_set = set(anchors)
    h = 0.0
    
    for load in loads:
        # 1. Find the entire connected structure branching from this specific load
        visited = {load}
        q = deque([load])
        reached_anchor = False
        
        while q:
            cur = q.popleft()
            if cur in anchor_set:
                reached_anchor = True
                break
            for nb in graph.get(cur, ()):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
                    
        # 2. If this part of the structure has reached the ground, it needs 0 more steel
        if reached_anchor:
            continue
            
        # 3. THE FIX: If not connected yet, find the shortest "gap" from ANY node
        # we have built so far (the frontier) to ANY ground anchor. 
        min_gap = min(math.dist(n, a) for n in visited for a in anchor_set)
        h += min_gap
        
    return h

class TrussState:
    __slots__=("members","frontier","g","h","f")
    def __init__(self,members,frontier,anchors_fs,loads_fs):
        self.members=members; self.frontier=frontier
        self.g=_total_length(members)
        self.h=_heuristic(members,loads_fs,anchors_fs)
        self.f=self.g+self.h
    def __lt__(self,other): return self.f<other.f


def run_astar(anchors, loads, p, load_angles, fixed_members=None,
              blacklist=None, status_callback=None):
    """
    A* search.  No L_max pruning — members of any length are explored.
    blacklist: set of frozensets of member-sets to skip (previously buckled solutions).
    Returns (members, forces, buckled_set, log, best_g).
    """
    anchors_fs=frozenset(anchors); loads_fs=frozenset(loads)
    all_nodes=list(anchors_fs|loads_fs)
    fixed=list(fixed_members) if fixed_members else []
    fixed_fs=frozenset(fixed)
    bl=set(blacklist) if blacklist else set()

    start=TrussState(fixed_fs,frozenset(loads),anchors_fs,loads_fs)
    heap=[(start.f,0,start)]
    counter=0; explored={}
    best=None; best_g=float("inf")
    log=[f"A* search  Load={p.applied_load:.0f}N  E={p.E:.0f}GPa  "
         f"T_MAX={p.T_MAX:.1f}N  (no L_max prune)"]

    while heap:
        _,_,cur=heapq.heappop(heap)
        prev_g=explored.get(cur.members,float("inf"))
        if cur.g>=prev_g: continue
        explored[cur.members]=cur.g
        if cur.g>=best_g: continue

        if is_connected(list(cur.members),anchors_fs,loads_fs):
            # Skip blacklisted topologies
            if cur.members in bl:
                pass
            else:
                ok,forces,buckled=solve_truss(
                    list(cur.members),anchors_fs,loads_fs,p,load_angles)
                if forces:   # statically solved (may or may not buckle)
                    if ok and cur.g<best_g:
                        best_g=cur.g
                        best=(cur.g,cur.members,forces,buckled)
                        log.append(f"  ✓ {len(cur.members)} members, "
                                   f"{cur.g/p.pixels_per_m*100:.1f}cm  [s={counter}]")
                        continue
                    elif not ok and buckled and cur.g<best_g:
                        # Yield-failed — skip; buckled but geometrically valid — return for preview
                        if len(buckled)>0:
                            # This is a "buckled candidate" — return it so caller can show it
                            log.append(f"  ⚠ Buckled candidate {len(cur.members)} members "
                                       f"[s={counter}]")
                            return list(cur.members),forces,buckled,log,cur.g

        deg=node_degrees(list(cur.members)); special=anchors_fs|loads_fs
        for fn in cur.frontier:
            for other in all_nodes:
                if other==fn: continue
                pair=(fn,other); pair_r=(other,fn)
                if pair in cur.members or pair_r in cur.members: continue
                # No L_max check here — any length is allowed
                if fn    not in special and deg.get(fn,   0)>=MAX_DEGREE: continue
                if other not in special and deg.get(other,0)>=MAX_DEGREE: continue
                new_members=cur.members|{pair}
                new_g=cur.g+math.dist(fn,other)
                if new_g>=best_g: continue
                if new_g>=explored.get(new_members,float("inf")): continue
                ns=TrussState(new_members,cur.frontier|{other},anchors_fs,loads_fs)
                if ns.f<best_g:
                    counter+=1; heapq.heappush(heap,(ns.f,counter,ns))

        if counter>=MAX_STATES:
            log.append(f"  ! State cap ({MAX_STATES}) reached."); break

    if best:
        _,mfs,forces,buckled=best
        log.append(f"  * Optimal: {best_g/p.pixels_per_m*100:.1f}cm, {len(mfs)} members.")
        return list(mfs),forces,buckled,log,best_g
    else:
        log.append("  ✗ No valid truss found.")
        return list(fixed_fs),{},set(),log,float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def lerp_col(c1,c2,t):
    t=max(0.0,min(1.0,t))
    return tuple(int(c1[i]+(c2[i]-c1[i])*t) for i in range(3))

def draw_canvas_area(surf, settings_open):
    sp=40; x0=SETTINGS_W if settings_open else 0
    for x in range(x0,WIDTH,sp): pygame.draw.line(surf,GRID_COL,(x,0),(x,HEIGHT-PANEL_H))
    for y in range(0,HEIGHT-PANEL_H,sp): pygame.draw.line(surf,GRID_COL,(x0,y),(WIDTH,y))

def draw_member_line(surf, a, b, force=None, max_force=1.0,
                     manual=False, buckled=False,
                     label_mode="both", font_tiny=None, p=None,
                     sizing=None):
    L_px=math.dist(a,b)
    pcr=(p.euler_pcr(L_px)/p.safety_factor) if p else float("inf")

    if buckled:
        color,w=MBR_BUCKLE,4
    elif force is not None and max_force>1e-9:
        t=min(1.0,abs(force)/max_force)
        if force>0:
            color=lerp_col((20,60,140),MBR_TENSION,0.4+0.6*t)
        else:
            t_b=min(1.0,abs(force)/pcr) if pcr<float("inf") else 0.0
            base=lerp_col((80,20,20),MBR_COMPRESS,0.4+0.6*t)
            color=lerp_col(base,MBR_BUCKLE,t_b*0.45)
        w=max(2,int(2+4*t))
    else:
        color,w=MBR_UNLOADED,2

    pygame.draw.line(surf,color,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),w)
    pygame.draw.circle(surf,WHITE,(int(a[0]),int(a[1])),4)
    pygame.draw.circle(surf,WHITE,(int(b[0]),int(b[1])),4)

    if label_mode!="none" and font_tiny:
        mid=((a[0]+b[0])/2,(a[1]+b[1])/2)
        dx=b[0]-a[0]; dy=b[1]-a[1]; Lv=math.hypot(dx,dy) or 1
        ox=int(mid[0]+(-dy/Lv)*14); oy=int(mid[1]+(dx/Lv)*14)
        rows=[]
        if label_mode in ("length","both"):
            cm=L_px/(p.pixels_per_m if p else 200)*100
            rows.append((f"{cm:.1f}cm",lerp_col(DIM,WHITE,0.55)))
        if label_mode in ("force","both") and force is not None:
            kind="T" if force>0 else "C"
            fcol=MBR_BUCKLE if buckled else (MBR_TENSION if force>0 else MBR_COMPRESS)
            rows.append((f"{abs(force):.1f}N {kind}",fcol))
        if label_mode=="sizing" and sizing is not None:
            # Show per-member minimum section
            dom_col=(MBR_COMPRESS if sizing["dominated"]=="buckling" else MBR_TENSION)
            rows.append((f"r={sizing['r_o_mm']:.1f}mm t={sizing['t_mm']:.1f}mm", dom_col))
            rows.append((f"V={sizing['volume_cm3']:.3f}cm³", (180,190,210)))
        for li,(txt,col) in enumerate(rows):
            s=font_tiny.render(txt,True,col)
            lx=ox-s.get_width()//2; ly=oy-s.get_height()//2+li*13
            pad=2
            bg=pygame.Surface((s.get_width()+pad*2,s.get_height()+pad*2),pygame.SRCALPHA)
            bg.fill((8,10,18,185)); surf.blit(bg,(lx-pad,ly-pad)); surf.blit(s,(lx,ly))

    if buckled:
        mid2=((a[0]+b[0])//2,(a[1]+b[1])//2)
        dx=b[0]-a[0]; dy=b[1]-a[1]; Lv=math.hypot(dx,dy) or 1
        perp=(-dy/Lv*9,dx/Lv*9)
        pts=[a,(mid2[0]+perp[0],mid2[1]+perp[1]),
             (mid2[0]-perp[0],mid2[1]-perp[1]),b]
        pygame.draw.lines(surf,MBR_BUCKLE,False,[(int(q[0]),int(q[1])) for q in pts],2)

def draw_node(surf, pos, kind, hover=False, selected=False):
    x,y=pos; ring=SEL_COL if selected else (HOVER_COL if hover else WHITE)
    if kind=="anchor":
        s=13; pts=[(x,y-s),(x-s,y+s),(x+s,y+s)]
        pygame.draw.polygon(surf,ANCHOR_COL,pts)
        pygame.draw.polygon(surf,ring,pts,2)
        for i in range(-2,4):
            bx=x-s+i*6; pygame.draw.line(surf,ANCHOR_COL,(bx,y+s),(bx-6,y+s+7),2)
    else:
        pygame.draw.circle(surf,LOAD_COL,(x,y),10)
        pygame.draw.circle(surf,ring,(x,y),10,2)

def draw_load_arrow(surf, pos, angle_deg):
    x,y=pos; ar=math.radians(angle_deg)
    dx=math.cos(ar)*28; dy=math.sin(ar)*28
    sx,sy=x-int(dx),y-int(dy)
    pygame.draw.line(surf,LOAD_COL,(sx,sy),(x,y),3)
    perp_x=-dy/28*6; perp_y=dx/28*6
    pygame.draw.polygon(surf,LOAD_COL,[
        (x,y),
        (int(x-dx/28*10+perp_x),int(y-dy/28*10+perp_y)),
        (int(x-dx/28*10-perp_x),int(y-dy/28*10-perp_y)),
    ])

def draw_force_table(surf, fonts, members, forces, buckled_set, manual_indices, p,
                     sizing_map=None):
    if not forces: return
    font_sm,font_md,font_tiny=fonts
    rows=[]
    for i,(a,b) in enumerate(members):
        f=forces.get((a,b),forces.get((b,a),None))
        if f is None: continue
        L_px=math.dist(a,b); L_cm=L_px/p.pixels_per_m*100
        bk=(a,b) in buckled_set or (b,a) in buckled_set
        mn=i in manual_indices
        sz=None
        if sizing_map:
            sz=sizing_map.get((a,b),sizing_map.get((b,a),None))
        rows.append((abs(f),f,L_cm,bk,mn,i+1,sz))
    rows.sort(key=lambda r:r[0],reverse=True)

    PAD,ROW_H,BAR_W,COL_W=8,18,70,195
    pw=COL_W+BAR_W+PAD*3
    ph=min(PAD*2+22+len(rows)*ROW_H+22,HEIGHT-PANEL_H-20)
    px=WIDTH-pw-8; py=8

    bg=pygame.Surface((pw,ph),pygame.SRCALPHA); bg.fill((12,15,28,215))
    surf.blit(bg,(px,py))
    pygame.draw.rect(surf,(45,55,85),(px,py,pw,ph),1)
    surf.blit(font_sm.render("MEMBER FORCES",True,(175,192,222)),(px+PAD,py+PAD))
    pygame.draw.line(surf,(40,52,82),(px,py+22+PAD),(px+pw,py+22+PAD),1)

    max_abs=max(r[0] for r in rows) if rows else 1.0
    vis=(ph-22-PAD*2-22)//ROW_H
    for ri,(abs_f,f,L_cm,bk,mn,midx,sz) in enumerate(rows[:vis]):
        ry=py+24+PAD+ri*ROW_H
        if ri%2==0:
            rb=pygame.Surface((pw-2,ROW_H),pygame.SRCALPHA); rb.fill((255,255,255,7))
            surf.blit(rb,(px+1,ry))
        if bk:    tc,tcol="B!",MBR_BUCKLE
        elif f>0: tc,tcol="T", MBR_TENSION
        else:     tc,tcol="C", MBR_COMPRESS
        if mn and not bk: tcol=lerp_col(tcol,(180,170,255),0.35)
        surf.blit(font_tiny.render(f"#{midx:02d}",True,(110,130,160)),(px+PAD,ry+2))
        surf.blit(font_tiny.render(tc,True,tcol),(px+PAD+28,ry+2))
        surf.blit(font_tiny.render(f"{abs_f:6.1f}N",True,tcol),(px+PAD+42,ry+2))
        surf.blit(font_tiny.render(f"{L_cm:5.1f}cm",True,(95,115,150)),(px+PAD+105,ry+2))
        bx=px+COL_W+PAD; bm=BAR_W-4; bl=int(bm*abs_f/max_abs)
        pygame.draw.rect(surf,(28,36,56),(bx,ry+4,bm,ROW_H-8))
        if bl>0: pygame.draw.rect(surf,tcol,(bx,ry+4,bl,ROW_H-8))
        # Show sizing hint in table if available
        if sz:
            hint=f"r{sz['r_o_mm']:.1f}mm V{sz['volume_cm3']:.2f}"
            surf.blit(font_tiny.render(hint,True,(85,105,140)),(px+COL_W-60,ry+2))
    if len(rows)>vis:
        surf.blit(font_tiny.render(f"...{len(rows)-vis} more",True,DIM),(px+PAD,py+ph-16))
    fy=py+ph-14
    for txt,col,ox in [("T=tension",MBR_TENSION,PAD),
                       ("C=compress",MBR_COMPRESS,70),
                       ("B!=buckled",MBR_BUCKLE,138)]:
        surf.blit(font_tiny.render(txt,True,col),(px+ox,fy))

def draw_stress_legend(surf, font_sm, font_tiny):
    lx,ly=14,HEIGHT-PANEL_H-68
    surf.blit(font_sm.render("tension",True,MBR_TENSION),(lx,ly-16))
    surf.blit(font_sm.render("compression",True,MBR_COMPRESS),(lx+54,ly-16))
    for i in range(50):
        c=lerp_col((20,60,140),MBR_TENSION,i/49)
        pygame.draw.line(surf,c,(lx+i,ly),(lx+i,ly+8))
    for i in range(50):
        c=lerp_col((80,20,20),MBR_COMPRESS,i/49)
        pygame.draw.line(surf,c,(lx+50+i,ly),(lx+50+i,ly+8))
    pygame.draw.line(surf,MBR_BUCKLE,(lx,ly+18),(lx+100,ly+18),3)
    surf.blit(font_sm.render("buckled",True,MBR_BUCKLE),(lx+4,ly+23))

def draw_status_bar(surf, fonts, mode, status, solved, label_mode,
                    n_members, total_len, p, sel_load, load_angles):
    font_sm,font_md,font_tiny=fonts
    bar=pygame.Surface((WIDTH,PANEL_H)); bar.fill(PANEL_COL)
    surf.blit(bar,(0,HEIGHT-PANEL_H))
    pygame.draw.line(surf,DIVIDER,(0,HEIGHT-PANEL_H),(WIDTH,HEIGHT-PANEL_H),2)
    mc={"ANCHOR":ANCHOR_COL,"LOAD":LOAD_COL,"MANUAL":MBR_MANUAL}
    ml={"ANCHOR":"ANCHOR [A]","LOAD":"LOAD [L]","MANUAL":"MANUAL [M]"}
    len_str=(f"  |  {total_len/p.pixels_per_m*100:.1f}cm total"
             if total_len<float("inf") else "")
    mode_str=ml.get(mode,mode)
    if mode=="LOAD" and sel_load is not None:
        ang=load_angles.get(sel_load,DEFAULT_LOAD_ANGLE)%360
        mode_str+=f"  (selected: {ang:.0f}°)"
    lines=[
        (f"MODE: {mode_str}",mc.get(mode,WHITE),(14,HEIGHT-PANEL_H+7)),
        ("A  L  M  S=settings  SPACE=solve  F=labels  R=reset  C=clear  Q=quit",
         DIM,(14,HEIGHT-PANEL_H+30)),
        (status,HIGHLIGHT if solved else TEXT_COL,(14,HEIGHT-PANEL_H+53)),
        (f"Load={p.applied_load:.0f}N  E={p.E:.0f}GPa  "
         f"OD={p.outer_r_mm*2:.1f}mm  t={p.wall_mm:.1f}mm  "
         f"T_MAX={p.T_MAX:.1f}N  SF={p.safety_factor:.1f}"
         f"{len_str}  |  labels:{label_mode}",
         DIM,(14,HEIGHT-PANEL_H+76)),
    ]
    if n_members:
        lines.append((f"{n_members} members",WHITE,(WIDTH-140,HEIGHT-PANEL_H+7)))
    for txt,col,pos in lines:
        surf.blit(font_sm.render(txt,True,col),pos)

def draw_buckle_banner(surf, fonts, n_buckled, attempt):
    """Flashing orange banner shown during buckled-solution preview."""
    font_sm,font_md,font_tiny=fonts
    t=time.time()
    alpha=int(200+55*math.sin(t*8))
    banner=pygame.Surface((WIDTH,36),pygame.SRCALPHA)
    banner.fill((180,60,0,min(255,alpha)))
    surf.blit(banner,(0,0))
    msg=(f"⚠  BUCKLED SOLUTION (attempt {attempt}) — "
         f"{n_buckled} member(s) buckle  — searching for better topology…")
    s=font_md.render(msg,True,(255,220,100))
    surf.blit(s,(WIDTH//2-s.get_width()//2,8))

def nearest_node(pos, nodes, threshold=NODE_SNAP):
    best,best_d=None,threshold
    for n in nodes:
        d=math.dist(pos,n)
        if d<best_d: best,best_d=n,d
    return best


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    win=pygame.display.set_mode((WIDTH,HEIGHT))
    pygame.display.set_caption("AI Truss Builder — A* + Live Parameters  v2")
    clock=pygame.time.Clock()
    try:
        font_sm  =pygame.font.SysFont("Consolas",15)
        font_md  =pygame.font.SysFont("Consolas",17,bold=True)
        font_tiny=pygame.font.SysFont("Consolas",11)
    except Exception:
        font_sm  =pygame.font.SysFont(None,15)
        font_md  =pygame.font.SysFont(None,17)
        font_tiny=pygame.font.SysFont(None,11)
    fonts=(font_sm,font_md,font_tiny)

    p=Params(); settings=SettingsPanel()

    anchors=[]; loads=[]; load_angles={}
    members=[]; forces={}; buckled_set=set()
    manual_indices=set()
    member_sizing={}   # per-member minimum section after solve

    mode="ANCHOR"; status="Place anchors (A) and loads (L), then SPACE to solve."
    solved=False; sel_node=None; sel_load=None
    total_len=float("inf"); label_mode="both"

    # Buckle-preview state
    preview_buckled=False       # are we currently showing a buckled solution?
    preview_end_time=0.0        # when to stop showing it
    preview_members=[]
    preview_forces={}
    preview_buckled_set=set()
    buckle_attempt=0

    def get_fixed():
        return [members[i] for i in sorted(manual_indices)]

    def rotate_sel_load(delta_deg):
        nonlocal forces,buckled_set,solved,total_len,member_sizing
        if sel_load is None: return
        load_angles[sel_load]=(load_angles.get(sel_load,DEFAULT_LOAD_ANGLE)+delta_deg)
        forces.clear(); buckled_set.clear(); member_sizing.clear()
        solved=False; total_len=float("inf")

    def do_solve():
        """Run A* with post-solve buckling loop.  Updates global state."""
        nonlocal members,forces,buckled_set,manual_indices,solved,total_len
        nonlocal preview_buckled,preview_end_time,preview_members
        nonlocal preview_forces,preview_buckled_set,buckle_attempt,member_sizing
        nonlocal status

        if len(anchors)<1 or len(loads)<1:
            status="Need >=1 anchor and >=1 load."; return

        blacklist=set()
        attempt=0
        fixed=get_fixed()

        while True:
            attempt+=1
            # Show "Solving…" splash
            win.fill(BG)
            win.blit(font_md.render(
                f"A* Solving… (attempt {attempt})",True,HIGHLIGHT),
                (WIDTH//2-90,HEIGHT//2-20))
            win.blit(font_sm.render(p.summary(),True,DIM),
                     (WIDTH//2-220,HEIGHT//2+10))
            if blacklist:
                win.blit(font_sm.render(
                    f"Blacklisted {len(blacklist)} buckled topology(s)",True,(200,120,60)),
                    (WIDTH//2-160,HEIGHT//2+35))
            pygame.display.flip()
            pygame.event.pump()   # keep window responsive

            t_start = time.time()

            new_m, new_f, new_b, log, opt_len = run_astar(
                anchors, loads, noloads, p,
                load_angles, load_magnitudes,
                fixed_members=fixed, blacklist=blacklist)
            t_end = time.time()
            print(f"Attempt {attempt} | Solve time: {t_end - t_start:.2f}s | "
                    f"Members: {len(new_m)} | Nodes: {len(anchors)} anchors, "
                    f"{len(loads)} loads")

            for line in log: print(line)

            # Rebuild member list
            def apply_result(src_members,src_forces,src_buckled,src_len):
                nonlocal members,forces,buckled_set,manual_indices,total_len
                members.clear(); forces.clear(); buckled_set.clear()
                manual_indices.clear()
                for mm in fixed:
                    manual_indices.add(len(members)); members.append(mm)
                for m in src_members:
                    norm=tuple(sorted([m[0],m[1]]))
                    if not any(tuple(sorted([e[0],e[1]]))==norm for e in members):
                        members.append(m)
                forces={}
                for m,f in src_forces.items():
                    if m in members: forces[m]=f
                    elif (m[1],m[0]) in members: forces[(m[1],m[0])]=f
                buckled_set=set()
                for m in src_buckled:
                    if m in members: buckled_set.add(m)
                    elif (m[1],m[0]) in members: buckled_set.add((m[1],m[0]))
                total_len=src_len

            if not new_f:
                # No solution found at all
                apply_result(new_m,new_f,new_b,opt_len)
                solved=False; member_sizing={}
                status="No valid truss — adjust parameters or add intermediate nodes."
                return

            if not new_b:
                # Clean solution — done
                apply_result(new_m,new_f,new_b,opt_len)
                solved=True; member_sizing=compute_member_sizing(members,forces,p)
                max_f=max(abs(f) for f in new_f.values()) if new_f else 0
                pct=100*max_f/p.T_MAX
                total_vol=sum(sz["volume_cm3"] for sz in member_sizing.values())
                status=(f"OK  {opt_len/p.pixels_per_m*100:.1f}cm  "
                        f"{len(members)} members  peak {pct:.0f}% T_MAX  "
                        f"min-vol={total_vol:.3f}cm³")
                return
            else:
                # Buckled — show preview then continue
                buckle_attempt=attempt
                preview_members=list(new_m)
                preview_forces=dict(new_f)
                preview_buckled_set=set(new_b)
                preview_buckled=True
                preview_end_time=time.time()+BUCKLE_PREVIEW_SEC

                # Apply to main display while we wait
                apply_result(new_m,new_f,new_b,opt_len)
                solved=False
                n_b=len(new_b)
                status=(f"Attempt {attempt}: {n_b} member(s) buckle — "
                        f"showing for {BUCKLE_PREVIEW_SEC:.1f}s then re-searching…")

                # Render loop for the preview period
                t_start=time.time()
                while time.time()-t_start<BUCKLE_PREVIEW_SEC:
                    for ev in pygame.event.get():
                        if ev.type==pygame.QUIT: pygame.quit(); sys.exit()
                        if ev.type==pygame.KEYDOWN and ev.key in (pygame.K_q,pygame.K_ESCAPE):
                            pygame.quit(); sys.exit()
                    win.fill(BG)
                    draw_canvas_area(win,settings.open)
                    max_fv=max((abs(f) for f in preview_forces.values()),default=1.0)
                    for i,(a,b) in enumerate(members):
                        f2=preview_forces.get((a,b),preview_forces.get((b,a),None))
                        draw_member_line(win,a,b,force=f2,max_force=max_fv,
                                         manual=(i in manual_indices),
                                         buckled=((a,b) in preview_buckled_set or
                                                  (b,a) in preview_buckled_set),
                                         label_mode=label_mode,font_tiny=font_tiny,p=p)
                    for a in anchors: draw_node(win,a,"anchor")
                    for ld in loads:
                        draw_node(win,ld,"load")
                        draw_load_arrow(win,ld,load_angles.get(ld,DEFAULT_LOAD_ANGLE))
                    draw_buckle_banner(win,fonts,n_b,attempt)
                    draw_force_table(win,fonts,members,preview_forces,
                                     preview_buckled_set,manual_indices,p)
                    draw_stress_legend(win,font_sm,font_tiny)
                    draw_status_bar(win,fonts,mode,status,False,label_mode,
                                    len(members),opt_len,p,sel_load,load_angles)
                    pygame.display.flip()
                    clock.tick(60)

                preview_buckled=False
                # Blacklist this member topology
                blacklist.add(frozenset(tuple(sorted([m[0],m[1]])) for m in new_m))

                if attempt>=8:
                    status=(f"Gave up after {attempt} attempts — "
                            "all found topologies buckle. Try adjusting parameters.")
                    member_sizing={}
                    return

    running=True
    while running:
        mp=pygame.mouse.get_pos()
        all_nodes=list(set(anchors)|set(loads))
        hover_node=nearest_node(mp,all_nodes)

        win.fill(BG)
        draw_canvas_area(win,settings.open)

        for event in pygame.event.get():
            if event.type==pygame.QUIT: running=False

            if settings.open: settings.handle_event(event,p)

            if (event.type==pygame.MOUSEWHEEL
                    and mode=="LOAD" and sel_load is not None):
                rotate_sel_load(event.y*5.0)

            if event.type==pygame.KEYDOWN:
                k=event.key
                if k in (pygame.K_q,pygame.K_ESCAPE): running=False
                elif k==pygame.K_a: mode="ANCHOR"; sel_node=None; sel_load=None
                elif k==pygame.K_l: mode="LOAD";   sel_node=None
                elif k==pygame.K_m: mode="MANUAL"; sel_node=None; sel_load=None
                elif k==pygame.K_s: settings.toggle()
                elif k==pygame.K_f:
                    idx=LABEL_CYCLE.index(label_mode)
                    label_mode=LABEL_CYCLE[(idx+1)%len(LABEL_CYCLE)]
                    status=f"Labels: {label_mode.upper()}"
                elif k==pygame.K_c:
                    anchors.clear(); loads.clear(); load_angles.clear()
                    members.clear(); forces.clear()
                    buckled_set.clear(); manual_indices.clear()
                    member_sizing.clear()
                    sel_node=None; sel_load=None
                    solved=False; total_len=float("inf"); status="Cleared."
                elif k==pygame.K_r:
                    fixed=get_fixed()
                    members.clear(); forces.clear()
                    buckled_set.clear(); manual_indices.clear(); member_sizing.clear()
                    for mm in fixed: manual_indices.add(len(members)); members.append(mm)
                    solved=False; total_len=float("inf"); status="Auto members cleared."
                elif (mode=="LOAD" and sel_load is not None
                      and k in (pygame.K_LEFT,pygame.K_RIGHT,
                                pygame.K_UP,  pygame.K_DOWN)):
                    shift=bool(pygame.key.get_mods()&pygame.KMOD_SHIFT)
                    step=(15.0 if k in (pygame.K_UP,pygame.K_DOWN) else 1.0)
                    if shift: step*=10
                    if k in (pygame.K_LEFT,pygame.K_DOWN): step=-step
                    rotate_sel_load(step)
                    status=f"Load angle: {load_angles[sel_load]%360:.0f}°"
                elif k==pygame.K_SPACE:
                    do_solve()

            if event.type==pygame.MOUSEBUTTONDOWN and not settings.any_active():
                if (settings.open and mp[0]<SETTINGS_W
                        and settings._anim_x>-SETTINGS_W*0.5): pass
                elif mp[1]>=HEIGHT-PANEL_H: pass
                elif event.button==3:
                    target=nearest_node(mp,all_nodes)
                    if target:
                        if target in anchors: anchors.remove(target)
                        if target in loads:
                            loads.remove(target); load_angles.pop(target,None)
                            if sel_load==target: sel_load=None
                        new_m2,new_mm=[],set()
                        for i,m in enumerate(members):
                            if target not in m:
                                if i in manual_indices: new_mm.add(len(new_m2))
                                new_m2.append(m)
                        members[:]=new_m2; manual_indices.clear(); manual_indices.update(new_mm)
                        forces.clear(); buckled_set.clear(); member_sizing.clear()
                        solved=False; total_len=float("inf"); status="Node removed."
                elif event.button==1:
                    if mode=="ANCHOR":
                        if mp not in loads and mp not in anchors:
                            anchors.append(mp)
                            forces.clear(); buckled_set.clear(); member_sizing.clear()
                            solved=False; total_len=float("inf")
                    elif mode=="LOAD":
                        clicked=nearest_node(mp,loads)
                        if clicked:
                            sel_load=clicked
                            ang=load_angles.get(sel_load,DEFAULT_LOAD_ANGLE)%360
                            status=f"Load node selected  angle={ang:.0f}°  scroll/arrows to rotate"
                        else:
                            if mp not in anchors:
                                loads.append(mp); load_angles[mp]=DEFAULT_LOAD_ANGLE
                                sel_load=mp
                                forces.clear(); buckled_set.clear(); member_sizing.clear()
                                solved=False; total_len=float("inf")
                                status="Load placed  angle=90° (down)  scroll/arrows to rotate"
                    elif mode=="MANUAL":
                        clicked=nearest_node(mp,all_nodes)
                        if clicked:
                            if sel_node is None:
                                sel_node=clicked; status="Click second node to connect."
                            elif clicked!=sel_node:
                                pair=(sel_node,clicked); rev=(clicked,sel_node)
                                if pair not in members and rev not in members:
                                    members.append(pair); manual_indices.add(len(members)-1)
                                    forces.clear(); buckled_set.clear(); member_sizing.clear()
                                    solved=False; total_len=float("inf")
                                    status="Manual member added."
                                else: status="Already exists."
                                sel_node=None
                            else: sel_node=None; status="Deselected."

        # ── Draw members ───────────────────────────────────────────────────────
        max_f=max((abs(f) for f in forces.values()),default=1.0)
        for i,(a,b) in enumerate(members):
            f=forces.get((a,b),forces.get((b,a),None))
            sz=member_sizing.get((a,b),member_sizing.get((b,a),None))
            draw_member_line(win,a,b,force=f,max_force=max_f,
                             manual=(i in manual_indices),
                             buckled=((a,b) in buckled_set or (b,a) in buckled_set),
                             label_mode=label_mode,font_tiny=font_tiny,p=p,
                             sizing=sz)

        # Manual preview line
        if mode=="MANUAL" and sel_node:
            Lpx=math.dist(sel_node,mp)
            # Warn if exceeds the reference L_max, but don't block
            col=(255,180,0) if Lpx>p.L_MAX_PX else SEL_COL
            pygame.draw.line(win,col,sel_node,mp,1)
            warn=" (may buckle — A* will try)" if Lpx>p.L_MAX_PX else ""
            lbl=font_sm.render(f"{Lpx/p.pixels_per_m*100:.1f}cm{warn}",True,col)
            win.blit(lbl,(mp[0]+12,mp[1]-18))

        # ── Draw nodes ─────────────────────────────────────────────────────────
        for a in anchors:
            draw_node(win,a,"anchor",hover=(a==hover_node and mode=="MANUAL"),selected=(a==sel_node))
        for ld in loads:
            is_sel=(ld==sel_load)
            draw_node(win,ld,"load",hover=(ld==hover_node and mode in ("MANUAL","LOAD")),selected=is_sel)
            draw_load_arrow(win,ld,load_angles.get(ld,DEFAULT_LOAD_ANGLE))
            if is_sel and mode=="LOAD":
                draw_angle_dial(win,ld,load_angles.get(ld,DEFAULT_LOAD_ANGLE),font_tiny,font_sm)

        # ── Force table ────────────────────────────────────────────────────────
        draw_force_table(win,fonts,members,forces,buckled_set,manual_indices,p,
                         sizing_map=member_sizing if label_mode=="sizing" else None)

        # ── Volume summary (sizing mode) ───────────────────────────────────────
        if solved and member_sizing and label_mode=="sizing":
            total_vol=sum(sz["volume_cm3"] for sz in member_sizing.values())
            sv=font_sm.render(f"Total min material volume: {total_vol:.4f} cm³",True,(180,220,160))
            win.blit(sv,(14,8))

        # ── Colour legend ──────────────────────────────────────────────────────
        draw_stress_legend(win,font_sm,font_tiny)

        # ── Settings panel ─────────────────────────────────────────────────────
        settings.draw(win,fonts,p)

        # ── Status bar ─────────────────────────────────────────────────────────
        draw_status_bar(win,fonts,mode,status,solved, label_mode,len(members),total_len,p,sel_load,load_angles)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__=="__main__":
    main()
