# -*- coding: utf-8 -*-
"""
Lightweight 3D viewer helpers for PullOver_App

Public API (used by the app):
- classify_elements(nodes, elements) -> (x_beams, y_beams, columns, walls)
- create_interactive_plot(
      nodes, elements, damage_map=None, damage_df=None, options=None,
      highlight_nodes_list=None, node_labels_dict=None,
      heatmap_mode='ductility', base_cost=15000.0
  )
"""

from __future__ import annotations
import math
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -------------------------------
# Utilities
# -------------------------------

def _vec(p, q):
    return (q[0]-p[0], q[1]-p[1], q[2]-p[2])

def _norm(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def _safe_get(nodes: Dict[int, Tuple[float,float,float]], tag: int):
    xyz = nodes.get(int(tag))
    if xyz is None:
        return (0.0, 0.0, 0.0)
    return tuple(float(c) for c in xyz)

# -------------------------------
# Element classification
# -------------------------------

def classify_elements(nodes: Dict[int, Tuple[float,float,float]],
                      elements: Dict[int, Tuple[int, ...]]
                     ) -> Tuple[List[Tuple[int, Tuple[int,...]]],
                                List[Tuple[int, Tuple[int,...]]],
                                List[Tuple[int, Tuple[int,...]]],
                                List[Tuple[int, Tuple[int,...]]]]:
    """
    Classify 2-node elements into x_beams, y_beams, columns; 4-node as walls.
    Returns lists of (ele_tag, conn).
    """
    x_beams, y_beams, columns, walls = [], [], [], []
    for ele_tag, conn in elements.items():
        if not conn:
            continue
        if len(conn) == 2:
            n1, n2 = int(conn[0]), int(conn[1])
            p1, p2 = _safe_get(nodes, n1), _safe_get(nodes, n2)
            dx, dy, dz = abs(p2[0]-p1[0]), abs(p2[1]-p1[1]), abs(p2[2]-p1[2])
            # Heuristic: vertical if Z span dominates
            if dz > max(dx, dy)*0.9:
                columns.append((ele_tag, conn))
            elif dx >= dy:
                x_beams.append((ele_tag, conn))
            else:
                y_beams.append((ele_tag, conn))
        elif len(conn) == 4:
            walls.append((ele_tag, conn))
        else:
            # default bucket to still render unusual topologies
            x_beams.append((ele_tag, conn))
    return x_beams, y_beams, columns, walls

# -------------------------------
# Colors & scales
# -------------------------------

def _default_colors():
    return {
        'columns': 'rgb(20,80,200)',
        'x_beams': 'rgb(20,160,80)',
        'y_beams': 'rgb(200,120,20)',
        'walls':   'rgb(120,120,120)',
        'nodes':   'rgb(60,60,60)',
        'highlight': 'rgb(200, 30, 30)',
        'labels': 'black'
    }

def _colorscale():
    # Perceptual-ish colorscale for damage (low to high)
    return [
        [0.0,  "rgb(230, 245, 255)"],
        [0.25, "rgb(169, 217, 255)"],
        [0.5,  "rgb(115, 179, 216)"],
        [0.75, "rgb(252, 141, 89)"],
        [1.0,  "rgb(215, 48, 39)"]
    ]

# -------------------------------
# Damage helpers
# -------------------------------

def _per_element_metric(ele_tag: int,
                        damage_map: Optional[Dict[int, dict]],
                        damage_df: Optional[pd.DataFrame],
                        heatmap_mode: str = 'ductility',
                        base_cost: float = 15000.0) -> Optional[float]:
    """
    Returns a numeric value per element for coloring.
    - heatmap_mode='ductility': use max ductility across axes/hinges (from damage_map if available).
    - heatmap_mode='cost': use 'Estimated Cost ($)' from damage_df; fallback to base_cost for "damaged".
    """
    if damage_map is None and damage_df is None:
        return None

    val = None
    if heatmap_mode == 'ductility':
        if damage_map and ele_tag in damage_map:
            # look for any key containing 'ductility'
            d = damage_map[ele_tag]
            duct_vals = []
            for k, v in d.items():
                if isinstance(k, str) and 'ductility' in k.lower():
                    try:
                        duct_vals.append(float(v))
                    except Exception:
                        pass
            if duct_vals:
                val = max(duct_vals)
    else:  # cost
        if isinstance(damage_df, pd.DataFrame) and not damage_df.empty and 'Estimated Cost ($)' in damage_df.columns and 'Element' in damage_df.columns:
            try:
                row = damage_df.loc[damage_df['Element'] == ele_tag]
                if not row.empty:
                    val = float(row['Estimated Cost ($)'].iloc[0])
            except Exception:
                val = None
        if val is None and damage_map and ele_tag in damage_map:
            # fallback: assign base_cost if marked damaged
            val = float(base_cost)

    return val

def _normalize(values: List[float]) -> Tuple[List[float], float, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return [], 0.0, 1.0
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1.0
    normed = ((arr - vmin) / (vmax - vmin)).tolist()
    return normed, vmin, vmax

# -------------------------------
# Plot construction
# -------------------------------

def _add_line_element(fig, p1, p2, color, width=4, name=None, showlegend=False, legendgroup=None):
    fig.add_trace(go.Scatter3d(
        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
        mode='lines',
        line=dict(color=color, width=width),
        hoverinfo='none',
        name=name,
        showlegend=showlegend,
        legendgroup=legendgroup
    ))

def _add_wall_quad(fig, pts4, color, name=None, opacity=0.35, showlegend=False, legendgroup=None):
    # pts4 must be in order (close to rectangular)
    x = [p[0] for p in pts4]
    y = [p[1] for p in pts4]
    z = [p[2] for p in pts4]
    # Two triangles: (0,1,2) and (0,2,3)
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=color, opacity=0.35, name=name,
        showlegend=showlegend, legendgroup=legendgroup
    ))

def _view_camera(view: str):
    if view == 'Plan View':
        return dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0), center=dict(x=0,y=0,z=0))
    if view == 'Front View':  # +Y
        return dict(eye=dict(x=0, y=2.5, z=0.01), up=dict(x=0, y=0, z=1), center=dict(x=0,y=0,z=0))
    if view == 'Side View':   # +X
        return dict(eye=dict(x=2.5, y=0, z=0.01), up=dict(x=0, y=0, z=1), center=dict(x=0,y=0,z=0))
    return dict(eye=dict(x=1.6, y=1.6, z=1.2))

def _axes_limits(nodes: Dict[int, Tuple[float,float,float]]):
    if not nodes:
        return dict(x=[0,1], y=[0,1], z=[0,1])
    xs = [float(x) for (x,_,_) in nodes.values()]
    ys = [float(y) for (_,y,_) in nodes.values()]
    zs = [float(z) for (*_,z) in nodes.values()]
    pad_x = max(1e-6, 0.05*(max(xs)-min(xs) if xs else 1.0))
    pad_y = max(1e-6, 0.05*(max(ys)-min(ys) if ys else 1.0))
    pad_z = max(1e-6, 0.05*(max(zs)-min(zs) if zs else 1.0))
    return dict(
        x=[min(xs)-pad_x, max(xs)+pad_x],
        y=[min(ys)-pad_y, max(ys)+pad_y],
        z=[min(zs)-pad_z, max(zs)+pad_z]
    )

# -------------------------------
# Public plotting API
# -------------------------------

def create_interactive_plot(
    nodes: Dict[int, Tuple[float,float,float]],
    elements: Dict[int, Tuple[int, ...]],
    damage_map: Optional[Dict[int, dict]] = None,
    damage_df: Optional[pd.DataFrame] = None,
    options: Optional[dict] = None,
    highlight_nodes_list: Optional[Iterable[int]] = None,
    node_labels_dict: Optional[Dict[int, str]] = None,
    heatmap_mode: str = 'ductility',
    base_cost: float = 15000.0
):
    """
    Build a 3D plotly scene for the current model.
    - Filters elements by options['element_range'] if present.
    - Colors by damage if options['show_damage'] and damage data provided.
    - Draws node cloud by default (options['show_nodes'] defaults to True).
    """
    options = options.copy() if options else {}
    colors = _default_colors()

    # Defensive clamp of element range
    ele_tags = sorted([int(t) for t in elements.keys()]) if elements else []
    if ele_tags:
        emin, emax = min(ele_tags), max(ele_tags)
    else:
        emin, emax = 0, 1
    opt_range = options.get('element_range', (emin, emax))
    try:
        rmin, rmax = int(opt_range[0]), int(opt_range[1])
    except Exception:
        rmin, rmax = emin, emax
    rmin, rmax = max(emin, rmin), min(emax, rmax)
    options['element_range'] = (rmin, rmax)

    # Prepare damage values if any
    per_ele_vals = {}
    if options.get('show_damage', False) and (damage_map is not None or damage_df is not None):
        vals = []
        for et in ele_tags:
            if rmin <= et <= rmax:
                v = _per_element_metric(et, damage_map, damage_df, heatmap_mode, base_cost)
                if v is not None:
                    per_ele_vals[et] = float(v)
                    vals.append(float(v))
        normed, vmin, vmax = _normalize(vals)
        for et in list(per_ele_vals.keys()):
            per_ele_vals[et] = (per_ele_vals[et] - vmin) / (vmax - vmin) if vmax != vmin else 0.0
        cmin, cmax = vmin, vmax
    else:
        cmin = cmax = None

    # Build figure
    fig = go.Figure()

    # Classify elements
    x_beams, y_beams, columns, walls = classify_elements(nodes, elements)

    # Base structure: frames
    if options.get('show_frames', True):
        for ele_tag, conn in columns:
            if not (rmin <= ele_tag <= rmax):
                continue
            p1, p2 = _safe_get(nodes, conn[0]), _safe_get(nodes, conn[1])
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                mode='lines', line=dict(color=colors['columns'], width=4),
                hoverinfo='none', showlegend=False, legendgroup='frames'))
        for ele_tag, conn in x_beams:
            if not (rmin <= ele_tag <= rmax):
                continue
            p1, p2 = _safe_get(nodes, conn[0]), _safe_get(nodes, conn[1])
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                mode='lines', line=dict(color=colors['x_beams'], width=3),
                hoverinfo='none', showlegend=False, legendgroup='frames'))
        for ele_tag, conn in y_beams:
            if not (rmin <= ele_tag <= rmax):
                continue
            p1, p2 = _safe_get(nodes, conn[0]), _safe_get(nodes, conn[1])
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                mode='lines', line=dict(color=colors['y_beams'], width=3),
                hoverinfo='none', showlegend=False, legendgroup='frames'))

    # Walls
    if options.get('show_walls', True):
        for ele_tag, conn in walls:
            if not (rmin <= ele_tag <= rmax):
                continue
            pts = [_safe_get(nodes, int(n)) for n in conn]
            _add_wall_quad(fig, pts, colors['walls'], legendgroup='walls')

    # --- Node cloud (by default) ---
    show_nodes = options.get('show_nodes', True)
    if show_nodes:
        # nodes to plot: those connected to any in-range element, plus highlighted nodes
        nodes_in_range: set[int] = set()
        for ele_tag, conn in elements.items():
            if not (rmin <= int(ele_tag) <= rmax):
                continue
            for n in conn:
                nodes_in_range.add(int(n))

        if highlight_nodes_list:
            for n in highlight_nodes_list:
                nodes_in_range.add(int(n))

        if not elements:
            # if no elements, plot all nodes
            nodes_in_range = set(int(n) for n in nodes.keys())

        if nodes_in_range:
            xs, ys, zs = [], [], []
            for nid in sorted(nodes_in_range):
                p = _safe_get(nodes, nid)
                xs.append(p[0]); ys.append(p[1]); zs.append(p[2])

            # Label policy: avoid clutter on very large models
            want_labels = bool(options.get('show_node_labels', False))
            add_labels_for_all = want_labels and (len(nodes_in_range) <= 600)

            text = None
            mode = 'markers'
            if want_labels:
                if add_labels_for_all:
                    text = [str(nid) for nid in sorted(nodes_in_range)]
                    mode = 'markers+text'
                elif highlight_nodes_list:
                    # label only highlighted nodes when too many
                    text_map = {int(n): str(int(n)) for n in highlight_nodes_list}
                    # We'll overlay labels via the highlight trace below
                    pass

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode=mode,
                marker=dict(size=3, color=colors['nodes']),
                text=text,
                textposition="top center",
                name="Nodes",
                showlegend=True
            ))

    # Overlay damaged elements (thick, colored)
    if options.get('show_damage', False) and per_ele_vals:
        cs = _colorscale()
        # Dummy trace for colorbar
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode='markers',
            marker=dict(
                size=7, color=[0,1], colorscale=cs,
                cmin=0, cmax=1,
                colorbar=dict(
                    title='Damage' if heatmap_mode=='ductility' else 'Cost',
                    titleside='right'
                )
            ),
            showlegend=False, hoverinfo='none'
        ))
        for ele_tag, conn in [*columns, *x_beams, *y_beams]:
            if not (rmin <= ele_tag <= rmax):
                continue
            val = per_ele_vals.get(ele_tag, None)
            if val is None:
                continue
            p1, p2 = _safe_get(nodes, conn[0]), _safe_get(nodes, conn[1])
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                mode='lines',
                line=dict(width=8, color=val, colorscale=cs, cmin=0, cmax=1),
                name=f"Ele {ele_tag}",
                showlegend=False,
                hovertemplate=f"Ele {ele_tag}<br>value={val:.3f}<extra></extra>"
            ))

    # Highlights: nodes (bigger markers, optional labels)
    if highlight_nodes_list:
        xs, ys, zs, texts = [], [], [], []
        for nid in highlight_nodes_list:
            p = _safe_get(nodes, int(nid))
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
            # If labels dict provided, use it; else use node id
            if node_labels_dict and int(nid) in node_labels_dict:
                texts.append(node_labels_dict[int(nid)])
            else:
                texts.append(str(int(nid)))
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers+text' if options.get('show_node_labels', False) else 'markers',
            marker=dict(size=7, color=_default_colors()['highlight']),
            text=texts if options.get('show_node_labels', False) else None,
            textposition="top center",
            name="Highlighted Nodes",
            showlegend=True
        ))

    # Optional element labels at midpoints (only for 2-node elements)
    if options.get('show_element_labels', False):
        lx, ly, lz, ltext = [], [], [], []
        for ele_tag, conn in elements.items():
            if not (rmin <= ele_tag <= rmax):
                continue
            if len(conn) >= 2:
                p1, p2 = _safe_get(nodes, int(conn[0])), _safe_get(nodes, int(conn[1]))
                mid = ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0, (p1[2]+p2[2])/2.0)
                lx.append(mid[0]); ly.append(mid[1]); lz.append(mid[2]); ltext.append(str(ele_tag))
        if lx:
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz, mode='text', text=ltext, textposition="top center",
                name="Element Labels", showlegend=False
            ))

    # Axes & camera
    limits = _axes_limits(nodes)
    camera = _view_camera(options.get('view', 'Isometric'))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=limits['x'], title='X'),
            yaxis=dict(range=limits['y'], title='Y'),
            zaxis=dict(range=limits['z'], title='Z'),
            aspectmode='data'
        ),
        scene_camera=camera,
        height=700,
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.7)')
    )

    if not options.get('show_grid', True):
        fig.update_scenes(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))

    return fig
