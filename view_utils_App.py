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

from typing import Dict, Tuple, List, Optional, Iterable
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# -------------------------------
# Element classification
# -------------------------------

def classify_elements(nodes: Dict[int, Tuple[float, float, float]],
                      elements: Dict[int, Tuple[int, ...]]
                     ) -> Tuple[List[Tuple[int, Tuple[int, ...]]],
                                List[Tuple[int, Tuple[int, ...]]],
                                List[Tuple[int, Tuple[int, ...]]],
                                List[Tuple[int, Tuple[int, ...]]]]:
    """Classify 2-node elements into x_beams, y_beams, columns; 4-node as walls.
    Returns lists of (ele_tag, conn).
    """
    x_beams: List[Tuple[int, Tuple[int, ...]]] = []
    y_beams: List[Tuple[int, Tuple[int, ...]]] = []
    columns: List[Tuple[int, Tuple[int, ...]]] = []
    walls:   List[Tuple[int, Tuple[int, ...]]] = []

    for ele_tag, conn in elements.items():
        if not conn:
            continue
        if len(conn) == 2:
            n1, n2 = int(conn[0]), int(conn[1])
            p1, p2 = nodes.get(n1), nodes.get(n2)
            if p1 is None or p2 is None:
                continue
            dx, dy, dz = (p2[0] - p1[0]), (p2[1] - p1[1]), (p2[2] - p1[2])
            adx, ady, adz = abs(dx), abs(dy), abs(dz)
            # Column if vertical dominates
            if adz >= max(adx, ady) * 1.2:
                columns.append((int(ele_tag), (n1, n2)))
            else:
                if adx >= ady:
                    x_beams.append((int(ele_tag), (n1, n2)))
                else:
                    y_beams.append((int(ele_tag), (n1, n2)))
        elif len(conn) == 4:
            walls.append((int(ele_tag), tuple(int(n) for n in conn)))
        else:
            # Fallback: still draw it
            x_beams.append((int(ele_tag), tuple(int(n) for n in conn)))
    return x_beams, y_beams, columns, walls

# -------------------------------
# Colors & scales
# -------------------------------

def _default_colors() -> Dict[str, str]:
    return {
        'columns':   'rgb(20,80,200)',
        'x_beams':   'rgb(20,160,80)',
        'y_beams':   'rgb(200,120,20)',
        'walls':     'rgb(150,150,150)',  # solid wall gray
        'nodes':     'rgb(60,60,60)',
        'highlight': 'rgb(200,30,30)',
        'labels':    'black'
    }

def _colorscale() -> List[List[object]]:
    # Low→High
    return [
        [0.00, 'rgb(230,245,255)'],
        [0.15, 'rgb(179,205,227)'],
        [0.35, 'rgb(140,150,198)'],
        [0.55, 'rgb(136,86,167)'],
        [0.75, 'rgb(129,15,124)'],
        [1.00, 'rgb(77,0,75)'],
    ]

# -------------------------------
# Helpers
# -------------------------------

def _safe_get(d: Dict[int, Tuple[float, float, float]], key: int) -> Tuple[float, float, float]:
    v = d.get(int(key))
    if v is None:
        return (float('nan'),) * 3
    return (float(v[0]), float(v[1]), float(v[2]))

def _value_from_damage(ele_tag: int,
                       heatmap_mode: str,
                       damage_map: Optional[Dict[int, dict]],
                       damage_df: Optional[pd.DataFrame],
                       base_cost: float) -> Optional[float]:
    """Return the scalar value to color the element (ductility or cost)."""
    val: Optional[float] = None
    if heatmap_mode == 'ductility':
        if isinstance(damage_df, pd.DataFrame) and not damage_df.empty and 'Element' in damage_df.columns:
            row = damage_df.loc[damage_df['Element'] == ele_tag]
            if not row.empty:
                duct_cols = [c for c in damage_df.columns if c.startswith('Ductility')]
                if duct_cols:
                    try:
                        val = float(np.nanmax(row[duct_cols].astype(float).values))
                    except Exception:
                        val = None
        if val is None and damage_map and ele_tag in damage_map:
            dm = damage_map.get(ele_tag, {})
            for k in ('max_ductility','ductility','mu','max_mu'):
                if k in dm:
                    try:
                        val = float(dm[k])
                    except Exception:
                        pass
                    break
    else:  # 'cost'
        if isinstance(damage_df, pd.DataFrame) and not damage_df.empty and \
           ('Element' in damage_df.columns) and ('Estimated Cost ($)' in damage_df.columns):
            row = damage_df.loc[damage_df['Element'] == ele_tag]
            if not row.empty:
                try:
                    val = float(row['Estimated Cost ($)'].iloc[0])
                except Exception:
                    val = None
        if val is None and damage_map and ele_tag in damage_map:
            dm = damage_map.get(ele_tag, {})
            if any(isinstance(v, str) and v.lower() == 'damaged' for v in dm.values()):
                val = float(base_cost)
    return val

def _normalize(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    vmin = min(values); vmax = max(values)
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return (0.0, 1.0)
    if vmax <= vmin:
        return (vmin, vmin + 1.0)
    return (vmin, vmax)

def _map_value_to_color(val: float, vmin: float, vmax: float, cs: List[List[object]]) -> str:
    t = 0.0 if vmax <= vmin else (val - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, float(t)))
    return sample_colorscale(cs, t)[0]

def _add_wall_face(
    fig: go.Figure,
    pts: List[Tuple[float, float, float]],
    face_color: str,
    opacity: float = 1.0,
    legendgroup: str = 'walls',
    hovertext: Optional[str] = None
) -> None:
    """
    Add a filled quadrilateral face for a wall using two triangles.
    The quad is assumed to be ordered around the perimeter (0-1-2-3).
    """
    if len(pts) != 4:
        return
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    z = [p[2] for p in pts]

    # Triangulate as (0,1,2) and (0,2,3)
    i = [0, 0]
    j = [1, 2]
    k = [2, 3]

    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=face_color,          # constant solid gray
        opacity=opacity,           # fully opaque
        flatshading=True,
        lighting=dict(ambient=0.7, diffuse=0.9, specular=0.05, roughness=1.0),
        name='Wall',
        legendgroup=legendgroup,
        hoverinfo='text' if hovertext else 'skip',
        hovertemplate=(hovertext + '<extra></extra>') if hovertext else None,
        showscale=False
    ))

def _add_colorbar(fig: go.Figure, cs: List[List[object]], vmin: float, vmax: float, title_text: str) -> None:
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None], mode='markers',
        marker=dict(
            size=0.1,
            color=[vmin, vmax],
            colorscale=cs,
            cmin=vmin, cmax=vmax,
            showscale=True,
            colorbar=dict(
                title=dict(text=title_text, side='right')
            )
        ),
        showlegend=False, hoverinfo='none'
    ))

def _build_hover_for_element(
    etag: int,
    etype: str,
    damage_df: Optional[pd.DataFrame],
    per_ele_val: Dict[int, float],
    heatmap_mode: str
) -> str:
    lines = [f"<b>Element</b>: {etag}  <b>Type</b>: {etype}"]
    if isinstance(damage_df, pd.DataFrame) and not damage_df.empty and 'Element' in damage_df.columns:
        row = damage_df.loc[damage_df['Element'] == etag]
        if not row.empty:
            if 'Damage State' in row.columns:
                lines.append(f"<b>Damage</b>: {row['Damage State'].iloc[0]}")
            if 'Estimated Cost ($)' in row.columns:
                try:
                    cost = float(row['Estimated Cost ($)'].iloc[0])
                    lines.append(f"<b>Estimated Cost</b>: ${cost:,.0f}")
                except Exception:
                    pass
            duct_cols = [c for c in row.columns if c.startswith('Ductility')]
            if duct_cols:
                try:
                    vals = row[duct_cols].iloc[0].astype(float)
                    lines.append(f"<b>Max Ductility</b>: {np.nanmax(vals):.2f}")
                except Exception:
                    pass
    if etag in per_ele_val:
        label = 'Ductility' if heatmap_mode=='ductility' else 'Estimated Cost'
        val = per_ele_val[etag]
        if heatmap_mode=='ductility':
            lines.append(f"<b>{label}</b>: {val:.2f}")
        else:
            lines.append(f"<b>{label}</b>: ${val:,.0f}")
    return "<br>".join(lines)

# -------------------------------
# Main plot
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
    """Build a 3D Plotly scene for the current model.

    Behaviour driven by *options* keys (all optional):
        - 'element_range': (min_tag, max_tag) to filter
        - 'show_columns'/'show_beams_x'/'show_beams_y'/'show_walls' (bool)
        - 'show_nodes' (bool)
        - 'show_node_labels' (bool)
        - 'show_damage' (bool): if True, color members by heatmap_mode
    """
    options = options.copy() if isinstance(options, dict) else {}
    colors = _default_colors()
    cs = _colorscale()

    # Filter range
    tags = [int(t) for t in elements.keys()] or [0]
    rmin, rmax = (min(tags), max(tags))
    if 'element_range' in options and options['element_range']:
        try:
            rmin = max(rmin, int(options['element_range'][0]))
            rmax = min(rmax, int(options['element_range'][1]))
        except Exception:
            pass

    # Classify elements
    x_beams, y_beams, columns, walls = classify_elements(nodes, elements)

    # Prepare damage values if requested (used for non-wall members)
    show_damage = bool(options.get('show_damage', False))
    per_ele_val: Dict[int, float] = {}
    vmin, vmax = 0.0, 1.0
    if show_damage and (heatmap_mode in ('ductility','cost')):
        tmp_vals: List[float] = []
        for etag in elements.keys():
            et = int(etag)
            if not (rmin <= et <= rmax):
                continue
            val = _value_from_damage(et, heatmap_mode, damage_map, damage_df, base_cost)
            if val is not None and math.isfinite(val):
                per_ele_val[et] = float(val)
                tmp_vals.append(float(val))
        vmin, vmax = _normalize(tmp_vals)

    # Create figure
    fig = go.Figure()

    # Helper to draw 2-node element with optional heat color and hover
    def draw_member(p1, p2, color, width: int, name: str, group: str, etag: Optional[int]=None, etype: Optional[str]=None):
        line_color = color
        if show_damage and etag is not None and etag in per_ele_val:
            line_color = _map_value_to_color(per_ele_val[etag], vmin, vmax, cs)
        hovertext = None
        if etag is not None and etype is not None:
            hovertext = _build_hover_for_element(etag, etype, damage_df, per_ele_val, heatmap_mode)
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color=line_color, width=width),
            name=name, showlegend=False, legendgroup=group,
            hoverinfo='text' if hovertext else 'none',
            hovertemplate=(hovertext + '<extra></extra>') if hovertext else None
        ))

    # Columns
    if options.get('show_columns', True):
        for etag, conn in columns:
            if not (rmin <= etag <= rmax):
                continue
            p1, p2 = _safe_get(nodes, int(conn[0])), _safe_get(nodes, int(conn[1]))
            draw_member(p1, p2, colors['columns'], 5, 'Column', 'columns', etag=etag, etype='column')

    # X beams
    if options.get('show_beams_x', True):
        for etag, conn in x_beams:
            if not (rmin <= etag <= rmax):
                continue
            p1, p2 = _safe_get(nodes, int(conn[0])), _safe_get(nodes, int(conn[1]))
            draw_member(p1, p2, colors['x_beams'], 3, 'Beam-X', 'frames', etag=etag, etype='beam-x')

    # Y beams
    if options.get('show_beams_y', True):
        for etag, conn in y_beams:
            if not (rmin <= etag <= rmax):
                continue
            p1, p2 = _safe_get(nodes, int(conn[0])), _safe_get(nodes, int(conn[1]))
            draw_member(p1, p2, colors['y_beams'], 3, 'Beam-Y', 'frames', etag=etag, etype='beam-y')

    # Walls: draw as a single solid gray panel (fully opaque, no outlines)
    if options.get('show_walls', True):
        for etag, conn in walls:
            if not (rmin <= etag <= rmax):
                continue
            pts = [_safe_get(nodes, int(n)) for n in conn]
            hover = f"<b>Element</b>: {etag}  <b>Type</b>: wall"
            _add_wall_face(
                fig, pts,
                face_color=colors['walls'],
                opacity=1.0,                   # completely solid
                legendgroup='walls',
                hovertext=hover
            )

    # Node cloud
    if options.get('show_nodes', True):
        nodes_in_range: set[int] = set()
        for etag, conn in elements.items():
            et = int(etag)
            if not (rmin <= et <= rmax):
                continue
            for n in conn:
                nodes_in_range.add(int(n))
        if highlight_nodes_list:
            for n in highlight_nodes_list:
                nodes_in_range.add(int(n))
        if not elements:
            nodes_in_range = set(int(n) for n in nodes.keys())

        xs, ys, zs = [], [], []
        for nid in nodes_in_range:
            x, y, z = _safe_get(nodes, nid)
            xs.append(x); ys.append(y); zs.append(z)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers',
            marker=dict(size=2, color=_default_colors()['nodes']),
            name='Nodes', showlegend=False, hoverinfo='skip'
        ))

        # Highlighted nodes
        if highlight_nodes_list:
            hx, hy, hz = [], [], []
            for nid in highlight_nodes_list:
                x, y, z = _safe_get(nodes, int(nid))
                hx.append(x); hy.append(y); hz.append(z)
            fig.add_trace(go.Scatter3d(
                x=hx, y=hy, z=hz, mode='markers',
                marker=dict(size=6, color=_default_colors()['highlight']),
                name='Highlighted Nodes', showlegend=True, hoverinfo='skip'
            ))

        # Node labels
        if options.get('show_node_labels', False) and node_labels_dict:
            lx, ly, lz, ltxt = [], [], [], []
            for nid in nodes_in_range:
                if nid in node_labels_dict:
                    x, y, z = _safe_get(nodes, int(nid))
                    lx.append(x); ly.append(y); lz.append(z); ltxt.append(str(node_labels_dict[nid]))
            if ltxt:
                fig.add_trace(go.Scatter3d(
                    x=lx, y=ly, z=lz, mode='text',
                    text=ltxt, textposition='top center',
                    textfont=dict(color='black', size=10),
                    showlegend=False, hoverinfo='skip'
                ))

    # Colorbar for damage view (from beams/columns only; walls stay solid gray)
    if show_damage:
        # If there are values from any non-wall members, show the colorbar
        non_wall_vals = [v for et, v in per_ele_val.items() if et not in {w[0] for w in walls}]
        if non_wall_vals:
            vmin2, vmax2 = _normalize(non_wall_vals)
            cb_title = 'Ductility Demand' if heatmap_mode == 'ductility' else 'Estimated Cost ($)'
            _add_colorbar(fig, cs, vmin2, vmax2, cb_title)

    # Layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(title='X', showgrid=True, zeroline=False),
            yaxis=dict(title='Y', showgrid=True, zeroline=False),
            zaxis=dict(title='Z', showgrid=True, zeroline=False),
            aspectmode='data'
        ),
        showlegend=False
    )
    return fig
