# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 15:13:55 2025

@author: jgomez
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openseespy.opensees import getNodeTags, nodeCoord, getEleTags, eleNodes

def order_rectangle_points(pts):
    """Helper function to order vertices of a 4-node element for plotting."""
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    if np.linalg.norm(np.cross(v1, v2)) < 1e-8:
        indices = np.lexsort((pts[:,2], pts[:,1], pts[:,0]))
        return [tuple(p) for p in pts[indices]]

    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    def project(p):
        vec = p - center
        u = np.dot(vec, v1)
        v = np.dot(vec, np.cross(normal, v1))
        return np.array([u, v])

    projected = np.array([project(p) for p in pts])
    angles = np.arctan2(projected[:,1], projected[:,0])
    order = np.argsort(angles)

    return [tuple(pts[i]) for i in order]

def classify_elements(nodes, elements):
    """Classifies elements into beams, columns, and walls based on geometry."""
    x_beams, y_beams, columns, walls = [], [], [], []
    for ele_tag, conn in elements.items(): 
        if not all(n in nodes for n in conn):
            continue
        pts = [np.array(nodes[n]) for n in conn]
        if len(conn) == 2:
            delta = pts[1] - pts[0]
            dx, dy, dz = np.abs(delta)
            if dz > max(dx, dy) * 0.95:
                columns.append((ele_tag, conn))
            elif dx >= dy:
                x_beams.append((ele_tag, conn))
            else:
                y_beams.append((ele_tag, conn))
        elif len(conn) == 4:
            walls.append((ele_tag, conn))
    return x_beams, y_beams, columns, walls

def add_legend_items(fig, damage_map=None, heatmap_mode='ductility', base_cost=15000):
    """
    Adds dummy traces for the legend and the appropriate color bar based on the view mode.
    """
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                              line=dict(color='#cccccc', width=5), name='Frame (Elastic)'))
    fig.add_trace(go.Mesh3d(x=[0], y=[0], z=[0], i=[0], j=[0], k=[0],
                           color='lightblue', opacity=0.4, name='Wall (Elastic)'))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                              marker=dict(size=12, color='cyan', symbol='diamond', line=dict(color='black', width=2)),
                              name='Instrumented Node'))
    # Restored "Structural Node" to the legend
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                              marker=dict(size=4, color='black', opacity=0.8),
                              name='Structural Node'))

    if damage_map:
        marker_colors = ['#fee090', '#fdae61', '#d73027'] # Yellow, Orange, Red
        custom_colorscale = [[0.0, marker_colors[0]], [0.5, marker_colors[1]], [1.0, marker_colors[2]]]
        
        if heatmap_mode == 'ductility':
            title_text, tickvals, ticktext = "Ductility Demand", [1, 2, 4], ["1: Yield", "2: Moderate", "4: High"]
            cmin, cmax = 0, 5
        else: # Cost mode
            title_text = "Repair Cost ($)"
            cmin, cmax = 0, base_cost
            tickvals = [0.15 * base_cost, 0.5 * base_cost, 1.0 * base_cost]
            ticktext = [f"${v:,.0f}" for v in tickvals]
            
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode='markers',
            marker=dict(
                colorscale=custom_colorscale, cmin=cmin, cmax=cmax, showscale=True,
                colorbar=dict(
                    title=dict(text=title_text, side="right"),
                    tickmode="array", tickvals=tickvals, ticktext=ticktext, ticks="outside"
                )
            ),
            hoverinfo='none', showlegend=False
        ))


def create_interactive_plot(nodes, elements, damage_map=None, damage_df=None, options=None, highlight_nodes_list=None, node_labels_dict=None, heatmap_mode='ductility', base_cost=15000):
    """
    Creates an interactive 3D Plotly figure with a dynamic heatmap for ductility or cost.
    (Handles node and element label toggling)
    """
    if options is None: options = {}
    if damage_map is None: damage_map = {}
    if highlight_nodes_list is None: highlight_nodes_list = []
    if node_labels_dict is None: node_labels_dict = {}
        
    fig = go.Figure()
    
    # --- NEW: Initialize a list for scene annotations ---
    scene_annotations = []

    if not nodes:
        fig.add_annotation(x=0.5, y=0.5, text="Model not built yet.", showarrow=False, xref="paper", yref="paper", font=dict(size=20, color="gray"))
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), height=600, width=800)
        return fig

    damage_df_indexed, ductility_cols = None, []
    if damage_df is not None and not damage_df.empty and 'Element' in damage_df.columns:
        damage_df_indexed = damage_df.set_index('Element')
        ductility_cols = [c for c in damage_df.columns if 'Ductility' in c]

    x_beams, y_beams, columns, walls = classify_elements(nodes, elements)
    min_ele, max_ele = options.get('element_range', (min(elements.keys(), default=0), max(elements.keys(), default=1)))
    
    node_x, node_y, node_z, node_labels = [], [], [], []
    for tag, coords in nodes.items():
        node_x.append(coords[0]); node_y.append(coords[1]); node_z.append(coords[2])
        node_labels.append(f"Node: {tag}<br>Coords: ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})")
        # --- NEW: Logic to add node labels if toggled on ---
        if options.get('show_node_labels', False):
            scene_annotations.append(dict(x=coords[0], y=coords[1], z=coords[2],
                                          text=str(tag), showarrow=False,
                                          font=dict(color='black', size=10),
                                          bgcolor='rgba(255, 255, 255, 0.6)'))

    fig.add_trace(go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers',
        marker=dict(size=4, color='black', opacity=0.8), hoverinfo='text', text=node_labels, name='Nodes', showlegend=False))

    all_elements = [('frame', item) for item in (x_beams + y_beams + columns)] + [('wall', item) for item in walls]

    for ele_type, (ele_tag, conn) in all_elements:
        if not (min_ele <= ele_tag <= max_ele): continue
        if (ele_type == 'frame' and not options.get('show_frames', True)) or \
           (ele_type == 'wall' and not options.get('show_walls', True)):
            continue

        pts = np.array([nodes[n] for n in conn])
        is_damaged = options.get('show_damage', True) and ele_tag in damage_map
        
        # --- NEW: Logic to add element labels if toggled on ---
        if options.get('show_element_labels', False):
            center_pt = np.mean(pts, axis=0)
            scene_annotations.append(dict(x=center_pt[0], y=center_pt[1], z=center_pt[2],
                                          text=str(ele_tag), showarrow=False,
                                          font=dict(color='purple', size=10, family="Arial Black"),
                                          bgcolor='rgba(255, 255, 255, 0.7)'))

        color = '#cccccc' if ele_type == 'frame' else 'lightblue'
        line_width, opacity = (5, 1.0) if ele_type == 'frame' else (None, 0.4)
        max_ductility, cost, rcf = 0, 0, 0

        if is_damaged and damage_df_indexed is not None and ele_tag in damage_df_indexed.index:
            line_width, opacity = (8, 1.0) if ele_type == 'frame' else (None, 0.7)
            if ductility_cols:
                max_ductility = damage_df_indexed.loc[ele_tag][ductility_cols].max()
            if 'Estimated Cost ($)' in damage_df_indexed.columns:
                cost = damage_df_indexed.loc[ele_tag]['Estimated Cost ($)']
                rcf = damage_df_indexed.loc[ele_tag]['Repair Cost Factor']

            value_for_coloring = max_ductility if heatmap_mode == 'ductility' else rcf
            if value_for_coloring >= (4.0 if heatmap_mode == 'ductility' else 1.0): color = '#d73027'
            elif value_for_coloring >= (2.0 if heatmap_mode == 'ductility' else 0.5): color = '#fdae61'
            elif value_for_coloring >= (1.0 if heatmap_mode == 'ductility' else 0.15): color = '#fee090'

        duct_text = f"<b>{max_ductility:.2f}</b>" if heatmap_mode == 'ductility' else f"{max_ductility:.2f}"
        cost_text = f"<b>${cost:,.0f}</b>" if heatmap_mode == 'cost' else f"${cost:,.0f}"
        hover_text = f"Element: {ele_tag}<br>Type: {ele_type.capitalize()}<br>Max Ductility: {duct_text}<br>Est. Cost: {cost_text}"

        if ele_type == 'frame':
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='lines',
                line=dict(color=color, width=line_width), hoverinfo='text', text=hover_text, showlegend=False))
        else: # Wall
            ordered_pts = order_rectangle_points(pts)
            x_wall, y_wall, z_wall = [p[0] for p in ordered_pts], [p[1] for p in ordered_pts], [p[2] for p in ordered_pts]
            fig.add_trace(go.Mesh3d(x=x_wall, y=y_wall, z=z_wall, i=[0, 0], j=[1, 2], k=[2, 3],
                opacity=opacity, color=color, hoverinfo='text', text=hover_text, showlegend=False))

    # --- Highlight specific nodes (e.g., instrumented nodes) ---
    if highlight_nodes_list:
        h_x, h_y, h_z, h_text = [], [], [], []
        for node_id in highlight_nodes_list:
            if node_id in nodes:
                coords = nodes[node_id]
                h_x.append(coords[0])
                h_y.append(coords[1])
                h_z.append(coords[2])
                label = node_labels_dict.get(node_id, f"Node: {node_id}")
                h_text.append(f"{label}<br>Coords: ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})")

        if h_x:
            fig.add_trace(go.Scatter3d(
                x=h_x, y=h_y, z=h_z,
                mode='markers',
                marker=dict(size=12, color='cyan', symbol='diamond', line=dict(color='black', width=2)),
                hoverinfo='text', text=h_text,
                name='Instrumented Nodes', showlegend=False
            ))

    # --- Add Legend and Color Bar ---
    add_legend_items(fig, damage_map=damage_map, heatmap_mode=heatmap_mode, base_cost=base_cost)

    # --- Layout and Camera Settings ---
    all_coords = np.array(list(nodes.values()))
    if all_coords.size > 0:
        max_range_dim = max(np.ptp(all_coords[:, i]) for i in range(3))
        mid_pt = np.mean(all_coords, axis=0)
        scene_limits = {f'{ax}axis': dict(range=[mid_pt[i] - max_range_dim / 2 * 1.1, mid_pt[i] + max_range_dim / 2 * 1.1], title=f'{ax.upper()} [m]') for i, ax in enumerate('xyz')}
    else:
        scene_limits = {f'{ax}axis': dict(title=f'{ax.upper()} [m]') for ax in 'xyz'}

    view = options.get('view')
    if view == 'Plan View': camera_eye = dict(x=0, y=0, z=2.5)
    elif view == 'Front View': camera_eye = dict(x=0, y=-2.5, z=0)
    elif view == 'Side View': camera_eye = dict(x=2.5, y=0, z=0)
    else: camera_eye = dict(x=1.5, y=-1.5, z=1)

    fig.update_layout(
        scene=scene_limits, scene_aspectmode='cube', height=700, margin=dict(l=0, r=0, b=0, t=0),
        hovermode='closest', scene_camera=dict(eye=camera_eye), showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.7)'))

    # --- NEW: Add the collected annotations to the layout ---
    if scene_annotations:
        fig.update_layout(scene_annotations=scene_annotations)

    if not options.get('show_grid', True):
        fig.update_layout(scene={f'{ax}axis_visible': False for ax in 'xyz'})

    return fig