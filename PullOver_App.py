# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 22:55:10 2025

@author: jgomez (Refactored by Gemini)
"""

import streamlit as st
import os
import importlib.util 
import pandas as pd
from openseespy.opensees import wipe, getNodeTags, nodeCoord, getEleTags, eleNodes
import numpy as np
import plotly.graph_objects as go
import yaml 
import zipfile 
import tempfile 
import sys 
import shutil

# --- 1. PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Cloud-Based Damage Detection System (General)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import the adapted helper modules
import damage_detector_App as dam 
import view_utils_App as vu      

# --- 2. HELPER FUNCTIONS ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV file for downloading."""
    return df.to_csv(index=False).encode('utf-8')

def load_module_from_path(module_name, file_path):
    """Dynamically loads a module from a given file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# --- 3. INITIALIZE SESSION STATE ---
if "project_path" not in st.session_state:
    st.session_state.project_path = None
if "project_config" not in st.session_state:
    st.session_state.project_config = {}
if "model_built" not in st.session_state:
    st.session_state.model_built = False
if "damage_detected" not in st.session_state:
    st.session_state.damage_detected = False 
if "nodes" not in st.session_state:
    st.session_state.nodes = {}
if "elements" not in st.session_state:
    st.session_state.elements = {}
if "model_summary" not in st.session_state:
    st.session_state.model_summary = {}
if "damage_map" not in st.session_state:
    st.session_state.damage_map = {} 
if "damage_df" not in st.session_state:
    st.session_state.damage_df = pd.DataFrame() 

# --- 4. CUSTOM CSS ---
st.markdown("""
<style>
    /* Main progress bar */
    .stProgress > div > div > div > div {
        background-color: #4CAF50 !important;
        height: 20px;
        border-radius: 5px;
    }
    .stProgress > div > div > div {
        background-color: #f0f2f6;
        height: 20px;
        border-radius: 5px;
    }
    /* Style for metric cards */
    div[data-testid="metric-container"] {
        background-color: #F0F2F6;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 10px;
        color: #31333F;
    }
    /* Style for headers */
    h1, h2, h3 {
        color: #004C82;
    }


    /* --- ADD THESE NEW STYLES --- */
    .custom-success-box {
        border: 2px solid #28a745;      /* Green border */
        border-radius: 10px;
        padding: 25px;                  /* Increased padding for a larger box */
        background-color: #d4edda;      /* Light green background */
        color: #155724;                 /* Dark green text */
        font-size: 1.15em;              /* Larger font */
        font-weight: 500;
        margin-bottom: 1em;
    }
    .custom-warning-box {
        border: 2px solid #fd7e14;      /* Orange border */
        border-radius: 10px;
        padding: 25px;                  /* Increased padding for a larger box */
        background-color: #fff3cd;      /* Light orange background */
        color: #856404;                 /* Dark orange text */
        font-size: 1.15em;              /* Larger font */
        font-weight: 500;
        margin-bottom: 1em;
    }
    /* ------------------------------ */

</style>
""", unsafe_allow_html=True)

# --- Sidebar for Global View Options ---
st.sidebar.title("🏢 Global View Options")
st.sidebar.markdown("---")
view_options = {
    'show_frames': st.sidebar.checkbox("Show Frames", value=True),
    'show_walls': st.sidebar.checkbox("Show Walls", value=True),
    'show_damage': st.sidebar.checkbox("Show Damage Overlay", value=True),
    'show_node_labels': st.sidebar.checkbox("Show Node Labels", value=False),
    'show_element_labels': st.sidebar.checkbox("Show Element Labels", value=False),
    'show_grid': st.sidebar.checkbox("Show Grid/Axes", value=True),
    'view': st.sidebar.selectbox("View Type", options=['Isometric', 'Plan View', 'Front View', 'Side View'], index=0),
}
st.sidebar.markdown("---")
st.sidebar.header("Filtering & Highlighting")
min_ele_default, max_ele_default = 0, 1
if st.session_state.model_built and st.session_state.elements:
    all_ele_tags = list(st.session_state.elements.keys())
    min_ele_default = min(all_ele_tags) if all_ele_tags else 0
    max_ele_default = max(all_ele_tags) if all_ele_tags else 1

selected_min_ele, selected_max_ele = st.sidebar.slider(
    "Element ID Range", 
    min_ele_default, max_ele_default, 
    (min_ele_default, max_ele_default),
    key="element_range_slider"
)
view_options['element_range'] = (selected_min_ele, selected_max_ele)

# --- Main Application Body ---
col_logo_left, col_title, col_logo_right = st.columns([1, 4, 1])
with col_logo_left:
    st.image("company_logo.png", width=120) 
with col_title:
    st.markdown("<h1 style='text-align: center; color: #004C82; font-size: 4.0em;'>StrucDamage</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #004C82;'>Real-time structural damage detection powered by PullOver™</h2>", unsafe_allow_html=True) 
    st.markdown("<h3 style='text-align: center; color: #004C82;'>Edificio de Ingenierias-Universidad EAFIT</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>A product by <a href='https://www.risk-and-design.com/' target='_blank'>Risk and Design Consulting</a></p>", unsafe_allow_html=True)
with col_logo_right:
    st.image("company_logo.png", width=120)
st.markdown("---")

# --- Step 0: Project Upload ---
with st.expander("STEP 0: Upload Project Package", expanded=True):
    uploaded_file = st.file_uploader(
        "Upload your Project Package (.zip)",
        type="zip",
        help="The zip file should contain: model.py, analysis.py, project_config.yml, and an 'inputs' folder."
    )
    if uploaded_file is not None:
        if st.button("Load New Project"):
             st.session_state.clear() 
             st.rerun()

        if st.session_state.project_path is None:
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            st.session_state.project_path = temp_dir
            st.success(f"Project '{uploaded_file.name}' uploaded and extracted.")
        
        config_path = os.path.join(st.session_state.project_path, 'project_config.yml')
        try:
            with open(config_path, 'r') as f:
                st.session_state.project_config = yaml.safe_load(f)
            st.info("`project_config.yml` loaded successfully.")
            with st.popover("View Configuration"):
                st.json(st.session_state.project_config)
        except Exception as e:
            st.error(f"Error loading or parsing `project_config.yml`: {e}")
            st.session_state.project_path = None

# --- Step 1: Build & Visualize Model ---
with st.expander("STEP 1: Build & Visualize Model", expanded=True):
    if not st.session_state.get('project_path'):
        st.warning("Please upload a Project Package in Step 0.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("This step is for validating your `model.py` script.")
            if st.button("Build & Visualize Model", key="build_model_initial", use_container_width=True):
                st.session_state.build_triggered = True 
            
            if st.session_state.model_built:
                st.success("✅ **Status:** Model Built Successfully")
                st.markdown("---")
                summary = st.session_state.model_summary
                st.subheader("Model Statistics")
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Total Nodes", summary.get("total_nodes", "N/A"))
                m_col1.metric("Columns", summary.get("columns", "N/A"))
                m_col1.metric("Walls", summary.get("walls", "N/A"))
                m_col2.metric("Total Elements", summary.get("total_elements", "N/A"))
                m_col2.metric("Beams (X-dir)", summary.get("x_beams", "N/A"))
                m_col2.metric("Beams (Y-dir)", summary.get("y_beams", "N/A"))

            if st.session_state.get('build_triggered'):
                del st.session_state.build_triggered
                with st.spinner("Building OpenSees model from `model.py`..."):
                    try:
                        wipe()
                        model_script_path = os.path.join(st.session_state.project_path, 'model.py')
                        if not os.path.exists(model_script_path):
                            raise FileNotFoundError("`model.py` not found in the project package.")
                        
                        user_model_module = load_module_from_path("user_model", model_script_path)
                        user_model_module.build_model()
                        
                        try:
                            st.session_state.nodes = {tag: nodeCoord(tag) for tag in getNodeTags()}
                            st.session_state.elements = {tag: eleNodes(tag) for tag in getEleTags()}
                        except Exception:
                            st.session_state.nodes, st.session_state.elements = {}, {}

                        config = st.session_state.project_config
                        st.session_state.building_metadata = config.get('building_info', {})
                        analysis_setup = config.get('analysis_setup', {})
                        st.session_state.sub_nodes_for_plot = analysis_setup.get('instrumented_nodes', [])
                        st.session_state.dt_npts_for_plot = (analysis_setup.get('dt', 0.0), analysis_setup.get('npts', 0))
                        st.session_state.sub_node_labels_for_plot = {n: f"Instrument {n}" for n in st.session_state.sub_nodes_for_plot}

                        if not st.session_state.elements:
                            st.error("Model build failed: No elements were created.")
                            st.session_state.model_built = False
                        else:
                            st.session_state.model_built = True
                            st.session_state.damage_detected = False
                            nodes, elements = st.session_state.nodes, st.session_state.elements
                            x_beams, y_beams, columns, walls = vu.classify_elements(nodes, elements)
                            st.session_state.model_summary = {
                                "total_nodes": len(nodes), "total_elements": len(elements),
                                "columns": len(columns), "x_beams": len(x_beams),
                                "y_beams": len(y_beams), "walls": len(walls),
                            }
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error building model: {e}")
                        st.session_state.model_built = False
                        st.rerun()
        with col2:
            if st.session_state.model_built:
                meta_col, viz_col = st.columns([1, 3])
                with meta_col:
                    st.subheader("Building Info")
                    if st.session_state.building_metadata:
                        for key, value in st.session_state.building_metadata.items():
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                with viz_col:
                    fig = vu.create_interactive_plot(
                        nodes=st.session_state.nodes, elements=st.session_state.elements,
                        options=view_options, highlight_nodes_list=st.session_state.sub_nodes_for_plot,
                        node_labels_dict=st.session_state.sub_node_labels_for_plot
                    )
                    st.plotly_chart(fig, use_container_width=True, key="initial_model_visualization")


#                    fig = vu.create_interactive_plot(
#                        nodes=st.session_state.nodes, elements=st.session_state.elements,
#                        options=view_options, highlight_nodes_list=st.session_state.sub_nodes_for_plot,
#                        node_labels_dict=st.session_state.sub_node_labels_for_plot
#                    )
#                    st.plotly_chart(fig, use_container_width=True) 
            else:
                st.info("Build the model to visualize the structure.")

# --- Step 2: Run Analysis & Damage Detection ---
st.markdown("---")
st.header("STEP 2: Run Full Analysis & Damage Detection")
st.markdown("Run the analysis in batches or bypass to detect damage from existing results.")

if not st.session_state.get('model_built'):
    st.warning("Please build the model in Step 1 first.")
else:
    # --- NEW: View Input Data and Event Metadata ---
    st.subheader("Event Information & Input Data")
    
    meta_col, plot_col = st.columns([1, 2])

    with meta_col:
        st.markdown("##### Event Metadata")
        try:
            if st.session_state.project_config:
                event_meta = st.session_state.project_config.get('event_metadata', {})
                if event_meta:
                    st.metric(label="Event Name", value=event_meta.get("event_name", "N/A"))
                    st.markdown(f"**Date & Time:** {event_meta.get('date_time', 'N/A')}")
                    st.markdown(f"**Epicentral Distance:** {event_meta.get('epicentral_distance_km', 'N/A')} km")
                    st.markdown(f"**Site Location:** {event_meta.get('site_location', 'N/A')}")
                    st.markdown(f"**Max Recorded Accel.:** {event_meta.get('max_recorded_acceleration_g', 'N/A')} g")
                else:
                    st.warning("No `event_metadata` section found in config file.")
        except Exception as e:
            st.error(f"Could not load event metadata: {e}")

    with plot_col:
        try:
            if st.session_state.project_config:
                analysis_setup = st.session_state.project_config.get('analysis_setup', {})
                instrumented_nodes = analysis_setup.get('instrumented_nodes', [])
                dt = analysis_setup.get('dt', 0.02)
                
                if instrumented_nodes:
                    input_dir = os.path.join(st.session_state.project_path, 'inputs')
                    node_id = st.selectbox("Select Instrumented Node to View", instrumented_nodes)
                    if node_id:
                        ux_file = os.path.join(input_dir, f"master{node_id}_Ux.txt")
                        uy_file = os.path.join(input_dir, f"master{node_id}_Uy.txt")
                        if os.path.exists(ux_file) and os.path.exists(uy_file):
                            ux_data, uy_data = np.loadtxt(ux_file), np.loadtxt(uy_file)
                            time_vector = np.arange(0, len(ux_data) * dt, dt)
                            fig_disp = go.Figure()
                            fig_disp.add_trace(go.Scatter(x=time_vector, y=ux_data, mode='lines', name='Ux'))
                            fig_disp.add_trace(go.Scatter(x=time_vector, y=uy_data, mode='lines', name='Uy'))
                            fig_disp.update_layout(title=f"Node {node_id} Displacement History", height=300, margin=dict(l=20,r=20,b=20,t=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                            st.plotly_chart(fig_disp, use_container_width=True)
                        else:
                            st.warning(f"Input files for node {node_id} not found in 'inputs' folder.")
                else:
                    st.info("No instrumented nodes defined in config.")
        except Exception as e:
            st.error(f"Could not load input data plot: {e}")
    st.markdown("---")


    # --- BATCH SELECTION LOGIC ---
    try:
        damage_config = st.session_state.project_config.get('damage_detection', {})
        element_ranges = damage_config.get('element_ranges', [])
        all_elements_to_record = []
        for ele_range in element_ranges:
            all_elements_to_record.extend(list(range(ele_range['start_ele'], ele_range['end_ele'] + 1)))

        batch_size = 100
        all_batches = [all_elements_to_record[i:i + batch_size] for i in range(0, len(all_elements_to_record), batch_size)]
        
        batch_options = []
        for i, batch in enumerate(all_batches):
            if batch:
                option_str = f"Batch {i+1}: Elements {batch[0]}-{batch[-1]}"
                batch_options.append(option_str)

        selected_batch_strs = st.multiselect(
            "Select element batches to analyze:",
            options=batch_options,
            default=batch_options,
            help="Choose which groups of elements to include in the analysis run."
        )
    except Exception as e:
        st.error(f"Could not define batches from project_config.yml: {e}")
        selected_batch_strs = []

    b_col1, b_col2, b_col3 = st.columns([2,2,1])

    # Button to run the full batched workflow
    if b_col1.button("Run Batched Workflow", key="run_workflow", use_container_width=True, type="primary"):
        if not selected_batch_strs:
            st.warning("Please select at least one batch to analyze.")
        else:
            try:
                if os.path.exists('damage'): shutil.rmtree('damage')
                if os.path.exists('outputs'): shutil.rmtree('outputs')

                with st.spinner("Starting batched workflow..."):
                    status_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    selected_indices = [batch_options.index(s) for s in selected_batch_strs]
                    batches_to_run = [all_batches[i] for i in selected_indices]
                    
                    project_path = st.session_state.project_path
                    user_model_module = load_module_from_path("user_model", os.path.join(project_path, 'model.py'))
                    user_analysis_module = load_module_from_path("user_analysis", os.path.join(project_path, 'analysis.py'))
                    
                    analysis_params = st.session_state.project_config.get('analysis_setup', {})
                    damage_config = st.session_state.project_config.get('damage_detection', {})
                    analysis_params['inputs_dir'] = os.path.join(project_path, 'inputs')
                    
                    total_batches = len(batches_to_run)
                    for i, batch in enumerate(batches_to_run):
                        status_text.info(f"Running Analysis Batch {i+1} of {total_batches} (Elements {batch[0]}-{batch[-1]})")
                        progress_bar.progress(0)
                        user_analysis_module.run_analysis_batch(
                            analysis_params=analysis_params,
                            model_module=user_model_module,
                            elements_to_record=batch,
                            progress_bar=progress_bar,
                            status_text=status_text
                        )

                    status_text.info("All selected batches complete. Detecting damage...")
                    
                    damage_dir = 'damage'
                    df, d_map = dam.detect_damaged_elements_by_moment(
                        damage_config=damage_config,
                        damage_dir=damage_dir
                    )
                    
                    st.session_state.damage_df = df
                    st.session_state.damage_map = d_map
                    st.session_state.damage_detected = True
                
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during the workflow: {e}")
                st.session_state.damage_detected = False

    # Button to bypass analysis and go straight to damage detection
    if b_col2.button("Bypass to Damage Detection", key="bypass", use_container_width=True):
        with st.spinner("Detecting damage from existing results..."):
            try:
                damage_dir = 'damage'
                if not os.path.exists(damage_dir) or not any(os.listdir(damage_dir)):
                    st.warning("Bypass failed: The 'damage' folder is empty or does not exist.")
                else:
                    damage_config = st.session_state.project_config.get('damage_detection', {})
                    df, d_map = dam.detect_damaged_elements_by_moment(
                        damage_config=damage_config,
                        damage_dir=damage_dir
                    )
                    st.session_state.damage_df = df
                    st.session_state.damage_map = d_map
                    st.session_state.damage_detected = True
                    st.rerun()
            except Exception as e:
                st.error(f"An error occurred during damage detection: {e}")

    # Button to clear old results
    if b_col3.button("Clear Old Results", key="clear_results"):
        if os.path.exists('damage'):
            shutil.rmtree('damage')
            st.toast("Cleared 'damage' folder.")
        if os.path.exists('outputs'):
            shutil.rmtree('outputs')
            st.toast("Cleared 'outputs' folder.")

# --- Step 3: Visualize Damage and Analyze Results ---
st.markdown("---")
st.header("STEP 3: Damage Assessment Dashboard")

if st.session_state.damage_detected:
    df = st.session_state.damage_df
    if df.empty:
        # Use the GREEN success box
        message = "✅ **Analysis Complete:** The structure performed within its elastic range. No damage was detected."
        st.markdown(f'<div class="custom-success-box">{message}</div>', unsafe_allow_html=True)
    else:
        # Use the ORANGE warning box
        num_damaged = len(df)
        message = f"⚠️ **Analysis Complete:** Found {num_damaged} element(s) that experienced inelastic behavior. Please review the dashboard for details."
        st.markdown(f'<div class="custom-warning-box">{message}</div>', unsafe_allow_html=True)

    kpi_col, main_viz_col = st.columns([1, 2.5])
    with kpi_col:
        st.subheader("Key Damage Metrics")
        damage_df_display = st.session_state.damage_df
        num_damaged = len(damage_df_display)
        max_ductility = 0.0
        if not damage_df_display.empty:
            ductility_cols = [col for col in damage_df_display.columns if 'Ductility' in col]
            if ductility_cols:
                max_ductility = damage_df_display[ductility_cols].max().max()

        total_cost = damage_df_display['Estimated Cost ($)'].sum() if 'Estimated Cost ($)' in damage_df_display else 0.0

        st.metric(label="Total Estimated Repair Cost", value=f"${total_cost:,.0f}")
        st.metric(label="Damaged Elements", value=f"{num_damaged}")
        st.metric(label="Max Ductility Demand", value=f"{max_ductility:.2f}")

        st.subheader("Damage Summary Report")
        st.dataframe(damage_df_display, use_container_width=True, height=300)
        
        if not damage_df_display.empty:
            st.download_button("📥 Download Summary (CSV)", convert_df_to_csv(damage_df_display),
                               'damage_summary.csv', 'text/csv', use_container_width=True)

    with main_viz_col:
        st.subheader("Interactive 3D Structure View")
        heatmap_mode_label = st.radio("Heatmap View:", ("Ductility Demand", "Estimated Cost"), horizontal=True)
        mode = 'ductility' if heatmap_mode_label == "Ductility Demand" else 'cost'
        base_cost = st.session_state.project_config.get('damage_detection', {}).get('base_cost_per_element', 15000)

        fig = vu.create_interactive_plot(
            nodes=st.session_state.nodes, 
            elements=st.session_state.elements,
            damage_map=st.session_state.damage_map, 
            damage_df=st.session_state.damage_df,
            options=view_options,
            highlight_nodes_list=st.session_state.sub_nodes_for_plot,
            node_labels_dict=st.session_state.sub_node_labels_for_plot,
            heatmap_mode=mode,
            base_cost=base_cost
        )
        st.plotly_chart(fig, use_container_width=True, key="damage_assessment_visualization")

        st.subheader("Moment vs. Curvature (M-κ) for Damaged Elements")
        if not damage_df_display.empty:
            all_damaged_ele_ids = sorted(damage_df_display['Element'].tolist())
            default_selection = all_damaged_ele_ids[:9] if len(all_damaged_ele_ids) >= 9 else all_damaged_ele_ids
            selected_elements_to_plot = st.multiselect("Select Elements to Plot M-κ", all_damaged_ele_ids, default=default_selection)
            
            plot_info = {}
            df_subset = damage_df_display.set_index('Element')
            for ele_id in selected_elements_to_plot:
                if ele_id in df_subset.index:
                    plot_info[ele_id] = {
                        'damage_details': st.session_state.damage_map.get(ele_id, {}),
                        'ductility_data': df_subset.loc[ele_id]
                    }

            if plot_info:
                yield_params_config = st.session_state.project_config.get('damage_detection', {}).get('yield_parameters', {})
                damage_dir = 'damage'

                fig_mk_multi = dam.plot_multiple_moment_vs_curvature(
                    plot_info=plot_info,
                    damage_dir=damage_dir,
                    yield_params_config=yield_params_config,
                    num_cols=3
                )
                st.plotly_chart(fig_mk_multi, use_container_width=True , key="moment_curvature_plots")
else:
    st.info("Run a workflow in Step 2 to view the dashboard.")

