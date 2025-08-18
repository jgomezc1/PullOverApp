# -*- coding: utf-8 -*-
"""
TEMPLATE FOR BATCH ANALYSIS (analysis.py)
"""
from openseespy.opensees import *
import numpy as np
import os

def run_analysis_batch(analysis_params, model_module, elements_to_record, progress_bar=None, status_text=None):
    """
    This function runs a full analysis but only defines recorders for the
    specific list of elements passed to it.
    """
    # --------------------------------------------------------------------
    # 1. UNPACK PARAMETERS
    # --------------------------------------------------------------------
    dt = analysis_params['dt']
    npts = analysis_params['npts']
    sub_nodes = analysis_params['instrumented_nodes']
    master_nodes = analysis_params['master_nodes']
    inputs_dir = analysis_params['inputs_dir']

    # --------------------------------------------------------------------
    # 2. BUILD THE MODEL
    # --------------------------------------------------------------------
    wipe()
    model_module.build_model()
    
    # Ensure local directories exist (outputs written to app's root folder)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('damage', exist_ok=True)

    # --------------------------------------------------------------------
    # 3. DEFINE LOADS, MASS, AND DAMPING
    # --------------------------------------------------------------------
    # --- Assign Mass ---
    # Example: M, M_r = 1.0e-6, 1.0e-6
    # for node in master_nodes:
    #     if nodeCoord(node):
    #         mass(node, M, M, 0.0, 0.0, 0.0, M_r)

    # --- Define Ground Motion / Time Series ---
    pattern('MultipleSupport', 3) # Or UniformExciitation, etc.
    # timeSeries('Path', ...) for ground motion files
    # groundMotion(...) or load(...)
    
    # --- Define Damping ---
    # rayleigh(alphaM, betaK, betaKinit, betaKcomm)

    # --------------------------------------------------------------------
    # 4. DEFINE RECORDERS FOR THE CURRENT BATCH
    # --------------------------------------------------------------------
    if status_text:
        status_text.info(f"Defining recorders for {len(elements_to_record)} elements...")
    
    section_ids = [1, 4] # Example section IDs to record
    for ele_id in elements_to_record:
        for sec_id in section_ids:
            recorder('Element', '-file', f'damage/Rele{ele_id}_h{sec_id}_deformation.out', '-time', '-ele', ele_id, '-section', sec_id, 'deformation')
            recorder('Element', '-file', f'damage/Rele{ele_id}_h{sec_id}_force.out', '-time', '-ele', ele_id, '-section', sec_id, 'force')
    
    # Record nodal displacements only once (check if it's the first batch)
    if 1 in elements_to_record:
        for node in master_nodes:
            recorder('Node', '-file', f'outputs/resdis{node}_history.out', '-time', '-dT', dt, '-node', node, '-dof', 1, 2, 6, 'disp')

    # --------------------------------------------------------------------
    # 5. DEFINE ANALYSIS OBJECTS
    # --------------------------------------------------------------------
    wipeAnalysis()
    constraints('Transformation')
    numberer('RCM')
    system('SparseGeneral')
    test('EnergyIncr', 1.0e-6, 20)
    algorithm('Newton')
    integrator('Newmark', 0.5, 0.25)
    analysis('Transient')

    # --------------------------------------------------------------------
    # 6. PERFORM THE ANALYSIS
    # --------------------------------------------------------------------
    tFinal = npts * dt
    tCurrent = getTime()
    ok = 0
    
    while ok == 0 and tCurrent < tFinal:    
        ok = analyze(1, dt)        
        if ok != 0:
            # Convergence failure logic
            if status_text: status_text.warning("Convergence failed, trying recovery step...")
            test('NormDispIncr', 1.0e-8, 100, 0)
            algorithm('ModifiedNewton', '-initial')
            ok = analyze(1, dt)
            # Revert to original settings
            test('EnergyIncr', 1.0e-6, 20)
            algorithm('Newton')
        
        tCurrent = getTime()
        
        # Update progress bar
        if progress_bar is not None:
            progress = int((tCurrent / tFinal) * 100) if tFinal > 0 else 100
            progress_bar.progress(progress)
    
    # --------------------------------------------------------------------
    # 7. FINALIZE
    # --------------------------------------------------------------------
    wipe()