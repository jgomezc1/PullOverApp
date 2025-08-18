# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 21:44:41 2025

@author: jgomez (Refactored by Gemini)
"""

from openseespy.opensees import *
import numpy as np
import os

def run_analysis_batch(analysis_params, model_module, elements_to_record, progress_bar=None, status_text=None):
    """
    FINAL BATCH VERSION: This version is designed to be called multiple times.
    It runs a full analysis but only defines recorders for the specific
    list of elements passed to it. It also updates the progress bar.
    """
    # 1. UNPACK PARAMETERS
    dt           = analysis_params['dt']
    npts         = analysis_params['npts']
    sub_nodes    = analysis_params['instrumented_nodes']
    master_nodes = analysis_params['master_nodes']
    
    # 2. WIPE AND BUILD THE MODEL
    wipe()
    model_module.build_model()
    
    # Ensure local directories exist
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('damage' , exist_ok=True)

    # 3. ASSIGN MASS
    M, M_r = 1.0e-6, 1.0e-6
    for node in master_nodes:
        if nodeCoord(node):
            mass(node, M, M, 0.0, 0.0, 0.0, M_r)

    loadConst('-time', 0.0)

    # 4. DEFINE GROUND MOTION
    series_tag, gm_tag = 3, 101
    pattern('MultipleSupport', 3)
    
    inputs_dir = analysis_params['inputs_dir']
    for node in sub_nodes:
        for dof, comp in zip([1, 2 , 6], ['Ux', 'Uy', 'Te']):
            file_path = os.path.join(inputs_dir, f"master{node}_{comp}.txt")
            if not os.path.exists(file_path):
                print(f"WARNING: Input file not found at {file_path}. Skipping.")
                continue
            timeSeries('Path', series_tag, '-dt', dt, '-filePath', file_path)
            groundMotion(gm_tag, 'Series', '-disp', series_tag)
            imposedSupportMotion(node, dof, gm_tag)
            series_tag += 1
            gm_tag += 1
            
    rayleigh(0.1416, 0.0, 0.0, 0.00281)

    # 5. DEFINE RECORDERS FOR THE CURRENT BATCH
    if status_text:
        status_text.info(f"Defining recorders for {len(elements_to_record)} elements...")
    
    section_ids = [1, 4]
    for ele_id in elements_to_record:
        for sec_id in section_ids:
            recorder('Element', '-file', f'damage/Rele{ele_id}_h{sec_id}_deformation.out', '-time', '-ele', ele_id, '-section', sec_id, 'deformation')
            recorder('Element', '-file', f'damage/Rele{ele_id}_h{sec_id}_force.out', '-time', '-ele', ele_id, '-section', sec_id, 'force')
    
    # Record nodal displacements only once (we check if the first batch is being run)
    if 1 in elements_to_record:
        for node in master_nodes:
            recorder('Node', '-file', f'outputs/resdis{node}_history.out', '-time', '-dT', dt, '-node', node, '-dof', 1, 2, 6, 'disp')

    # 6. ANALYSIS OBJECT DEFINITION
    wipeAnalysis()
    constraints('Transformation')
    numberer('RCM')
    system('SparseGeneral')
    test('EnergyIncr', 1.0e-3, 20)
    algorithm('Newton')
    integrator('Newmark', 0.5, 0.25)
    analysis('Transient')

    # 7. PERFORM THE TRANSIENT ANALYSIS
    tFinal = npts * dt
    tCurrent = getTime()
    ok = 0
    
    while ok == 0 and tCurrent < tFinal:    
        ok = analyze(1, dt)        
        if ok != 0:
            if status_text: status_text.warning("Convergence failed. Trying a new algorithm...")
            test('NormDispIncr', 1.0e-6, 100, 0)
            algorithm('ModifiedNewton', '-initial')
            ok = analyze(1, dt)
            if ok == 0:
                if status_text: status_text.info("Succeeded. Reverting to original algorithm.")
            else:
                 if status_text: status_text.error("Analysis failed to converge.")
                 break
            test('EnergyIncr', 1.0e-3, 20)
            algorithm('Newton')
        
        tCurrent = getTime()
        
        # --- NEW: Update progress bar within the loop ---
        if progress_bar is not None:
            progress = int((tCurrent / tFinal) * 100) if tFinal > 0 else 100
            progress_bar.progress(progress)
    
    # 8. FINALIZE AND CLOSE RECORDER FILES
    wipe()