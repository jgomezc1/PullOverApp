# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 13:56:42 2025

@author: jgomez (Refactored by Gemini)
"""
from openseespy.opensees import *
import math

def build_model():
    """
    This function encapsulates the entire model-building process.
    The main Streamlit app will call this function to build the model.
    """
    wipe()
    model('basic', '-ndm', 3, '-ndf', 6)
    #
    ########################################################################################################################
    #                                                                                                                      #
    #                           GEOMETRIC PARAMETERS, NODES, BCs AND RIGID DIAPHRAGMS                                      #
    #                                                                                                                      #
    ########################################################################################################################
    story_height = 2.8
    num_stories  = 20
    x1 = 6.0
    x2 = 11.0
    y1 = 4.0
    y2 = 8.0

    # Floor plan nodes (i, j) → (x, y)
    node_coords_base = {
        (0, 0): (0.0, 0.0),
        (1, 0): (x1, 0.0),
        (2, 0): (x2, 0.0),
        (0, 1): (0.0, y1),
        (1, 1): (x1, y1),
        (2, 1): (x2, y1),
        (0, 2): (0.0, y2),
        (1, 2): (x1, y2)
    }

    # Master node location (approximate centroid)
    master_xy = (5.23, 2.0)

    # Node ID bookkeeping
    node_map = {}
    node_id  = 1

    master_node_ids = []

    # Loop through stories and assign floor nodes + master nodes
    for k in range(num_stories + 1):
        z = round(k * story_height, 3)
        for (i, j), (x, y) in node_coords_base.items():
            node(node_id, x, y, z)
            node_map[(i, j, k)] = node_id
            node_id += 1
        if k > 0:
            # Master node
           node(node_id, *master_xy, z)
           master_id = node_id
           master_node_ids.append(node_id)
    #        Slave nodes
           slave_ids = [node_map[(i, j, k)] for (i, j) in node_coords_base.keys()]
           rigidDiaphragm(3, master_id, *slave_ids)
           fix(master_id, 0, 0, 1, 1, 1, 0)  # Free in X, Y, Rz
           node_id += 1


    # Fix all base nodes
    for (i, j) in node_coords_base.keys():
        base_node = node_map[(i, j, 0)]
        fix(base_node, 1, 1, 1, 1, 1, 1)   
    #
    ########################################################################################################################
    #                                                                                                                      #
    #                                                COLUMNS (Inelastic)                                                   #
    #                                                                                                                      #
    ########################################################################################################################
    #
    geomTransf('Linear', 111, 1, 0, 0)  # Local y-axis parallel to global X

    elasticTag = 4444

    b_col = 0.70  # width
    h_col = 0.70  # depth

    # Material properties
    E_col  = 2.5e10    # Pa (typical for concrete)
    nu_col = 0.2      # Poisson ratio (assumed)
    G_col  = E_col / (2 * (1 + nu_col))  # Shear modulus

    # Section properties
    A_col  =  b_col * h_col                           # Area [m²]
    Iy_col = (b_col * h_col**3) / 12                  # Moment of inertia about local y-axis [m⁴]
    Iz_col = (h_col * b_col**3) / 12                  # Moment of inertia about local z-axis [m⁴]
    J_col  =  b_col * h_col**3 / 3                    # Torsional constant [m⁴] (approximation)



    section('Elastic', elasticTag, E_col, A_col, Iz_col, Iy_col, G_col, J_col)

    # ------------------------------------------------------------------------------ ---------------------------------------
    #
    # Hinge definition for My
    #
    matMy_x    = 6                                   # Tags
    My_yield_x = 3000000.0                           # N-m
    theta_y    = 0.005                               # rad (yield rotation)
    Lpy        = 0.10                                # m (hinge length)
    E_curv_y_x = My_yield_x / (theta_y / Lpy)        # Slope of the M vs Curv relationship
    by         = 0.01 
    R0         = 20
    cR1        = 0.925
    cR2        = 0.15
    a1         = 0.01                                # isotropic hardening
    a2         = 1.0
    a3         = 0.01
    a4         = 1.0

    uniaxialMaterial('Steel02', matMy_x, My_yield_x, E_curv_y_x, by, R0, cR1, cR2, a1, a2, a3, a4)

    # ------------------------------------------------------------------------------ ---------------------------------------
    #
    # Hinge definition for Mz
    #
    matMz_x    = 7
    Mz_yield_x = 3000000.0                           
    theta_z    = 0.005                              
    Lpz        = 0.10  
    E_curv_z_x = Mz_yield_x / (theta_z / Lpz)
    bz         = 0.01
    R0         = 20
    cR1        = 0.925
    cR2        = 0.15
    a1         = 0.01                              
    a2         = 1.0
    a3         = 0.01
    a4         = 1.0

    uniaxialMaterial('Steel02', matMz_x, Mz_yield_x, E_curv_z_x, bz, R0, cR1, cR2, a1, a2, a3, a4)

    # ------------------------------------------------------------------------------ ---------------------------------------
    # AGGREGATOR sections for end i and end j
    # ------------------------------------------------------------------------------ --------------------------------------
    secHingeTag_i = 40
    section('Aggregator', secHingeTag_i,
             matMy_x, 'My',                            # flexure about local y (strong axis)
             matMz_x, 'Mz',                            # flexure about local z (weak axis)
             '-section', elasticTag)

    secHingeTag_j = 41
    section('Aggregator', secHingeTag_j,
             matMy_x, 'My',                           # flexure about local y (strong axis)
             matMz_x, 'Mz',                           # flexure about local z (weak axis)
             '-section', elasticTag)


    #
    # ------------------------------------------------------------------------------ --------------------------------------
    # BEAM INTEGRATION: concentrated plasticity at both ends + Elastic behaviour in the middle of the element
    # ------------------------------------------------------------------------------ --------------------------------------
    hingeLength = 0.20
    beamIntTag  = 33
    beamIntegration('HingeEndpoint', beamIntTag,
                    secHingeTag_i, hingeLength,
                    secHingeTag_j, hingeLength,
                    elasticTag)
    eleType = 'forceBeamColumn'
    elem_id = 1
    for k in range(num_stories):
        for (i, j) in node_coords_base.keys():  # Loop over all grid points (excluding master)
            node_i = node_map[(i, j, k)]
            node_j = node_map[(i, j, k + 1)]
            element(eleType, elem_id, node_i, node_j, 111 , 33)
            elem_id += 1
    #
    ########################################################################################################################
    #                                                                                                                      #
    #                                                BEAMS (Inelastic)                                                     #
    #                                                                                                                      #
    ########################################################################################################################
    #
    # Define Geometric Transformations for beams
    # in X-Direction (Use transformation tag 222)
    #
    geomTransf('Linear', 222, 0, 0, 1)  # Local y-axis parallel to global Z
    #
    #======================================================================================================================
    #
    # START Sección properties for beams parallel to X
    #
    #======================================================================================================================
    # ------------------------------------------------------------------------------ ---------------------------------------
    # Elastic section ( axial, shear and torsion modes)
    # ------------------------------------------------------------------------------ ---------------------------------------
    elasticTag = 1111
    b_sec      = 0.40                      # m
    h_sec      = 0.50                      # m
    A          = b_sec * h_sec
    Iy         = (b_sec * h_sec**3) / 12
    Iz         = (h_sec * b_sec**3) / 12
    J          = b_sec * h_sec**3 / 3
    E          = 2.50e10                      # Pa (concrete)
    G          = 1.04e10                      # Pa
    section('Elastic', elasticTag, E, A, Iz, Iy, G, J)
    # ------------------------------------------------------------------------------ ---------------------------------------
    #
    # Hinge definition for My
    #
    matMy_x    = 1                                   # Tags
    My_yield_x = 612000.0                            # N-m
    theta_y    = 0.005                               # rad (yield rotation)
    Lpy        = 0.10                                # m (hinge length)
    E_curv_y_x = My_yield_x / (theta_y / Lpy)
    by         = 0.01 
    R0         = 20
    cR1        = 0.925
    cR2        = 0.15
    a1         = 0.01                                # isotropic hardening
    a2         = 1.0
    a3         = 0.01
    a4         = 1.0

    uniaxialMaterial('Steel02', matMy_x, My_yield_x, E_curv_y_x, by, R0, cR1, cR2, a1, a2, a3, a4)
    # ------------------------------------------------------------------------------ ---------------------------------------
    #
    # Hinge definition for Mz
    #
    matMz_x    = 2
    Mz_yield_x = 1650000.0                          # N-m
    theta_z    = 0.005                              # rad
    Lpz        = 0.10  
    E_curv_z_x = Mz_yield_x / (theta_z / Lpz)
    bz         = 0.01
    R0         = 20
    cR1        = 0.925
    cR2        = 0.15
    a1         = 0.01                              # isotropic hardening
    a2         = 1.0
    a3         = 0.01
    a4         = 1.0

    uniaxialMaterial('Steel02', matMz_x, Mz_yield_x, E_curv_z_x, bz, R0, cR1, cR2, a1, a2, a3, a4)
    # ------------------------------------------------------------------------------ ---------------------------------------
    # AGGREGATOR sections for end i and end j
    # ------------------------------------------------------------------------------ --------------------------------------
    secHingeTag_i = 20
    section('Aggregator', secHingeTag_i,
             matMy_x, 'My',   # flexure about local y (strong axis)
             matMz_x, 'Mz',   # flexure about local z (weak axis)
             '-section', elasticTag)

    secHingeTag_j = 21
    section('Aggregator', secHingeTag_j,
             matMy_x, 'My',   # flexure about local y (strong axis)
             matMz_x, 'Mz',   # flexure about local z (weak axis)
             '-section', elasticTag)
    #
    # ------------------------------------------------------------------------------ --------------------------------------
    # BEAM INTEGRATION: concentrated plasticity at both ends + Elastic behaviour in the middle of the element
    # ------------------------------------------------------------------------------ --------------------------------------
    hingeLength = 0.10
    beamIntTag  = 22
    beamIntegration('HingeEndpoint', beamIntTag,
                    secHingeTag_i, hingeLength,
                    secHingeTag_j, hingeLength,
                    elasticTag)
    eleType = 'forceBeamColumn'
    # -----------------------------
    # BEAMS PARALLEL TO X-DIRECTION
    # -----------------------------
    for k in range(1, num_stories + 1):  # Floor level
    #    print("Storey =" , k)
        for j in [0, 1, 2]:  # Y-positions
            for i in [0, 1]:  # X-beams between i=0→1 and i=1→2
                if (i, j) in node_coords_base and (i + 1, j) in node_coords_base:
                    # Skip the missing panel at (2,2)
                    if (i, j) == (1, 2):
                        continue
                    n1 = node_map[(i, j, k)]
                    n2 = node_map[(i + 1, j, k)]
                    element(eleType, elem_id, n1, n2, 222 , 22)
    #                print("Beam ID =", elem_id)
                    elem_id += 1
                    
    #======================================================================================================================
    #
    # START Sección properties for beams parallel to Y
    #
    #======================================================================================================================
    geomTransf('Linear', 333, 0, 0, 1)  # Local y-axis = global Z
    # ------------------------------------------------------------------------------ ---------------------------------------
    # Elastic section ( axial, shear and torsion modes)
    # ------------------------------------------------------------------------------ ---------------------------------------
    elasticTag = 2222
    b_sec = 0.40   # in
    h_sec = 0.50   # in
    A = b_sec * h_sec
    Iy = (b_sec * h_sec**3) / 12
    Iz = (h_sec * b_sec**3) / 12
    J = b_sec * h_sec**3 / 3
    E = 2.50e10     # ksi (concrete)
    G = 1.04e10     # ksi
    section('Elastic', elasticTag, E, A, Iz, Iy, G, J)
    # ------------------------------------------------------------------------------ ---------------------------------------
    #
    # Hinge definition for My
    #
    matMy_y = 3
    My_yield_y = 612000.0      # kip-in
    theta_y = 0.005        # rad (yield rotation)
    Lpy = 0.10              # in (hinge length)
    E_curv_y_y = My_yield_y / (theta_y / Lpy)
    by = 0.01  # hardening ratio
    R0 = 20
    cR1 = 0.925
    cR2 = 0.15
    a1 = 0.01  # isotropic hardening
    a2 = 1.0
    a3 = 0.01
    a4 = 1.0

    uniaxialMaterial('Steel02', matMy_y, My_yield_y, E_curv_y_y, by, R0, cR1, cR2, a1, a2, a3, a4)

    # ------------------------------------------------------------------------------ ---------------------------------------
    #
    # Hinge definition for Mz
    #
    matMz_y = 4
    Mz_yield_y = 1650000.0      # kip-in
    theta_z = 0.005        # rad
    Lpz = 0.10              # in (hinge length)
    E_curv_z_y = Mz_yield_y / (theta_z / Lpz)
    bz = 0.01
    R0 = 20
    cR1 = 0.925
    cR2 = 0.15
    a1 = 0.01  # isotropic hardening
    a2 = 1.0
    a3 = 0.01
    a4 = 1.0

    uniaxialMaterial('Steel02', matMz_y, Mz_yield_y, E_curv_z_y, bz, R0, cR1, cR2, a1, a2, a3, a4)

    # ------------------------------------------------------------------------------ --------------------------------------
    # AGGREGATOR sections for end i and end j
    # ------------------------------------------------------------------------------ --------------------------------------
    secHingeTag_i = 32
    section('Aggregator', secHingeTag_i,
             matMy_y, 'My',   # flexure about local y (strong axis)
             matMz_y, 'Mz',   # flexure about local z (weak axis)
             '-section', elasticTag)

    secHingeTag_j = 34
    section('Aggregator', secHingeTag_j,
             matMy_y, 'My',   # flexure about local y (strong axis)
             matMz_y, 'Mz',   # flexure about local z (weak axis)
             '-section', elasticTag)
    #
    # ------------------------------------------------------------------------------ --------------------------------------
    # BEAM INTEGRATION: concentrated plasticity at both ends + Elastic behaviour in the middle of the element
    # ------------------------------------------------------------------------------ --------------------------------------
    #
    hingeLength = 0.10
    beamIntTag  = 44
    beamIntegration('HingeEndpoint', beamIntTag,
                    secHingeTag_i, hingeLength,
                    secHingeTag_j, hingeLength,
                    elasticTag)

    # -----------------------------
    # BEAMS PARALLEL TO Y-DIRECTION
    # -----------------------------
    for k in range(1, num_stories + 1):  # Each floor
    #    print("Storey =" , k)
        for i in [0, 1, 2]:  # X-grid positions
            for j in [0, 1]:  # Y-beams between j=0→1 and j=1→2
                if (i, j) in node_coords_base and (i, j + 1) in node_coords_base:
                    # Skip the missing panel (2,1)-(2,2)
                    if (i, j) == (2, 1):
                        continue
                    n1 = node_map[(i, j, k)]
                    n2 = node_map[(i, j + 1, k)]
                    element(eleType, elem_id, n1, n2, 333 , 44)
    #                print("Beam ID =", elem_id)
                    elem_id += 1

    print("User-defined Python model built successfully.")