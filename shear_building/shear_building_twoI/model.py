"""
TEMPLATE FOR MODEL DEFINITION (model.py)
This file contains the full model definition from 'muro_corte.py',
placed within the build_model() function to be compatible with the Streamlit app.
"""
from openseespy.opensees import *

def build_model():
    """
    This function contains the entire OpenSees model definition.
    The main Streamlit app will call this function to build the model.
    """
    # --------------------------------------------------------------------
    # 1. BASIC MODEL SETUP
    # (Code from muro_corte.py starts here)
    # --------------------------------------------------------------------
    wipe()
    model('basic', '-ndm', 3, '-ndf', 6)  # 3D model with 6 DOF per node

    # --------------------------------------------------------------------
    # 2. NODAL POINTS
    # --------------------------------------------------------------------
    # Story height
    h = 3.0

    # Define nodes for the moment frame, story by story
    # Floor 1 frame nodes
    node(1060 , 5.0 , 0.0, 0.0)
    node(1070 , 5.0 , 5.0, 0.0)
    node(1080 , 5.0 , 0.0, h)
    node(1090 , 5.0 , 5.0, h)
    node(1100 , 10.0 , 0.0, 0.0)
    node(1110 , 10.0 , 5.0, 0.0)
    node(1120 , 10.0 , 0.0, h)
    node(1130 , 10.0 , 5.0, h)

    # Floor 2 frame nodes
    node(1160 , 5.0 , 0.0, 2*h)
    node(1170 , 5.0 , 5.0, 2*h)
    node(1180 , 10.0 , 0.0, 2*h)
    node(1190 , 10.0 , 5.0, 2*h)

    # Floor 3 frame nodes
    node(1260 , 5.0 , 0.0, 3*h)
    node(1270 , 5.0 , 5.0, 3*h)
    node(1280 , 10.0 , 0.0, 3*h)
    node(1290 , 10.0 , 5.0, 3*h)

    # Floor 4 frame nodes
    node(1360 , 5.0 , 0.0, 4*h)
    node(1370 , 5.0 , 5.0, 4*h)
    node(1380 , 10.0 , 0.0, 4*h)
    node(1390 , 10.0 , 5.0, 4*h)

    # Floor 5 frame nodes
    node(1460 , 5.0 , 0.0, 5*h)
    node(1470 , 5.0 , 5.0, 5*h)
    node(1480 , 10.0 , 0.0, 5*h)
    node(1490 , 10.0 , 5.0, 5*h)

    # Floor 6 frame nodes
    node(1560 , 5.0 , 0.0, 6*h)
    node(1570 , 5.0 , 5.0, 6*h)
    node(1580 , 10.0 , 0.0, 6*h)
    node(1590 , 10.0 , 5.0, 6*h)

    # Floor 7 frame nodes
    node(1660 , 5.0 , 0.0, 7*h)
    node(1670 , 5.0 , 5.0, 7*h)
    node(1680 , 10.0 , 0.0, 7*h)
    node(1690 , 10.0 , 5.0, 7*h)

    # Floor 8 frame nodes
    node(1760 , 5.0 , 0.0, 8*h)
    node(1770 , 5.0 , 5.0, 8*h)
    node(1780 , 10.0 , 0.0, 8*h)
    node(1790 , 10.0 , 5.0, 8*h)

    # Floor 9 frame nodes
    node(1860 , 5.0 , 0.0, 9*h)
    node(1870 , 5.0 , 5.0, 9*h)
    node(1880 , 10.0 , 0.0, 9*h)
    node(1890 , 10.0 , 5.0, 9*h)

    # Floor 10 frame nodes
    node(1960 , 5.0 , 0.0, 10*h)
    node(1970 , 5.0 , 5.0, 10*h)
    node(1980 , 10.0 , 0.0, 10*h)
    node(1990 , 10.0 , 5.0, 10*h)

    # --------------------------------------------------------------------
    # 3. MATERIAL AND SECTION DEFINITIONS
    # --------------------------------------------------------------------

    # --- Materials for the Shear Wall (Inelastic) ---
    uniaxialMaterial('Concrete02', 201, -40e6, -0.0035, -25e6, -0.02, 0.1, 2.5e6, 150e6)
    uniaxialMaterial('SteelMPF', 301, 5.0e8, 5.5e8, 2.0e11, 0.01, 0.15, 20, 0.925, 0.15)
    uniaxialMaterial('Elastic', 1, 6.0e8) # Shear material for MVLEM

    # --- Materials for Columns (Inelastic Hinges) ---
    matMy_col = 6
    My_yield_col = 1000000.0
    theta_y_col = 0.005
    Lpy_col = 0.10
    E_curv_y_col = My_yield_col / (theta_y_col / Lpy_col)
    uniaxialMaterial('Steel02', matMy_col, My_yield_col, E_curv_y_col, 0.01, 20, 0.925, 0.15, 0.01, 1.0, 0.01, 1.0)

    matMz_col = 7
    Mz_yield_col = 1000000.0
    theta_z_col = 0.005
    Lpz_col = 0.10
    E_curv_z_col = Mz_yield_col / (theta_z_col / Lpz_col)
    uniaxialMaterial('Steel02', matMz_col, Mz_yield_col, E_curv_z_col, 0.01, 20, 0.925, 0.15, 0.01, 1.0, 0.01, 1.0)

    # --- Sections for Columns ---
    elasticTag_col = 4444
    b_col, h_col = 0.70, 0.70
    E_col, nu_col = 2.5e10, 0.2
    G_col = E_col / (2 * (1 + nu_col))
    A_col, Iy_col, Iz_col = b_col * h_col, (b_col * h_col**3)/12, (h_col * b_col**3)/12
    J_col = b_col * h_col**3 / 3
    section('Elastic', elasticTag_col, E_col, A_col, Iz_col, Iy_col, G_col, J_col)

    secHingeTag_i_col, secHingeTag_j_col = 40, 41
    section('Aggregator', secHingeTag_i_col, matMy_col, 'My', matMz_col, 'Mz', '-section', elasticTag_col)
    section('Aggregator', secHingeTag_j_col, matMy_col, 'My', matMz_col, 'Mz', '-section', elasticTag_col)

    # --- Materials for Beams in X-Direction (Inelastic Hinges) ---
    matMy_beamX, matMz_beamX = 100, 200
    My_yield_beamX, Mz_yield_beamX = 612000.0, 1650000.0
    uniaxialMaterial('Steel02', matMy_beamX, My_yield_beamX, My_yield_beamX / (0.005 / 0.10), 0.01, 20, 0.925, 0.15, 0.01, 1.0, 0.01, 1.0)
    uniaxialMaterial('Steel02', matMz_beamX, Mz_yield_beamX, Mz_yield_beamX / (0.005 / 0.10), 0.01, 20, 0.925, 0.15, 0.01, 1.0, 0.01, 1.0)

    # --- Sections for Beams in X-Direction ---
    elasticTag_beamX = 1111
    b_beamX, h_beamX = 0.40, 0.50
    E_beam, G_beam = 2.50e10, 1.04e10
    A_beamX, Iy_beamX, Iz_beamX = b_beamX * h_beamX, (b_beamX * h_beamX**3)/12, (h_beamX * b_beamX**3)/12
    J_beamX = b_beamX * h_beamX**3 / 3
    section('Elastic', elasticTag_beamX, E_beam, A_beamX, Iz_beamX, Iy_beamX, G_beam, J_beamX)

    secHingeTag_i_beamX, secHingeTag_j_beamX = 20, 21
    section('Aggregator', secHingeTag_i_beamX, matMy_beamX, 'My', matMz_beamX, 'Mz', '-section', elasticTag_beamX)
    section('Aggregator', secHingeTag_j_beamX, matMy_beamX, 'My', matMz_beamX, 'Mz', '-section', elasticTag_beamX)

    # --- Materials for Beams in Y-Direction (Inelastic Hinges) ---
    matMy_beamY, matMz_beamY = 300, 400
    My_yield_beamY, Mz_yield_beamY = 612000.0, 1650000.0
    uniaxialMaterial('Steel02', matMy_beamY, My_yield_beamY, My_yield_beamY / (0.005 / 0.10), 0.01, 20, 0.925, 0.15, 0.01, 1.0, 0.01, 1.0)
    uniaxialMaterial('Steel02', matMz_beamY, Mz_yield_beamY, Mz_yield_beamY / (0.005 / 0.10), 0.01, 20, 0.925, 0.15, 0.01, 1.0, 0.01, 1.0)

    # --- Sections for Beams in Y-Direction ---
    elasticTag_beamY = 2222
    b_beamY, h_beamY = 0.40, 0.50
    A_beamY, Iy_beamY, Iz_beamY = b_beamY * h_beamY, (b_beamY * h_beamY**3)/12, (h_beamY * b_beamY**3)/12
    J_beamY = b_beamY * h_beamY**3 / 3
    section('Elastic', elasticTag_beamY, E_beam, A_beamY, Iz_beamY, Iy_beamY, G_beam, J_beamY)

    secHingeTag_i_beamY, secHingeTag_j_beamY = 32, 34
    section('Aggregator', secHingeTag_i_beamY, matMy_beamY, 'My', matMz_beamY, 'Mz', '-section', elasticTag_beamY)
    section('Aggregator', secHingeTag_j_beamY, matMy_beamY, 'My', matMz_beamY, 'Mz', '-section', elasticTag_beamY)

    # --------------------------------------------------------------------
    # 4. GEOMETRIC TRANSFORMATIONS & BEAM INTEGRATION
    # --------------------------------------------------------------------
    geomTransf('Linear', 111, 1, 0, 0)  # Columns
    geomTransf('Linear', 222, 0, 0, 1)  # Beams in X
    geomTransf('Linear', 333, 0, 0, 1)  # Beams in Y

    beamIntTag_col = 33
    beamIntegration('HingeEndpoint', beamIntTag_col, secHingeTag_i_col, 0.20, secHingeTag_j_col, 0.20, elasticTag_col)

    beamIntTag_beamX = 22
    beamIntegration('HingeEndpoint', beamIntTag_beamX, secHingeTag_i_beamX, 0.10, secHingeTag_j_beamX, 0.10, elasticTag_beamX)

    beamIntTag_beamY = 44
    beamIntegration('HingeEndpoint', beamIntTag_beamY, secHingeTag_i_beamY, 0.10, secHingeTag_j_beamY, 0.10, elasticTag_beamY)

    # --------------------------------------------------------------------
    # 5. ELEMENT CREATION
    # --------------------------------------------------------------------
    # --- Shear Wall (Inelastic) using MVLEM_3D ---
    eleTag_start, nodeTag_start = 1, 1
    x1_wall, y1_wall = 0.0, 0.0
    wall_length, wall_height = 5.0, 30.0
    n_elements, n_fibers = 30, 5
    dir_vector = (0.0, 1.0)
    wall_thickness, rho = 0.3, 0.01
    matConcrete, matSteel, matShear = 201, 301, 1

    dx, dy = dir_vector
    element_height = wall_height / n_elements
    fiber_width = wall_length / n_fibers
    
    width_list = [fiber_width] * n_fibers
    thickness_list = [wall_thickness] * n_fibers
    rho_list = [rho] * n_fibers
    matConcrete_list = [matConcrete] * n_fibers
    matSteel_list = [matSteel] * n_fibers

    nodeTag, eleTag = nodeTag_start, eleTag_start
    x2_wall, y2_wall = x1_wall + wall_length * dx, y1_wall + wall_length * dy
    
    bottom_node_1_tag, bottom_node_2_tag = nodeTag, nodeTag + 1
    node(bottom_node_1_tag, x1_wall, y1_wall, 0.0)
    node(bottom_node_2_tag, x2_wall, y2_wall, 0.0)
    nodeTag += 2

    for i in range(n_elements):
        z_top = (i + 1) * element_height
        top_node_1_tag, top_node_2_tag = nodeTag, nodeTag + 1
        
        node(top_node_1_tag, x2_wall, y2_wall, z_top)
        node(top_node_2_tag, x1_wall, y1_wall, z_top)

        node_tags_for_element = [bottom_node_1_tag, bottom_node_2_tag, top_node_1_tag, top_node_2_tag]
        
        element('MVLEM_3D', eleTag, *node_tags_for_element, n_fibers,
                '-thick', *thickness_list, '-width', *width_list, '-rho', *rho_list,
                '-matConcrete', *matConcrete_list, '-matSteel', *matSteel_list, '-matShear', matShear)

        bottom_node_1_tag, bottom_node_2_tag = top_node_2_tag, top_node_1_tag
        nodeTag += 2
        eleTag += 1

    # --- Columns (Inelastic) ---
    eleType = 'forceBeamColumn'
    element(eleType, 10006, 1060, 1080, 111, 33)
    element(eleType, 10007, 1070, 1090, 111, 33)
    element(eleType, 10008, 1100, 1120, 111, 33)
    element(eleType, 10009, 1110, 1130, 111, 33)
    element(eleType, 10010, 1080, 1160, 111, 33)
    element(eleType, 10011, 1090, 1170, 111, 33)
    element(eleType, 10012, 1120, 1180, 111, 33)
    element(eleType, 10013, 1130, 1190, 111, 33)
    element(eleType, 10014, 1160, 1260, 111, 33)
    element(eleType, 10015, 1170, 1270, 111, 33)
    element(eleType, 10016, 1180, 1280, 111, 33)
    element(eleType, 10017, 1190, 1290, 111, 33)
    element(eleType, 10018, 1260, 1360, 111, 33)
    element(eleType, 10019, 1270, 1370, 111, 33)
    element(eleType, 10020, 1280, 1380, 111, 33)
    element(eleType, 10021, 1290, 1390, 111, 33)
    element(eleType, 10046, 1360, 1460, 111, 33)
    element(eleType, 10047, 1370, 1470, 111, 33)
    element(eleType, 10048, 1380, 1480, 111, 33)
    element(eleType, 10049, 1390, 1490, 111, 33)
    element(eleType, 10056, 1460, 1560, 111, 33)
    element(eleType, 10057, 1470, 1570, 111, 33)
    element(eleType, 10058, 1480, 1580, 111, 33)
    element(eleType, 10059, 1490, 1590, 111, 33)
    element(eleType, 10066, 1560, 1660, 111, 33)
    element(eleType, 10067, 1570, 1670, 111, 33)
    element(eleType, 10068, 1580, 1680, 111, 33)
    element(eleType, 10069, 1590, 1690, 111, 33)
    element(eleType, 10076, 1660, 1760, 111, 33)
    element(eleType, 10077, 1670, 1770, 111, 33)
    element(eleType, 10078, 1680, 1780, 111, 33)
    element(eleType, 10079, 1690, 1790, 111, 33)
    element(eleType, 10086, 1760, 1860, 111, 33)
    element(eleType, 10087, 1770, 1870, 111, 33)
    element(eleType, 10088, 1780, 1880, 111, 33)
    element(eleType, 10089, 1790, 1890, 111, 33)
    element(eleType, 10096, 1860, 1960, 111, 33)
    element(eleType, 10097, 1870, 1970, 111, 33)
    element(eleType, 10098, 1880, 1980, 111, 33)
    element(eleType, 10099, 1890, 1990, 111, 33)

    # --- Beams (Inelastic) ---
    element(eleType, 10022, 8, 1080, 222, 22)
    element(eleType, 10023, 1080, 1120, 222, 22)
    element(eleType, 10024, 7, 1090, 222, 22)
    element(eleType, 10025, 1090, 1130, 222, 22)
    element(eleType, 10026, 14, 1160, 222, 22)
    element(eleType, 10027, 1160, 1180, 222, 22)
    element(eleType, 10028, 13, 1170, 222, 22)
    element(eleType, 10029, 1170, 1190, 222, 22)
    element(eleType, 10030, 20, 1260, 222, 22)
    element(eleType, 10031, 1260, 1280, 222, 22)
    element(eleType, 10032, 19, 1270, 222, 22)
    element(eleType, 10033, 1270, 1290, 222, 22)
    element(eleType, 10034, 26, 1360, 222, 22)
    element(eleType, 10035, 1360, 1380, 222, 22)
    element(eleType, 10036, 25, 1370, 222, 22)
    element(eleType, 10037, 1370, 1390, 222, 22)
    element(eleType, 10050, 32, 1460, 222, 22)
    element(eleType, 10051, 1460, 1480, 222, 22)
    element(eleType, 10052, 31, 1470, 222, 22)
    element(eleType, 10053, 1470, 1490, 222, 22)
    element(eleType, 10060, 38, 1560, 222, 22)
    element(eleType, 10061, 1560, 1580, 222, 22)
    element(eleType, 10062, 37, 1570, 222, 22)
    element(eleType, 10063, 1570, 1590, 222, 22)
    element(eleType, 10070, 44, 1660, 222, 22)
    element(eleType, 10071, 1660, 1680, 222, 22)
    element(eleType, 10072, 43, 1670, 222, 22)
    element(eleType, 10073, 1670, 1690, 222, 22)
    element(eleType, 10080, 50, 1760, 222, 22)
    element(eleType, 10081, 1760, 1780, 222, 22)
    element(eleType, 10082, 49, 1770, 222, 22)
    element(eleType, 10083, 1770, 1790, 222, 22)
    element(eleType, 10090, 56, 1860, 222, 22)
    element(eleType, 10091, 1860, 1880, 222, 22)
    element(eleType, 10092, 55, 1870, 222, 22)
    element(eleType, 10093, 1870, 1890, 222, 22)
    element(eleType, 10100, 62, 1960, 222, 22)
    element(eleType, 10101, 1960, 1980, 222, 22)
    element(eleType, 10102, 61, 1970, 222, 22)
    element(eleType, 10103, 1970, 1990, 222, 22)
    
    element(eleType, 10038, 1080, 1090, 333, 44)
    element(eleType, 10039, 1120, 1130, 333, 44)
    element(eleType, 10040, 1160, 1170, 333, 44)
    element(eleType, 10041, 1180, 1190, 333, 44)
    element(eleType, 10042, 1260, 1270, 333, 44)
    element(eleType, 10043, 1280, 1290, 333, 44)
    element(eleType, 10044, 1360, 1370, 333, 44)
    element(eleType, 10045, 1380, 1390, 333, 44)
    element(eleType, 10054, 1460, 1470, 333, 44)
    element(eleType, 10055, 1480, 1490, 333, 44)
    element(eleType, 10064, 1560, 1570, 333, 44)
    element(eleType, 10065, 1580, 1590, 333, 44)
    element(eleType, 10074, 1660, 1670, 333, 44)
    element(eleType, 10075, 1680, 1690, 333, 44)
    element(eleType, 10084, 1760, 1770, 333, 44)
    element(eleType, 10085, 1780, 1790, 333, 44)
    element(eleType, 10094, 1860, 1870, 333, 44)
    element(eleType, 10095, 1880, 1890, 333, 44)
    element(eleType, 10104, 1960, 1970, 333, 44)
    element(eleType, 10105, 1980, 1990, 333, 44)

    # --------------------------------------------------------------------
    # 6. BOUNDARY CONDITIONS & DIAPHRAGMS
    # --------------------------------------------------------------------
    for node_id in [1, 2, 1060, 1070, 1100, 1110]:
        fix(node_id, 1, 1, 1, 1, 1, 1)

    for floor in range(1, 11):
        z = h * floor
        master_tag = 500 + floor
        x_master, y_master = 7.5, 2.5
        node(master_tag, x_master, y_master, z)
        fix(master_tag, 0, 0, 1, 1, 1, 0)

    rigidDiaphragm(3, 501, 7,8  ,1080,1090,1120,1130)
    rigidDiaphragm(3, 502, 13,14,1160,1170,1180,1190)
    rigidDiaphragm(3, 503, 19,20,1260,1270,1280,1290)
    rigidDiaphragm(3, 504, 25,26,1360,1370,1380,1390)
    rigidDiaphragm(3, 505, 31,32,1460,1470,1480,1490)
    rigidDiaphragm(3, 506, 37,38,1560,1570,1580,1590)
    rigidDiaphragm(3, 507, 43,44,1660,1670,1680,1690)
    rigidDiaphragm(3, 508, 49,50,1760,1770,1780,1790)
    rigidDiaphragm(3, 509, 55,56,1860,1870,1880,1890)
    rigidDiaphragm(3, 510, 61,62,1960,1970,1980,1990)

    print("User-defined Python model built successfully.")

