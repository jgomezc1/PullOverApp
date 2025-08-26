# analysis.py
"""
MOCK ANALYSIS SCRIPT for MyPerform3D model visualization in PullOver.

This script provides the necessary function (`run_analysis_batch`) that the
PullOver application expects to find. However, the function body is empty
and does not perform any actual OpenSees analysis.

Its purpose is to allow the user to click through the analysis step in the
PullOver UI without errors, focusing solely on visualizing the model built
by the `model.py` script.
"""
from openseespy.opensees import wipe
import os

def run_analysis_batch(analysis_params, model_module, elements_to_record, progress_bar=None, status_text=None):
    """
    This is a mock function. It does not run any analysis.
    It exists to satisfy the PullOver application's requirements.
    """
    # Print a message to the console/terminal to inform the user.
    print("="*60)
    print("== MOCK ANALYSIS SCRIPT RUNNING ==")
    print("== No analysis will be performed. This is for visualization only.")
    print("="*60)

    # You can optionally create the output directories so the app doesn't error
    # if it checks for them.
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('damage'):
        os.makedirs('damage')

    # Update the UI components to simulate completion.
    if status_text:
        status_text.info("Mock analysis complete. Bypassing to damage detection.")
    if progress_bar:
        progress_bar.progress(100)

    # A final wipe() is good practice.
    wipe()

    # The function completes without running analyze().
    return

