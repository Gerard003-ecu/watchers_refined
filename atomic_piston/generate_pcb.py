#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KiCad 9 PCB Generation and Diagnostics Script for 'Pistón Atómico'.

This script provides a robust, step-by-step workflow for generating a PCB
from a KiCad project. It uses the `kicad-cli` for stable, command-line
operations like ERC, DRC, and schematic-to-PCB updates, and the `pcbnew`
Python API for board manipulation tasks that require it.

The process is designed to be modular and provide clear logging to help
diagnose issues in the PCB generation flow.

Author: Jules
Date: 2025-09-22
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys

# --- KICAD API IMPORT ---
# Encapsulate pcbnew import in a try-except block to provide a clear
# error message if the script is not run with KiCad's Python environment.
try:
    import pcbnew
except ImportError:
    print("Error: The 'pcbnew' library could not be imported.", file=sys.stderr)
    print("Please ensure you are running this script using the Python interpreter", file=sys.stderr)
    print("that is bundled with KiCad 9.", file=sys.stderr)
    print(r'Example (Windows): "C:\Program Files\KiCad\9.0\bin\python.exe" generate_pcb.py <args>', file=sys.stderr)
    sys.exit(1)

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def run_erc(schematic_file: str, output_dir: str) -> bool:
    """Phase 1: Run Electrical Rule Check (ERC) on the schematic."""
    logger.info("PHASE 1: Running Electrical Rule Check (ERC)...")
    report_file = os.path.join(output_dir, "erc_report.json")
    command = [
        "kicad-cli", "sch", "erc",
        "--output", report_file,
        "--format", "json",
        "--exit-code-violations",
        schematic_file
    ]

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        logger.info(f"kicad-cli erc stdout:\n{process.stdout}")
        if process.stderr:
            logger.warning(f"kicad-cli erc stderr:\n{process.stderr}")

        if process.returncode == 0:
            logger.info("ERC check passed. No violations found.")
            return True
        else:
            logger.error(f"ERC check failed. Violations found. See report: {report_file}")
            with open(report_file, 'r') as f:
                violations = json.load(f).get('violations', [])
            for violation in violations:
                logger.error(f"  - {violation['description']} (Severity: {violation['severity']})")
            return False

    except FileNotFoundError:
        logger.error("`kicad-cli` not found. Is KiCad installed and in your system's PATH?")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during ERC check: {e}")
        return False


def verify_footprints(schematic_file: str) -> bool:
    """
    Phase 1: Verify that all schematic components have footprints.
    This function performs a basic text-based search in the .kicad_sch file.
    """
    logger.info("PHASE 1: Verifying footprint assignments in schematic...")
    missing_footprints = []
    try:
        with open(schematic_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Use regex to find all symbol instances, capturing their reference and content
        symbol_pattern = re.compile(r'\(\s*symbol\s+([^)]*?)\s*\((.*?)\)\s*\)', re.DOTALL)
        symbols = symbol_pattern.finditer(content)

        for match in symbols:
            symbol_content = match.group(2)

            # Find the reference designator (e.g., R1, U1)
            ref_match = re.search(r'\(\s*property\s+"Reference"\s+"([^"]+)"', symbol_content)
            ref = ref_match.group(1) if ref_match else "UNKNOWN"

            # Check for a non-empty footprint property
            fp_match = re.search(r'\(\s*property\s+"Footprint"\s+"([^"]*)"', symbol_content)
            if not fp_match or not fp_match.group(1).strip():
                missing_footprints.append(ref)

    except FileNotFoundError:
        logger.error(f"Schematic file not found for footprint check: {schematic_file}")
        return False
    except Exception as e:
        logger.error(f"Failed to parse schematic for footprint check: {e}")
        return False

    if not missing_footprints:
        logger.info("Footprint assignment verification passed.")
        return True
    else:
        logger.error("Footprint verification failed. The following components are missing a footprint:")
        for comp_ref in missing_footprints:
            logger.error(f"  - {comp_ref}")
        return False


def update_pcb_from_schematic(schematic_file: str, pcb_file: str) -> bool:
    """Phase 2: Update the PCB from the schematic using kicad-cli."""
    logger.info("PHASE 2: Updating PCB from schematic...")
    command = [
        "kicad-cli", "sch", "pcb", "update",
        "--schematic", schematic_file,
        pcb_file
    ]

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        logger.info(f"kicad-cli pcb update stdout:\n{process.stdout}")
        if process.stderr:
            logger.warning(f"kicad-cli pcb update stderr:\n{process.stderr}")

        process.check_returncode()  # Raises CalledProcessError if return code is non-zero
        logger.info("PCB updated successfully from schematic.")
        return True

    except FileNotFoundError:
        logger.error("`kicad-cli` not found. Is KiCad installed and in your system's PATH?")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to update PCB from schematic. kicad-cli returned error: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during PCB update: {e}")
        return False


def place_components(board: 'pcbnew.BOARD'):
    """Phase 3: Place all components in the center of the board."""
    logger.info("PHASE 3: Placing all components at the center for diagnostics...")
    center_x, center_y = 100, 100  # Center point in mm
    try:
        footprints = list(board.GetFootprints())
        if not footprints:
            logger.warning("No footprints found on the board to place.")
            return

        logger.info(f"Moving {len(footprints)} footprints to ({center_x}, {center_y}) mm.")
        for i, fp in enumerate(footprints):
            # Stagger the components slightly to avoid a perfect pile-up
            pos = pcbnew.wxPointMM(center_x + i * 0.1, center_y + i * 0.1)
            fp.SetPosition(pos)
            fp.SetOrientationDegrees(0) # Reset rotation

        logger.info("Component placement complete.")
    except Exception as e:
        logger.error(f"An error occurred during component placement: {e}")
        # Re-raise the exception to stop the script if placement fails.
        # This is a critical step, and we should not proceed if it fails.
        raise


def route_test_track(board: 'pcbnew.BOARD'):
    """Phase 4: Route a single test track to verify routing engine."""
    logger.info("PHASE 4: Attempting to route a test track...")

    try:
        # Find a suitable net for routing. A ground net is a common choice.
        # Let's find any net with at least two pads.
        net_to_route = None
        target_net_name = "GND" # A common default
        net = board.FindNet(target_net_name)
        if net and len(net.Pads()) >= 2:
            net_to_route = net
        else: # Fallback: find the first net with enough pads
            for n in board.GetNetsByName().values():
                if len(n.Pads()) >= 2:
                    net_to_route = n
                    break

        if not net_to_route:
            logger.warning("Could not find any net with >= 2 pads to route. Skipping test track.")
            return

        logger.info(f"Found net '{net_to_route.GetNetname()}' to route.")

        pads = list(net_to_route.Pads())
        pad1 = pads[0]
        pad2 = pads[1]

        start_point = pad1.GetCenter()
        end_point = pad2.GetCenter()

        logger.info(f"Routing from {pad1.GetParent().GetReference()}/{pad1.GetPadName()} to {pad2.GetParent().GetReference()}/{pad2.GetPadName()} on F.Cu")

        # Create and add the track
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(start_point)
        track.SetEnd(end_point)
        track.SetNet(net_to_route)
        track.SetWidth(pcbnew.FromMM(0.25))  # Standard signal track width
        track.SetLayer(pcbnew.F_Cu)
        board.Add(track)

        logger.info("Test track routed successfully.")

    except Exception as e:
        logger.error("CRITICAL: An error occurred during the test track routing phase.")
        logger.error("This is a likely point of failure for PCB automation scripts.")
        logger.error(f"The exact error was: {e}")
        # Re-raise to halt execution
        raise


def run_drc(pcb_file: str, output_dir: str) -> bool:
    """Phase 5: Run Design Rule Check (DRC) on the PCB."""
    logger.info("PHASE 5: Running Design Rule Check (DRC)...")
    report_file = os.path.join(output_dir, "drc_report.json")
    command = [
        "kicad-cli", "pcb", "drc",
        "--output", report_file,
        "--format", "json",
        "--exit-code-violations",
        pcb_file
    ]

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        logger.info(f"kicad-cli drc stdout:\n{process.stdout}")
        if process.stderr:
            logger.warning(f"kicad-cli drc stderr:\n{process.stderr}")

        if process.returncode == 0:
            logger.info("DRC check passed. No violations found.")
            return True
        else:
            logger.error(f"DRC check failed. Violations found. See report: {report_file}")
            with open(report_file, 'r') as f:
                report = json.load(f)
                violations = report.get('violations', [])
                unconnected = report.get('unconnected_items', 0)

            if unconnected > 0:
                logger.error(f"  - {unconnected} unconnected items found.")

            for violation in violations:
                logger.error(f"  - {violation['description']} (Severity: {violation['severity']})")
            return False

    except FileNotFoundError:
        logger.error("`kicad-cli` not found. Is KiCad installed and in your system's PATH?")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during DRC check: {e}")
        return False


def main():
    """Main execution flow."""
    parser = argparse.ArgumentParser(
        description="KiCad 9 PCB Generation and Diagnostics Script."
    )
    parser.add_argument(
        "project_file",
        help="Path to the KiCad project file (.kicad_pro)."
    )
    args = parser.parse_args()

    project_path = os.path.abspath(args.project_file)
    if not os.path.exists(project_path):
        logger.error(f"Project file not found: {project_path}")
        sys.exit(1)

    project_dir = os.path.dirname(project_path)
    project_name = os.path.splitext(os.path.basename(project_path))[0]

    # Create an output directory for reports
    output_dir = os.path.join(project_dir, "generation_output")
    os.makedirs(output_dir, exist_ok=True)

    # Construct paths to schematic and PCB files
    schematic_file = os.path.join(project_dir, f"{project_name}.kicad_sch")
    pcb_file = os.path.join(project_dir, f"{project_name}.kicad_pcb")

    if not os.path.exists(schematic_file):
        logger.error(f"Schematic file not found: {schematic_file}")
        sys.exit(1)
    if not os.path.exists(pcb_file):
        logger.error(f"PCB file not found: {pcb_file}")
        sys.exit(1)

    logger.info(f"Starting PCB generation for project: {project_name}")
    logger.info(f"  - Project Dir: {project_dir}")
    logger.info(f"  - Output Dir:  {output_dir}")
    logger.info(f"  - Schematic:   {schematic_file}")
    logger.info(f"  - PCB:         {pcb_file}")

    # --- EXECUTION PHASES ---

    # Phase 1: Pre-flight checks
    if not run_erc(schematic_file, output_dir):
        logger.error("ERC checks failed. Please fix schematic errors before proceeding.")
        sys.exit(1)

    if not verify_footprints(schematic_file):
        logger.error("Footprint assignment is incomplete. Please assign all footprints in the schematic.")
        sys.exit(1)

    # Phase 2: Synchronization
    if not update_pcb_from_schematic(schematic_file, pcb_file):
        logger.error("Failed to update PCB from schematic. Check logs for details.")
        sys.exit(1)

    # Reload the board after update
    logger.info("Reloading PCB after update...")
    board = pcbnew.LoadBoard(pcb_file)

    # Phase 3: Placement
    place_components(board)

    # Phase 4: Routing
    route_test_track(board)

    # Save before DRC to ensure routed track is checked
    logger.info("Saving board before final DRC check...")
    board.Save(pcb_file)

    # Phase 5: Final Verification
    if not run_drc(pcb_file, output_dir):
        logger.warning("DRC checks found violations. See report for details.")
    else:
        logger.info("DRC checks passed.")

    # Final Save
    board.Save(pcb_file)
    logger.info(f"Process complete. Modified PCB saved to: {pcb_file}")
    sys.exit(0)


if __name__ == "__main__":
    main()
