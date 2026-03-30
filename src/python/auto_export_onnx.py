#!/usr/bin/env python3
"""
Auto-export ONNX script - monitors checkpoint folder and exports ONNX for new checkpoints
"""

import os
import time
import subprocess
import glob
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ONNXExporter:
    def __init__(self, checkpoint_folder, onnx_folder, check_interval=60):
        """
        Args:
            checkpoint_folder: Path to folder containing .ckpt files (folder A)
            onnx_folder: Path to folder for exported ONNX files (folder B)
            check_interval: Seconds between checks
        """
        self.checkpoint_folder = Path(checkpoint_folder)
        self.onnx_folder = Path(onnx_folder)
        self.check_interval = check_interval

        # Ensure folders exist
        self.checkpoint_folder.mkdir(parents=True, exist_ok=True)
        self.onnx_folder.mkdir(parents=True, exist_ok=True)

        # Track processed checkpoints
        self.processed_checkpoints = set()

    def get_checkpoint_name(self, ckpt_path):
        """Extract name from checkpoint path for ONNX filename"""
        return ckpt_path.stem

    def get_onnx_path(self, ckpt_path):
        """Get corresponding ONNX path for a checkpoint"""
        onnx_name = self.get_checkpoint_name(ckpt_path) + ".onnx"
        return self.onnx_folder / onnx_name

    def find_checkpoints(self):
        """Find all .ckpt files in checkpoint folder"""
        pattern = str(self.checkpoint_folder / "**" / "*.ckpt")
        ckpt_files = glob.glob(pattern, recursive=True)
        return [Path(f) for f in ckpt_files]

    def needs_export(self, ckpt_path):
        """Check if checkpoint needs ONNX export"""
        onnx_path = self.get_onnx_path(ckpt_path)
        return not onnx_path.exists()

    def export_onnx(self, ckpt_path):
        """Run ONNX export command"""
        onnx_path = self.get_onnx_path(ckpt_path)

        cmd = [
            "python3",
            "-m",
            "piper_train.export_onnx",
            str(ckpt_path),
            str(onnx_path),
        ]

        logger.info(f"Exporting ONNX: {ckpt_path.name} -> {onnx_path.name}")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Export successful: {onnx_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Export failed for {ckpt_path.name}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False

    def check_and_export(self):
        """Check for new checkpoints and export if needed"""
        checkpoints = self.find_checkpoints()

        new_exports = 0
        for ckpt_path in checkpoints:
            if ckpt_path not in self.processed_checkpoints:
                if self.needs_export(ckpt_path):
                    if self.export_onnx(ckpt_path):
                        new_exports += 1
                else:
                    logger.debug(f"ONNX already exists for {ckpt_path.name}")

                self.processed_checkpoints.add(ckpt_path)

        if new_exports > 0:
            logger.info(f"Exported {new_exports} new ONNX file(s)")

        return new_exports

    def run(self):
        """Main loop - run indefinitely"""
        logger.info(f"Starting ONNX auto-exporter")
        logger.info(f"Checkpoint folder: {self.checkpoint_folder}")
        logger.info(f"ONNX folder: {self.onnx_folder}")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                self.check_and_export()
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            logger.info("\nStopped by user")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Auto-export ONNX from checkpoints")
    parser.add_argument(
        "--checkpoint-folder",
        "-c",
        required=True,
        help="Folder containing .ckpt files (folder A)",
    )
    parser.add_argument(
        "--onnx-folder",
        "-o",
        required=True,
        help="Folder for exported ONNX files (folder B)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--once", action="store_true", help="Run once and exit (no loop)"
    )

    args = parser.parse_args()

    exporter = ONNXExporter(
        checkpoint_folder=args.checkpoint_folder,
        onnx_folder=args.onnx_folder,
        check_interval=args.interval,
    )

    if args.once:
        exporter.check_and_export()
    else:
        exporter.run()


if __name__ == "__main__":
    main()
