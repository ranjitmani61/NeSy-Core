#!/usr/bin/env python3
"""
scripts/export_model.py
========================
Export a NeSy-Core neural backbone to ONNX, TFLite, or CoreML.

The symbolic engine, concept graph, and metacognition monitor run
on CPU and are NOT exported — only the neural backbone is converted.

Usage:
    python scripts/export_model.py --format onnx --output model.onnx --input-dim 128
    python scripts/export_model.py --format tflite --output model.tflite
    python scripts/export_model.py --format coreml --output model.mlmodel
"""
import argparse
import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nesy.deployment.exporter import ModelExporter
from nesy.neural.base import PassthroughBackbone

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export NeSy-Core Backbone")
    parser.add_argument("--format", choices=["onnx", "tflite", "coreml"],
                        required=True, help="Export format")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--input-dim", type=int, default=128,
                        help="Input feature dimension (default: 128)")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply quantization (TFLite only)")
    parser.add_argument("--opset", type=int, default=13,
                        help="ONNX opset version (default: 13)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # In production, load the actual trained backbone here.
    # For demonstration, use PassthroughBackbone as a placeholder backbone.
    logger.info("Loading backbone for export...")
    backbone = PassthroughBackbone(dim=args.input_dim)

    exporter = ModelExporter()

    try:
        if args.format == "onnx":
            logger.info(f"Exporting to ONNX (opset {args.opset})...")
            exporter.to_onnx(
                backbone=backbone,
                output_path=args.output,
                input_dim=args.input_dim,
                opset_version=args.opset,
            )

        elif args.format == "tflite":
            logger.info(f"Exporting to TFLite (quantize={args.quantize})...")
            exporter.to_tflite(
                backbone=backbone,
                output_path=args.output,
                input_dim=args.input_dim,
                quantise=args.quantize,
            )

        elif args.format == "coreml":
            logger.info("Exporting to CoreML...")
            exporter.to_coreml(
                backbone=backbone,
                output_path=args.output,
                input_dim=args.input_dim,
            )

        logger.info(f"Export complete: {args.output}")

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
