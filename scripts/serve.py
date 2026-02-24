#!/usr/bin/env python3
"""
scripts/serve.py
=================
Start the NeSy-Core inference server.

Usage:
    python scripts/serve.py --host 0.0.0.0 --port 8000 --reload
    python scripts/serve.py --domain medical --rules configs/medical_rules.json
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="NeSy-Core Inference Server")
    parser.add_argument("--host",   default="0.0.0.0", help="Bind host")
    parser.add_argument("--port",   default=8000, type=int, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Hot reload")
    parser.add_argument("--domain", default="general", help="Domain context")
    parser.add_argument("--rules",  default=None, help="Path to rules JSON")
    args = parser.parse_args()

    print(f"Starting NeSy-Core server on {args.host}:{args.port}")
    print(f"Domain: {args.domain}")
    if args.rules:
        print(f"Loading rules from: {args.rules}")

    try:
        from nesy.deployment.server.app import serve
        serve(host=args.host, port=args.port, reload=args.reload)
    except ImportError:
        print("Server requires: pip install nesy-core[server]")
        sys.exit(1)


if __name__ == "__main__":
    main()
