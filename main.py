#!/usr/bin/env python3
"""
STL Torus Boolean Subtraction Tool - Web UI

A web-based GUI application for loading STL models, selecting 3 points
to define a torus segment, and performing boolean subtraction.
"""
from gui.web_viewer import create_app


def main():
    """Application entry point."""
    server = create_app()
    print("\n" + "="*50)
    print("STL Torus Subtraction Tool")
    print("="*50)
    print("\nStarting web server...")
    print("Open your browser to: http://localhost:8080")
    print("\nPress Ctrl+C to stop the server")
    print("="*50 + "\n")
    server.start()


if __name__ == "__main__":
    main()
