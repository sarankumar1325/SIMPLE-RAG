#!/usr/bin/env python3
"""
Simple script to run the RAG application.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed."""
    try:
        import streamlit
        import chromadb
        import google.generativeai
        import sentence_transformers
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has required variables."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please copy .env.example to .env and add your Gemini API key")
        return False
    
    # Check if GEMINI_API_KEY is set
    with open(env_file, 'r') as f:
        content = f.read()
        if "GEMINI_API_KEY=" not in content or "your_gemini_api_key_here" in content:
            print("❌ GEMINI_API_KEY not properly set in .env file")
            print("Please add your actual Gemini API key to the .env file")
            return False
    
    print("✅ Environment configuration looks good")
    return True

def main():
    """Main function to run the application."""
    print("🤖 Simple RAG Application Launcher")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_env_file():
        sys.exit(1)
    
    print("\n🚀 Starting Streamlit application...")
    print("The application will open in your default web browser")
    print("Press Ctrl+C to stop the application")
    print("-" * 40)
    
    try:
        # Run streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

