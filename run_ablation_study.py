#!/usr/bin/env python3
"""
Learning Rate Ablation Study Runner

This script runs a simple ablation study on learning rates for the MoE model.

Usage:
    python run_ablation_study.py [--quick]
    
Options:
    --quick: Run a quick study with fewer experiments
"""

import argparse
import sys
from pathlib import Path

from simple_lr_ablation import run_learning_rate_ablation


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run learning rate ablation study")
    parser.add_argument("--quick", action="store_true", help="Run quick study with fewer experiments")
    
    args = parser.parse_args()
    
    if args.quick:
        print("⚡ Running quick ablation study...")
    else:
        print("🚀 Running full ablation study...")
    
    try:
        output_dir = run_learning_rate_ablation()
        print(f"\n✅ Complete! Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Study interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during study: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
