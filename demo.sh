#!/bin/bash
# Demo script for ClarificationAgent

echo "═══════════════════════════════════════════════════════════════"
echo "  MULTI-AGENT PRD GENERATOR - ClarificationAgent Demo"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Activate virtual environment
source venv/bin/activate

# Example 1: Simple run
echo "📋 Example 1: Fintech Product"
echo "---"
python -m app.main "Build a tool for freelance designers to track invoices" 2>&1 | grep -v "INFO"
echo ""

# Example 2: Healthcare with verbose
echo "📋 Example 2: Healthcare Product (with --verbose)"
echo "---"
python -m app.main "HIPAA-compliant telemedicine platform" --verbose 2>&1 | grep -v "INFO" | tail -40
echo ""

# Example 3: List all runs
echo "📋 Example 3: List All Runs"
echo "---"
python -m app.main --list
echo ""

echo "✨ Demo complete!"
