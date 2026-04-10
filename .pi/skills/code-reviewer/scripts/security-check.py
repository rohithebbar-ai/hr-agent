#!/usr/bin/env python3
"""
Security-only code check
Fast security audit of Python/JavaScript/TypeScript files
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_security_checks(file_path: Path) -> bool:
    """Run security checks, return True if passed."""
    print(f"\n🔒 Security Check: {file_path}")
    print("=" * 70)
    
    passed = True
    
    # Run bandit
    print("\n1. Running Bandit (Python security)...")
    result = subprocess.run(
        ["bandit", "-ll", "-ii", "-r", str(file_path)],
        capture_output=True,
        text=True
    )
    if "No issues identified" in result.stdout:
        print("   ✅ Bandit: No issues")
    else:
        print("   ⚠️  Bandit found issues:")
        print(result.stdout)
        passed = False
    
    # Run semgrep
    print("\n2. Running Semgrep (multi-language)...")
    result = subprocess.run(
        ["semgrep", "--config=auto", "--quiet", "--error", str(file_path)],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("   ✅ Semgrep: No issues")
    else:
        print("   ⚠️  Semgrep found issues")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        passed = False
    
    # Check for hardcoded secrets
    print("\n3. Checking for hardcoded secrets...")
    # Simple grep patterns
    secret_patterns = [
        "password\s*=\s*[\"']",
        "api_key\s*=\s*[\"']",
        "secret\s*=\s*[\"'][^\"']{10,}",
        "AWS_ACCESS_KEY",
        "PRIVATE_KEY",
    ]
    
    try:
        content = file_path.read_text()
        import re
        found_secrets = False
        for pattern in secret_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                found_secrets = True
                print(f"   ⚠️  Possible hardcoded secret pattern: {pattern[:30]}...")
        if not found_secrets:
            print("   ✅ No obvious hardcoded secrets")
    except Exception as e:
        print(f"   ⚠️  Could not check: {e}")
    
    print("\n" + "=" * 70)
    if passed:
        print("✅ SECURITY CHECK PASSED")
    else:
        print("❌ SECURITY ISSUES FOUND - Review required")
    print("=" * 70)
    
    return passed


def main():
    parser = argparse.ArgumentParser(description="Security Code Check")
    parser.add_argument("path", help="File or directory to check")
    args = parser.parse_args()
    
    target = Path(args.path)
    
    if target.is_file():
        files = [target]
    else:
        files = (
            list(target.rglob("*.py")) + 
            list(target.rglob("*.js")) + 
            list(target.rglob("*.ts"))
        )
    
    all_passed = True
    for file in files:
        if not run_security_checks(file):
            all_passed = False
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
