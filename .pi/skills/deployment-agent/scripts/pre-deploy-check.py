#!/usr/bin/env python3
"""
Pre-Deployment Validation Script
Runs comprehensive checks before allowing production deployment
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def log(message: str, color: str = ""):
    """Print colored log message."""
    print(f"{color}{message}{Colors.RESET}")


def run_command(cmd: list[str], cwd: Optional[Path] = None) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=120
        )
        return result.returncode, result.stdout, result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return -1, "", str(e)


class PreDeployChecker:
    """Runs all pre-deployment validations."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.checks_passed = 0
        self.checks_failed = 0
        self.blockers = []  # Critical failures that block deployment
        self.warnings = []  # Warnings that don't block
        
    def check_git_status(self) -> bool:
        """Verify working directory is clean."""
        log("\n📋 Checking Git Status...", Colors.BLUE)
        
        rc, stdout, stderr = run_command(["git", "status", "--porcelain"])
        
        if rc != 0:
            self.blockers.append("Not a git repository")
            log("  ❌ Not a git repository", Colors.RED)
            return False
            
        if stdout.strip():
            lines = stdout.strip().split('\n')
            self.blockers.append(f"Uncommitted changes: {len(lines)} files")
            log(f"  ❌ {len(lines)} uncommitted files", Colors.RED)
            if self.verbose:
                print(stdout)
            return False
            
        log("  ✅ Working directory clean", Colors.GREEN)
        self.checks_passed += 1
        return True
    
    def check_current_branch(self, allowed_branches: list[str]) -> bool:
        """Verify on allowed deployment branch."""
        log("\n🌿 Checking Branch...", Colors.BLUE)
        
        rc, stdout, stderr = run_command(["git", "branch", "--show-current"])
        branch = stdout.strip() if rc == 0 else "unknown"
        
        if allowed_branches and branch not in allowed_branches:
            self.blockers.append(f"Branch '{branch}' not in allowed: {allowed_branches}")
            log(f"  ❌ Current branch: {branch}", Colors.RED)
            log(f"     Allowed: {', '.join(allowed_branches)}", Colors.YELLOW)
            return False
            
        log(f"  ✅ On branch: {branch}", Colors.GREEN)
        self.checks_passed += 1
        return True
    
    def run_security_scan(self) -> bool:
        """Run security checks on codebase."""
        log("\n🔒 Running Security Scan...", Colors.BLUE)
        
        # Check bandit
        rc, stdout, stderr = run_command([
            "bandit", "-r", ".", "-f", "json", "-ll", "-ii"
        ])
        
        if rc != 0:
            try:
                data = json.loads(stdout)
                critical = [r for r in data.get("results", []) 
                          if r.get("issue_severity") == "HIGH"]
                if critical:
                    self.blockers.append(f"Bandit found {len(critical)} HIGH severity issues")
                    log(f"  ❌ Found {len(critical)} HIGH severity security issues", Colors.RED)
                    return False
            except json.JSONDecodeError:
                pass
        
        log("  ✅ No critical security issues", Colors.GREEN)
        self.checks_passed += 1
        return True
    
    def check_dependencies(self) -> bool:
        """Audit dependencies for CVEs."""
        log("\n📦 Checking Dependencies...", Colors.BLUE)
        
        rc, stdout, stderr = run_command(["safety", "check", "--json"])
        
        if rc != 0:
            try:
                data = json.loads(stdout)
                vulnerabilities = data.get("vulnerabilities", [])
                critical = [v for v in vulnerabilities 
                            if v.get("severity") in ["critical", "high"]]
                if critical:
                    self.blockers.append(f"{len(critical)} CRITICAL/HIGH CVEs in dependencies")
                    log(f"  ❌ Found {len(critical)} critical CVEs", Colors.RED)
                    return False
            except json.JSONDecodeError:
                pass
        
        log("  ✅ No critical CVEs in dependencies", Colors.GREEN)
        self.checks_passed += 1
        return True
    
    def check_tests(self, coverage_threshold: int = 80) -> bool:
        """Run test suite and check coverage."""
        log("\n🧪 Running Tests...", Colors.BLUE)
        
        # Run pytest with coverage
        rc, stdout, stderr = run_command([
            "python", "-m", "pytest",
            "--cov=.",
            "--cov-report=term-missing",
            "--cov-fail-under", str(coverage_threshold),
            "-q"
        ])
        
        if rc != 0:
            self.blockers.append("Tests failed or coverage below threshold")
            log("  ❌ Tests failed or insufficient coverage", Colors.RED)
            if stdout:
                print(stdout[-500:])
            return False
            
        log("  ✅ All tests pass", Colors.GREEN)
        self.checks_passed += 1
        return True
    
    def check_environment_vars(self, required: list[str]) -> bool:
        """Verify required environment variables."""
        log("\n🔐 Checking Environment Variables...", Colors.BLUE)
        
        import os
        missing = []
        
        for var in required:
            if not os.getenv(var):
                missing.append(var)
        
        if missing:
            self.blockers.append(f"Missing environment variables: {', '.join(missing)}")
            log(f"  ❌ Missing: {', '.join(missing)}", Colors.RED)
            return False
            
        log(f"  ✅ All {len(required)} required variables present", Colors.GREEN)
        self.checks_passed += 1
        return True
    
    def check_docker(self) -> bool:
        """Verify Docker setup."""
        log("\n🐳 Checking Docker...", Colors.BLUE)
        
        rc, stdout, stderr = run_command(["docker", "--version"])
        if rc != 0:
            self.warnings.append("Docker not available")
            log("  ⚠️  Docker not available", Colors.YELLOW)
            return True  # Warning, not blocker
            
        # Check if Dockerfile exists
        dockerfile = Path("Dockerfile")
        if not dockerfile.exists():
            self.warnings.append("No Dockerfile found")
            log("  ⚠️  No Dockerfile found", Colors.YELLOW)
        else:
            log("  ✅ Dockerfile present", Colors.GREEN)
            
        # Check docker-compose.yml
        compose = Path("docker-compose.yml")
        if compose.exists():
            log("  ✅ docker-compose.yml present", Colors.GREEN)
            
        self.checks_passed += 1
        return True
    
    def check_secrets(self) -> bool:
        """Check for hardcoded secrets."""
        log("\n🕵️  Checking for Hardcoded Secrets...", Colors.BLUE)
        
        patterns = [
            "password\s*=\s*[\"'][^\"']+[\"']",
            "api_key\s*=\s*[\"'][^\"']+[\"']",
            "secret\s*=\s*[\"'][^\"']{10,}[\"']",
            "AWS_ACCESS_KEY_ID",
            "PRIVATE_KEY",
        ]
        
        import re
        
        found = []
        for pattern in patterns:
            rc, stdout, stderr = run_command([
                "grep", "-r", "-n", "-i", 
                "--include=*.py", "--include=*.js", "--include=*.ts",
                "--include=*.env*",
                pattern, "."
            ])
            if rc == 0 and stdout:
                lines = stdout.strip().split('\n')[:5]  # Limit output
                found.extend(lines)
        
        if found:
            self.blockers.append(f"Possible hardcoded secrets found: {len(found)} occurrences")
            log(f"  ❌ Found {len(found)} potential secrets", Colors.RED)
            if self.verbose:
                for line in found:
                    print(f"     {line}")
            return False
            
        log("  ✅ No hardcoded secrets detected", Colors.GREEN)
        self.checks_passed += 1
        return True
    
    def generate_report(self) -> str:
        """Generate final deployment readiness report."""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  PRE-DEPLOYMENT CHECK REPORT")
        lines.append("  Generated:", datetime.now().isoformat())
        lines.append("=" * 70)
        
        lines.append(f"\n  Checks Passed: {self.checks_passed}")
        lines.append(f"  Checks Failed: {self.checks_failed}")
        
        if self.blockers:
            lines.append(f"\n  🔴 DEPLOYMENT BLOCKED ({len(self.blockers)} critical issues):")
            for blocker in self.blockers:
                lines.append(f"     ❌ {blocker}")
        
        if self.warnings:
            lines.append(f"\n  ⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"     ⚠️  {warning}")
        
        lines.append("\n" + "=" * 70)
        
        if self.blockers:
            lines.append("  ❌ DEPLOYMENT NOT RECOMMENDED")
            lines.append("  Fix blockers before deploying")
        else:
            lines.append("  ✅ READY FOR DEPLOYMENT")
        lines.append("=" * 70)
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Pre-deployment Validation")
    parser.add_argument("--branch", default="main,master", 
                       help="Allowed deployment branches (comma-separated)")
    parser.add_argument("--env-vars", default="",
                       help="Required env vars (comma-separated)")
    parser.add_argument("--coverage", type=int, default=80,
                       help="Minimum test coverage percentage")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--report", help="Save report to file")
    args = parser.parse_args()
    
    checker = PreDeployChecker(verbose=args.verbose)
    
    # Run all checks
    allowed_branches = [b.strip() for b in args.branch.split(",")]
    required_env = [v.strip() for v in args.env_vars.split(",") if v.strip()]
    
    checker.check_git_status()
    checker.check_current_branch(allowed_branches)
    checker.check_secrets()
    checker.run_security_scan()
    checker.check_dependencies()
    checker.check_tests(args.coverage)
    
    if required_env:
        checker.check_environment_vars(required_env)
    
    checker.check_docker()
    
    # Generate report
    report = checker.generate_report()
    print(report)
    
    if args.report:
        Path(args.report).write_text(report)
        print(f"\n📄 Report saved to: {args.report}")
    
    # Exit with appropriate code
    sys.exit(1 if checker.blockers else 0)


if __name__ == "__main__":
    main()
