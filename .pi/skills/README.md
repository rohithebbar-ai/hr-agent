# Custom Pi Skills for HR-Agent Project

This directory contains custom skills (sub-agents) designed for the hr-agent project. These skills extend pi's capabilities for code review and production deployment.

---

## Available Skills

### 1. `code-reviewer`
Comprehensive code review assistant for Python and JavaScript/TypeScript.

**What it does:**
- Security vulnerability detection (SQL injection, XSS, hardcoded secrets)
- Static analysis for bugs and code smells
- Style checking (PEP 8 compliance)
- Type checking for Python (mypy)
- Performance issue identification

**Usage:**
```bash
# Automatic activation
"Review this Python code for security issues"
"Check this file for bugs"

# Manual invocation
/skill:code-reviewer ./rag/baseline_rag.py
/skill:code-reviewer --security-only ./api/
```

**Files:**
- `SKILL.md` - Main instructions
- `scripts/review.py` - Comprehensive review script
- `scripts/security-check.py` - Fast security-only scan
- `references/security-checklist.md` - Security review reference

### 2. `deployment-agent`
Production deployment assistant with safety-first approach.

**What it does:**
- Pre-deployment validation (tests, security, dependencies)
- Multi-strategy deployments (rolling, blue-green, canary)
- Automatic rollback on failure
- Post-deployment health checks
- Docker image scanning

**Usage:**
```bash
# Automatic activation
"Deploy to production"
"Prepare for deployment"
"Roll back the last deployment"

# Manual invocation
/skill:deployment-agent --canary 10
/skill:deployment-agent --rollback
/skill:deployment-agent --strategy blue-green
```

**Files:**
- `SKILL.md` - Main instructions
- `scripts/pre-deploy-check.py` - Pre-flight validation
- `scripts/deploy.sh` - Main deployment script
- `scripts/rollback.sh` - Emergency rollback
- `scripts/health-check.sh` - Post-deploy verification
- `references/pre-deployment-checklist.md` - Production checklist

---

## Integration with HR-Agent

### Usage Flow

1. **Code Review → Deployment**
   ```
   User: "Review the RAG pipeline code"
   → pi loads code-reviewer
   → Finds critical security issue
   → User: "Deploy the application"
   → pi loads deployment-agent
   → deployment-agent calls code-reviewer
   → BLOCKS deployment due to security issue
   → Reports issue to user
   ```

2. **Pre-deployment Check**
   ```
   User: "Deploy to production"
   → deployment-agent runs pre-deploy-check.py
   ├─ Git status clean?
   ├─ Tests passing?
   ├─ Security scan clean?
   ├─ No hardcoded secrets?
   └─ Dependencies audited?
   → All checks pass
   → Proceed with build & deploy
   → Health check verification
   → Deployment complete
   ```

3. **Emergency Rollback**
   ```
   User: "Roll back now!"
   → deployment-agent executes rollback.sh
   → Stops current deployment
   → Reverts to previous version
   → Health check verification
   → Confirms rollback success
   ```

---

## Setup Requirements

### For Code Reviewer
```bash
pip install ruff bandit mypy semgrep safety
```

### For Deployment Agent
```bash
# Docker
brew install docker docker-compose

# Security scanning
pip install safety
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh

# Make scripts executable
chmod +x .pi/skills/*/scripts/*
```

---

## Skill Discovery in Pi

Pi automatically discovers these skills from the `.pi/skills/` directory. The skills are loaded based on:

1. **Project context** - When working in hr-agent/, skills are available
2. **Task matching** - Pi matches your request against skill descriptions
3. **Manual invocation** - Use `/skill:name` to force load

### Skill Commands

```bash
# List available skills in context
"What skills are available?"

# Load and use a skill
/skill:code-reviewer ./rag/

# Get help for a skill
/skill:code-reviewer --help

# Force load even if pi thinks it's not needed
"Use the code-reviewer skill to audit this file"
```

---

## Customization

Add your own tools or modify existing ones:

### Add a new check to code-reviewer
Edit `.pi/skills/code-reviewer/scripts/review.py`:
```python
def check_custom_rule(file_path: Path) -> list[dict]:
    """Your custom check."""
    issues = []
    # Your logic here
    return issues
```

### Customize deployment strategy
Edit `.pi/skills/deployment-agent/scripts/deploy.sh`:
```bash
# Add your custom deployment logic
deploy_custom() {
    # Your deployment steps
}
```

---

## Best Practices

1. **Always run code-reviewer before deployment-agent** - The deployment agent will call it automatically, but explicit review is better

2. **Use canary deployments for risky changes** - Test with a small percentage first

3. **Keep rollback plan ready** - Know how to revert before you deploy

4. **Monitor after deployment** - Check health checks and metrics for 30 minutes post-deploy

5. **Never skip pre-deploy checks in production** - Use `--force` only in emergencies with approval

---

## Troubleshooting

### Skills not appearing
- Ensure `.pi/skills/` directory exists in project root
- Verify SKILL.md files have valid frontmatter (name, description)
- Run pi with `--verbose` to see skill loading messages

### Scripts not found
```bash
chmod +x .pi/skills/*/scripts/*
```

### Security tools not installed
```bash
# Check if tools are available
which bandit ruff mypy trivy

# Install missing ones
pip install ruff bandit mypy semgrep safety
brew install trivy
```

---

## Safety Notes

- **deployment-agent** will **refuse** to deploy if:
  - Critical security vulnerabilities detected
  - Tests are failing
  - Hardcoded secrets found
  - Git working directory is dirty
  - Required environment variables missing

- Use `--force` only after understanding and accepting risks

---

## Future Enhancements

Consider adding:
1. `test-agent` - Automated test generation and execution
2. `docs-agent` - Documentation generation and maintenance
3. `monitoring-agent` - Post-deployment monitoring and alerting
4. `migration-agent` - Database migration safety checker
5. `security-agent` - Deeper security analysis with SAST/DAST

---

## Related Documentation

- [Agent Skills Specification](https://agentskills.io/specification)
- [Pi Documentation](https://github.com/mariozechner/pi-coding-agent)
- [Anthropic Skills Examples](https://github.com/anthropics/skills)