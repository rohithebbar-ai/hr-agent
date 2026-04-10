# Pre-Deployment Checklist

## Security Checklist

### Code Security
- [ ] No hardcoded secrets (passwords, API keys, tokens)
- [ ] No SQL injection vulnerabilities
- [ ] No XSS vulnerabilities
- [ ] Input validation on all user inputs
- [ ] No eval() or exec() with user input
- [ ] Dependencies scanned for CVEs (CRITICAL/HIGH = block)

### Infrastructure Security
- [ ] Non-root user in Docker container
- [ ] No sensitive data in environment variables (use secrets manager)
- [ ] HTTPS only for production
- [ ] Security headers configured (HSTS, CSP, etc.)
- [ ] Rate limiting enabled
- [ ] CORS properly configured

### Authentication/Authorization
- [ ] JWT tokens have short expiration
- [ ] Password requirements enforced
- [ ] 2FA for admin accounts
- [ ] Session management secure
- [ ] Principle of least privilege

## Deployment Checklist

### Pre-Deploy
- [ ] All tests passing
- [ ] Code coverage >= 80%
- [ ] Security scan passed
- [ ] Dependency audit clean (no CRITICAL/HIGH CVEs)
- [ ] Database migrations tested
- [ ] Configuration validated
- [ ] Rollback plan documented

### Build
- [ ] Docker image builds successfully
- [ ] Image scanned for vulnerabilities
- [ ] Image size optimized (multi-stage build)
- [ ] Proper labels/tags applied

### Database
- [ ] Backup created before migration
- [ ] Migration tested in staging
- [ ] Rollback script tested
- [ ] No destructive changes without approval

### Deploy
- [ ] Health checks configured
- [ ] Monitoring alerts configured
- [ ] Log aggregation working
- [ ] Graceful shutdown configured
- [ ] Circuit breakers configured

### Post-Deploy
- [ ] Smoke tests passed
- [ ] Key metrics nominal
- [ ] Error rates acceptable
- [ ] Response times acceptable
- [ ] No critical alerts

## Monitoring Verification

- [ ] Logs are being collected
- [ ] Error tracking (Sentry) active
- [ ] Performance metrics (APM) visible
- [ ] Database metrics visible
- [ ] Custom business metrics active
- [ ] Alerts routed to correct channels

## Rollback Scenarios

**Automatic Rollback Triggers:**
- Error rate > 5%
- P99 latency > 2x baseline
- Health check failures > 3 in 5 minutes
- Database connection failures

**Manual Rollback Required:**
- Data corruption detected
- Security breach suspected
- Compliance violations
- Business logic errors affecting revenue

## Sign-Off

| Role | Name | Date | Sign-off |
|------|------|------|----------|
| Developer | | | |
| Security | | | |
| QA | | | |