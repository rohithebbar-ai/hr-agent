# Security Review Checklist

## Critical Vulnerabilities (Must Fix)

### Injection Flaws
- [ ] SQL Injection: No string concatenation in SQL queries
  ```python
  # BAD
  query = f"SELECT * FROM users WHERE id = {user_id}"
  cursor.execute(query)
  
  # GOOD
  cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
  ```

- [ ] Command Injection: No shell execution with user input
  ```python
  # BAD
  os.system(f"ls {user_input}")
  
  # GOOD
  subprocess.run(["ls", user_input], shell=False)
  ```

- [ ] No-eval: No eval/exec with dynamic input
  ```python
  # BAD
  result = eval(user_input)
  
  # GOOD
  # Use ast.literal_eval for safe parsing
  ```

### Authentication Issues
- [ ] No hardcoded credentials
- [ ] Password hashing (bcrypt, scrypt, Argon2)
- [ ] Session tokens secure and rotated
- [ ] JWT secrets not in code

### Exposure of Sensitive Data
- [ ] No debug info in production
- [ ] Error messages don't leak stack traces
- [ ] .env files in .gitignore
- [ ] No secrets in logs

### Access Control
- [ ] Authorization checks on all endpoints
- [ ] No direct object references
- [ ] Principle of least privilege

## High Priority (Should Fix)

### Security Misconfiguration
- [ ] Default credentials changed
- [ ] Unnecessary features disabled
- [ ] Security headers configured
- [ ] Error handling doesn't leak info

### XSS Prevention
- [ ] Output encoding for HTML contexts
- [ ] Content-Security-Policy header
- [ ] No inline JavaScript

### Insecure Dependencies
- [ ] Dependencies updated
- [ ] No known CVEs in dependencies
- [ ] Minimal base images (distroless, alpine)

## Medium Priority (Nice to Fix)

### Logging
- [ ] Security events logged
- [ ] Failed authentication attempts logged
- [ ] No sensitive data in logs

### Cryptography
- [ ] Strong algorithms only (AES-256, SHA-256)
- [ ] No deprecated algorithms (MD5, SHA1)
- [ ] Random values from secure source (secrets, os.urandom)

### API Security
- [ ] Rate limiting configured
- [ ] Input validation
- [ ] API versioning
- [ ] Documentation doesn't expose sensitive endpoints

## Python-Specific

### Dangerous Patterns
```python
# NEVER use these:
pickle.loads(untrusted_data)  # Code execution
yaml.load(untrusted, Loader=yaml.Loader)  # Code execution
marshal.loads(untrusted)  # Code execution
xml.etree.ElementTree.parse(untrusted)  # XXE
```

### Safe Alternatives
```python
# Use these instead:
json.loads(data)  # Safe parsing
yaml.safe_load(data)  # Safe YAML
xml.etree.ElementTree.parse(StringIO(data), forbid_dtd=True)  # Defused XML
```

## Common False Positives

These patterns are usually safe:
- `password = os.getenv("DB_PASSWORD")` - Environment variables OK
- `hashlib.sha256(data).hexdigest()` - SHA-256 for integrity OK
- `hmac.compare_digest(a, b)` - Timing-safe comparison OK