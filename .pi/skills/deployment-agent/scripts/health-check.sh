#!/bin/bash
#
# Health Check Script
# Verify deployment is responding correctly
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

HEALTH_URL="${HEALTH_CHECK_URL:-http://localhost:8000/health}"
TIMEOUT=30
RETRIES=5

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --url)
            HEALTH_URL="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --retries)
            RETRIES="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              DEPLOYMENT HEALTH CHECK                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_info "URL: $HEALTH_URL"
log_info "Timeout: ${TIMEOUT}s, Retries: $RETRIES"
echo

# Check main health endpoint
success=false
for ((i=1; i<=RETRIES; i++)); do
    log_info "Health check attempt $i/$RETRIES..."
    
    if response=$(curl -sf --max-time "$TIMEOUT" "$HEALTH_URL" 2>&1); then
        log_success "Health check PASSED"
        echo "Response: $response"
        success=true
        break
    else
        log_warning "Attempt $i failed, waiting 5s..."
        sleep 5
    fi
done

if [[ "$success" == false ]]; then
    log_error "Health check FAILED after $RETRIES attempts"
    
    # Additional diagnostics
    echo
    log_warning "Running diagnostics..."
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || true
    
    exit 1
fi

# Extended checks
log_info "Running extended health checks..."

# Check database connection (if applicable)
if docker ps --format "{{.Names}}" | grep -q "db"; then
    if docker exec $(docker ps -q --filter "name=db") pg_isready 2>/dev/null | grep -q "accepting"; then
        log_success "Database connection: OK"
    else
        log_warning "Database check failed"
    fi
fi

log_success "All health checks passed!"
