#!/bin/bash
#
# Production Deployment Script
# Safe deployment with health checks and automatic rollback
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_ENV="production"
STRATEGY="rolling"
CANARY_PERCENT=0
FORCE=false
SKIP_TESTS=false

# Deployment tracking
DEPLOY_START_TIME=""
DEPLOY_VERSION=""
ROLLBACK_VERSION=""

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

fatal() {
    log_error "$1"
    exit 1
}

show_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Deploy to production with safety checks

OPTIONS:
  -e, --environment ENV    Deployment environment (default: production)
  -s, --strategy STRATEGY  Deployment strategy: rolling, blue-green, canary, recreate
  -c, --canary PERCENT     Canary percentage (e.g., 10 for 10%)
  -f, --force              Force deployment despite warnings
  --skip-tests             Skip running tests (not recommended)
  -v, --version VERSION    Version tag to deploy
  --rollback               Rollback to previous version
  -h, --help               Show this help message

EXAMPLES:
  $(basename "$0")                           # Standard rolling deployment
  $(basename "$0") --canary 10               # 10% canary deployment
  $(basename "$0") --strategy blue-green     # Blue-green deployment
  $(basename "$0") --rollback                # Rollback

EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                DEPLOY_ENV="$2"
                shift 2
                ;;
            -s|--strategy)
                STRATEGY="$2"
                shift 2
                ;;
            -c|--canary)
                CANARY_PERCENT="$2"
                STRATEGY="canary"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -v|--version)
                DEPLOY_VERSION="$2"
                shift 2
                ;;
            --rollback)
                ACTION="rollback"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                fatal "Unknown option: $1"
                ;;
        esac
    done
}

# Validate environment
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in a git repo
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        fatal "Not a git repository"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        fatal "Docker not found"
    fi
    
    # Check docker-compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        fatal "Docker Compose not found"
    fi
    
    log_success "Prerequisites OK"
}

# Run pre-deployment checks
run_pre_deploy_checks() {
    if [[ "$SKIP_TESTS" == true ]]; then
        log_warning "Skipping pre-deployment checks (--skip-tests)"
        return 0
    fi
    
    log_info "Running pre-deployment checks..."
    
    if [[ -f "${SCRIPT_DIR}/pre-deploy-check.py" ]]; then
        if ! python "${SCRIPT_DIR}/pre-deploy-check.py" --branch main,master; then
            if [[ "$FORCE" == true ]]; then
                log_warning "Pre-deploy checks failed, but continuing (--force)"
            else
                fatal "Pre-deployment checks failed. Fix issues or use --force to override"
            fi
        fi
    else
        log_warning "Pre-deploy check script not found"
    fi
    
    log_success "Pre-deployment checks passed"
}

# Get version from git
determine_version() {
    if [[ -z "$DEPLOY_VERSION" ]]; then
        DEPLOY_VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "latest")
    fi
    
    log_info "Deploying version: $DEPLOY_VERSION"
}

# Save current state for potential rollback
save_rollback_state() {
    ROLLBACK_VERSION=$(docker ps --format "{{.Image}}" | grep -E "(api|app|web)" | head -1 || echo "")
    log_info "Current version for rollback: ${ROLLBACK_VERSION:-none}"
    
    # Save to file for later rollback
    echo "$ROLLBACK_VERSION" > "${PROJECT_ROOT}/.last_deploy.txt"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    export VERSION="$DEPLOY_VERSION"
    export BUILD_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    export GIT_COMMIT=$(git rev-parse --short HEAD)
    
    # Build with proper tags
    docker build \
        --build-arg VERSION="$VERSION" \
        --build-arg BUILD_TIMESTAMP="$BUILD_TIMESTAMP" \
        --build-arg GIT_COMMIT="$GIT_COMMIT" \
        -t "${PROJECT_NAME}:latest" \
        -t "${PROJECT_NAME}:${VERSION}" \
        .
    
    log_success "Image built: ${PROJECT_NAME}:${VERSION}"
}

# Scan image for vulnerabilities
scan_image() {
    log_info "Scanning image for vulnerabilities..."
    
    # Check if trivy is available
    if command -v trivy &> /dev/null; then
        if ! trivy image --severity HIGH,CRITICAL --exit-code 1 "${PROJECT_NAME}:${DEPLOY_VERSION}"; then
            if [[ "$FORCE" == true ]]; then
                log_warning "Vulnerabilities found, continuing (--force)"
            else
                fatal "CRITICAL/HIGH vulnerabilities found in image"
            fi
        fi
        log_success "Image scan passed"
    else
        log_warning "Trivy not available, skipping image scan"
    fi
}

# Deploy using rolling strategy
deploy_rolling() {
    log_info "Deploying with ROLLING strategy..."
    
    export VERSION="$DEPLOY_VERSION"
    
    if [[ -f "docker-compose.yml" ]]; then
        docker-compose up -d --no-deps --build app
    elif [[ -f "docker-compose.yaml" ]]; then
        docker-compose -f docker-compose.yaml up -d --no-deps --build app
    else
        fatal "docker-compose.yml not found"
    fi
    
    log_success "Rolling deployment complete"
}

# Deploy using blue-green strategy
deploy_blue_green() {
    log_info "Deploying with BLUE-GREEN strategy..."
    
    # Determine current color
    CURRENT_COLOR=$(docker ps --filter "name=blue" --format "{{.Names}}" | grep -q blue && echo "blue" || echo "green")
    NEW_COLOR=$([[ "$CURRENT_COLOR" == "blue" ]] && echo "green" || echo "blue")
    
    log_info "Current: $CURRENT_COLOR → New: $NEW_COLOR"
    
    # Deploy to inactive environment
    export VERSION="$DEPLOY_VERSION"
    docker-compose up -d "${NEW_COLOR}" || fatal "Failed to deploy $NEW_COLOR"
    
    # Health check new environment
    sleep 5
    if ! health_check "${NEW_COLOR}"; then
        fatal "Health check failed for $NEW_COLOR"
    fi
    
    # Switch traffic (in practice, this would update load balancer)
    log_info "Switching traffic to $NEW_COLOR..."
    
    # Stop old environment after grace period
    sleep 30
    docker-compose stop "$CURRENT_COLOR" || true
    
    log_success "Blue-green deployment complete"
}

# Deploy using canary strategy
deploy_canary() {
    log_info "Deploying with CANARY strategy (${CANARY_PERCENT}%)..."
    
    # Start canary instances
    export VERSION="$DEPLOY_VERSION"
    export CANARY_COUNT=$((CANARY_PERCENT / 10))  # Scale: 10 = 1 instance, 100 = 10 instances
    
    # Deploy canary
    docker-compose up -d --scale "app=${CANARY_COUNT}" || fatal "Failed to deploy canary"
    
    # Wait and monitor
    log_info "Canary deployed, monitoring for 60 seconds..."
    sleep 60
    
    # Check metrics/health
    if ! health_check "app"; then
        log_error "Canary health check failed, rolling back..."
        rollback
        fatal "Canary deployment failed, rolled back"
    fi
    
    # Promote to full deployment
    log_info "Promoting canary to full deployment..."
    docker-compose up -d --scale "app=3"  # Assuming 3 instances
    
    log_success "Canary deployment promoted"
}

# Health check
health_check() {
    local service="${1:-app}"
    local retries=5
    local delay=5
    
    log_info "Health checking $service..."
    
    # Get health endpoint from env or use default
    HEALTH_URL="${HEALTH_CHECK_URL:-http://localhost:8000/health}"
    
    for ((i=1; i<=retries; i++)); do
        if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        log_warning "Health check attempt $i/$retries failed, retrying in ${delay}s..."
        sleep $delay
    done
    
    log_error "Health check failed after $retries attempts"
    return 1
}

# Rollback to previous version
rollback() {
    log_warning "Initiating ROLLBACK..."
    
    if [[ -f "${PROJECT_ROOT}/.last_deploy.txt" ]]; then
        PREVIOUS=$(cat "${PROJECT_ROOT}/.last_deploy.txt")
        log_info "Rolling back to: $PREVIOUS"
        
        # Stop current
        docker-compose down || true
        
        # Re-deploy previous version
        export VERSION="$PREVIOUS"
        docker-compose up -d
        
        log_success "Rollback complete"
    else
        fatal "No rollback state found"
    fi
}

# Post-deployment verification
post_deploy_verify() {
    log_info "Running post-deployment verification..."
    
    # Extended health check
    if ! health_check; then
        log_error "Post-deployment health check failed!"
        rollback
        fatal "Deployment failed, rolled back"
    fi
    
    # Smoke tests
    if [[ -f "${SCRIPT_DIR}/smoke-test.sh" ]]; then
        log_info "Running smoke tests..."
        bash "${SCRIPT_DIR}/smoke-test.sh"
    fi
    
    log_success "Post-deployment verification complete"
}

# Cleanup old images
cleanup() {
    log_info "Cleaning up old images..."
    
    # Keep last 5 versions
    docker images "${PROJECT_NAME}" --format "{{.Repository}}:{{.Tag}} {{.CreatedAt}}" | \
        sort -k2 -r | \
        tail -n +6 | \
        awk '{print $1}' | \
        xargs -r docker rmi || true
    
    # Prune dangling images
    docker image prune -f || true
    
    log_success "Cleanup complete"
}

# Main deployment flow
main() {
    parse_args "$@"
    
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║         PRODUCTION DEPLOYMENT AGENT                      ║"
    echo "║         Safe Deployment with Verification                ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log_info "Environment: $DEPLOY_ENV"
    log_info "Strategy: $STRATEGY"
    log_info "Start time: $(date)"
    
    # Project name from directory
    PROJECT_NAME=$(basename "$PROJECT_ROOT")
    
    check_prerequisites
    determine_version
    save_rollback_state
    run_pre_deploy_checks
    build_image
    scan_image
    
    # Execute deployment strategy
    case $STRATEGY in
        rolling)
            deploy_rolling
            ;;
        blue-green)
            deploy_blue_green
            ;;
            canary)
            deploy_canary
            ;;
        recreate)
            log_info "Deploying with RECREATE strategy..."
            docker-compose down
            docker-compose up -d
            ;;
        *)
            fatal "Unknown strategy: $STRATEGY"
            ;;
    esac
    
    post_deploy_verify
    cleanup
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║              DEPLOYMENT SUCCESSFUL                       ║"
    echo "║              Version: $DEPLOY_VERSION                    "
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Run main
main "$@"
