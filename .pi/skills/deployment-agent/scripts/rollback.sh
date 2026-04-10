#!/bin/bash
#
# Emergency Rollback Script
# Quickly revert to previous stable version
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMMEDIATE=false
VERSION=""

cat << "EOF"
${RED}
╔══════════════════════════════════════════════════════════╗
║              ⚠️  EMERGENCY ROLLBACK  ⚠️                   ║
║         Reverting to previous stable version             ║
╚══════════════════════════════════════════════════════════╝
${NC}
EOF

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
fatal() { log_error "$1"; exit 1; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --immediate)
            IMMEDIATE=true
            shift
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        *)
            fatal "Unknown option: $1"
            ;;
    esac
done

# Confirmation if not immediate
if [[ "$IMMEDIATE" == false ]]; then
    echo
    log_warning "This will rollback the production deployment!"
    read -p "Are you sure? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo "Rollback cancelled."
        exit 0
    fi
fi

log_info "Starting rollback at $(date)..."

# Get previous version
if [[ -z "$VERSION" ]]; then
    if [[ -f "${PROJECT_ROOT}/.last_deploy.txt" ]]; then
        VERSION=$(cat "${PROJECT_ROOT}/.last_deploy.txt")
    else
        fatal "No previous version found. Use --version to specify."
    fi
fi

log_info "Rolling back to: $VERSION"

# Stop current deployment
log_info "Stopping current deployment..."
docker-compose down || true

# Re-deploy previous version
log_info "Deploying previous version..."
export VERSION="$VERSION"
docker-compose up -d

# Wait for services
sleep 10

# Health check
log_info "Running health checks..."
HEALTH_URL="${HEALTH_CHECK_URL:-http://localhost:8000/health}"

if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    log_success "Rollback successful! Services are healthy."
else
    log_error "Rollback health check failed! Manual intervention required."
    exit 1
fi

echo
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              ROLLBACK COMPLETE                           ║"
echo "║              Version: $VERSION                           "
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
