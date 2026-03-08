#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# ForestGuard — Server Cleanup Script
# Очистка места на сервере от Docker-мусора и тяжёлых файлов
# ═══════════════════════════════════════════════════════════════════

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

section() {
    echo ""
    echo -e "${1}${BOLD}  ┌─────────────────────────────────────────────────────┐${NC}"
    echo -e "${1}${BOLD}  │ ${2}$(printf '%*s' $((49 - ${#2})) '')│${NC}"
    echo -e "${1}${BOLD}  └─────────────────────────────────────────────────────┘${NC}"
    echo ""
}

step()    { echo -e "  ${CYAN}▸${NC} $1"; }
success() { echo -e "  ${GREEN}✔${NC} $1"; }
warn()    { echo -e "  ${YELLOW}⚠${NC} $1"; }
info()    { echo -e "  ${DIM}$1${NC}"; }
separator() { echo -e "  ${DIM}──────────────────────────────────────────────────────${NC}"; }

bytes_to_human() {
    local bytes=$1
    if   [ "$bytes" -ge 1073741824 ]; then echo "$(( bytes / 1073741824 )) GB"
    elif [ "$bytes" -ge 1048576 ];    then echo "$(( bytes / 1048576 )) MB"
    elif [ "$bytes" -ge 1024 ];       then echo "$(( bytes / 1024 )) KB"
    else echo "${bytes} B"
    fi
}

echo ""
echo -e "${RED}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║                                                          ║"
echo "  ║       ForestGuard — Server Disk Cleanup                  ║"
echo "  ║                                                          ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 0. Текущее состояние ─────────────────────────────────────────

section "$BLUE" "📊 Current Disk Usage"

df -h / | tail -1 | awk '{printf "  Total: %s | Used: %s (%s) | Free: %s\n", $2, $3, $5, $4}'
echo ""
step "Docker disk usage:"
docker system df 2>/dev/null || warn "Cannot access Docker"
separator

# ── 1. Dangling images (неиспользуемые слои) ─────────────────────

section "$YELLOW" "🗑️  Step 1: Remove Dangling Images"

DANGLING=$(docker images -f "dangling=true" -q 2>/dev/null | wc -l)
if [ "$DANGLING" -gt 0 ]; then
    step "Found ${DANGLING} dangling image(s)"
    docker image prune -f
    success "Dangling images removed"
else
    info "No dangling images found"
fi

# ── 2. Остановленные контейнеры ──────────────────────────────────

section "$YELLOW" "🧹 Step 2: Remove Stopped Containers"

STOPPED=$(docker ps -a -f "status=exited" -q 2>/dev/null | wc -l)
if [ "$STOPPED" -gt 0 ]; then
    step "Found ${STOPPED} stopped container(s)"
    docker container prune -f
    success "Stopped containers removed"
else
    info "No stopped containers found"
fi

# ── 3. Неиспользуемые volumes ────────────────────────────────────

section "$YELLOW" "💾 Step 3: Remove Unused Volumes"

VOLUMES=$(docker volume ls -f "dangling=true" -q 2>/dev/null | wc -l)
if [ "$VOLUMES" -gt 0 ]; then
    step "Found ${VOLUMES} unused volume(s)"
    docker volume prune -f
    success "Unused volumes removed"
else
    info "No unused volumes found"
fi

# ── 4. Build cache ───────────────────────────────────────────────

section "$YELLOW" "📦 Step 4: Clear Docker Build Cache"

step "Clearing builder cache..."
docker builder prune -f --all 2>/dev/null && success "Build cache cleared" || info "No build cache to clear"

# ── 5. Неиспользуемые образы (НЕ dangling) ──────────────────────

section "$RED" "🔴 Step 5: Remove Unused Images (aggressive)"

echo -e "  ${YELLOW}This removes ALL images not used by running containers.${NC}"
echo -e "  ${YELLOW}Amnezia images will be kept if containers are running.${NC}"
echo ""

read -p "  Proceed? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker image prune -a -f
    success "Unused images removed"
else
    info "Skipped"
fi

# ── 6. Поиск тяжёлых файлов на диске ────────────────────────────

section "$BLUE" "🔍 Step 6: Large Files on Disk"

step "Top 15 largest files (>50MB):"
separator
find / -xdev -type f -size +50M 2>/dev/null \
    | head -20 \
    | while read -r f; do
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo 0)
        HUMAN=$(bytes_to_human "$SIZE")
        printf "  %-8s %s\n" "$HUMAN" "$f"
    done
separator

step "Directory sizes in /var:"
du -sh /var/log /var/cache /var/lib/docker /var/tmp 2>/dev/null | while read -r line; do
    echo "  $line"
done
separator

# ── 7. Очистка логов ─────────────────────────────────────────────

section "$YELLOW" "📝 Step 7: Truncate Docker Logs"

step "Container log sizes:"
find /var/lib/docker/containers -name "*.log" -type f 2>/dev/null | while read -r logfile; do
    SIZE=$(stat -c%s "$logfile" 2>/dev/null || echo 0)
    HUMAN=$(bytes_to_human "$SIZE")
    CONTAINER=$(basename "$(dirname "$logfile")" | cut -c1-12)
    printf "  %-8s %s\n" "$HUMAN" "$CONTAINER"
done
separator

read -p "  Truncate all container logs? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    find /var/lib/docker/containers -name "*.log" -type f -exec truncate -s 0 {} \; 2>/dev/null
    success "Container logs truncated"
else
    info "Skipped"
fi

# ── 8. Очистка apt cache ─────────────────────────────────────────

section "$YELLOW" "📦 Step 8: Clear System Caches"

if command -v apt-get &>/dev/null; then
    step "Clearing apt cache..."
    apt-get clean 2>/dev/null && success "apt cache cleared" || info "Skipped (no root?)"
fi

if [ -d /tmp ]; then
    step "Clearing old temp files (>7 days)..."
    find /tmp -type f -mtime +7 -delete 2>/dev/null
    success "Old temp files cleaned"
fi

# ── 9. Итог ──────────────────────────────────────────────────────

section "$GREEN" "✅ Cleanup Complete"

echo -e "  ${BOLD}Disk usage after cleanup:${NC}"
df -h / | tail -1 | awk '{printf "  Total: %s | Used: %s (%s) | Free: %s\n", $2, $3, $5, $4}'
echo ""
step "Docker usage after cleanup:"
docker system df 2>/dev/null
echo ""

echo -e "  ${DIM}If still low on space, consider:${NC}"
echo -e "  ${BOLD}  docker system prune -a --volumes${NC}  ${DIM}(nuclear option — removes EVERYTHING unused)${NC}"
echo ""
