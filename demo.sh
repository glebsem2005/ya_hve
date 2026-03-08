#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# ForestGuard — Demo Deployment Script
# Останавливает старые контейнеры, пересобирает, запускает тесты
# ═══════════════════════════════════════════════════════════════════

# ── Цвета и стили ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# ── Красивые вставки ──────────────────────────────────────────────

banner() {
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║                                                          ║"
    echo "  ║     ForestGuard  —  Acoustic Forest Monitoring           ║"
    echo "  ║     Demo Deployment & Test Runner                        ║"
    echo "  ║                                                          ║"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

section() {
    local title="$1"
    local icon="$2"
    local color="$3"
    echo ""
    echo -e "${color}${BOLD}  ┌─────────────────────────────────────────────────────┐${NC}"
    echo -e "${color}${BOLD}  │ ${icon}  ${title}$(printf '%*s' $((47 - ${#title})) '')│${NC}"
    echo -e "${color}${BOLD}  └─────────────────────────────────────────────────────┘${NC}"
    echo ""
}

step() {
    echo -e "  ${CYAN}▸${NC} $1"
}

success() {
    echo -e "  ${GREEN}✔${NC} $1"
}

warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

fail() {
    echo -e "  ${RED}✘${NC} $1"
}

info() {
    echo -e "  ${DIM}$1${NC}"
}

separator() {
    echo -e "  ${DIM}──────────────────────────────────────────────────────${NC}"
}

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

cd "$(dirname "$0")"

banner

# ── 1. Показать текущее состояние ─────────────────────────────────

section "Current Containers" "📋" "$BLUE"

echo -e "  ${DIM}CONTAINER ID   IMAGE                 STATUS             NAMES${NC}"
separator
docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}" \
    | tail -n +2 \
    | while IFS= read -r line; do
        if echo "$line" | grep -q "ya_hve"; then
            echo -e "  ${YELLOW}${line}${NC}"
        else
            echo -e "  ${DIM}${line}${NC}"
        fi
    done
separator

# ── 2. Остановить ТОЛЬКО ya_hve контейнеры ───────────────────────

section "Stopping ForestGuard Containers" "🛑" "$RED"

YA_HVE_CONTAINERS=$(docker ps -q --filter "name=ya_hve" 2>/dev/null || true)

if [ -n "$YA_HVE_CONTAINERS" ]; then
    step "Found ForestGuard containers, stopping..."
    docker compose down --remove-orphans 2>/dev/null || docker-compose down --remove-orphans 2>/dev/null
    success "ForestGuard containers stopped"
else
    info "No running ForestGuard containers found"
fi

echo ""
warn "Amnezia containers left untouched:"
docker ps --format "  {{.Names}} ({{.Image}})" --filter "name=amnezia" 2>/dev/null || info "No amnezia containers running"

# ── 3. Пересборка образов ────────────────────────────────────────

section "Building Fresh Images" "🔨" "$MAGENTA"

step "Building Docker images (this may take a while)..."
echo ""
docker compose build --no-cache 2>&1 | while IFS= read -r line; do
    echo -e "  ${DIM}│${NC} ${line}"
done
echo ""
success "Images rebuilt successfully"

# ── 4. Запуск тестов в отдельном контейнере ──────────────────────

section "Running Test Suite" "🧪" "$CYAN"

step "Running pytest inside fresh container..."
separator
echo ""

# Запуск тестов с красивым выводом (verbose + цвета)
TEST_EXIT=0
docker compose run --rm --no-deps \
    -e TELEGRAM_BOT_TOKEN=test-token \
    -e YANDEX_API_KEY=test-key \
    -e YANDEX_FOLDER_ID=test-folder \
    -e MIC_MODE=sim \
    -e DEMO_SCENARIO=chainsaw \
    cloud \
    python -m pytest tests/ -v --tb=short --color=yes 2>&1 \
    | while IFS= read -r line; do
        # Подсветка PASSED / FAILED / ERROR
        if echo "$line" | grep -qE "PASSED"; then
            echo -e "  ${GREEN}│${NC} ${line}"
        elif echo "$line" | grep -qE "FAILED|ERROR"; then
            echo -e "  ${RED}│${NC} ${line}"
        elif echo "$line" | grep -qE "^tests/"; then
            echo -e "  ${CYAN}│${NC} ${line}"
        elif echo "$line" | grep -qE "passed|failed|error|warning"; then
            echo -e "  ${BOLD}│${NC} ${line}"
        else
            echo -e "  ${DIM}│${NC} ${line}"
        fi
    done || TEST_EXIT=$?

separator
echo ""

if [ "${TEST_EXIT}" -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}"
    echo "  ┌─────────────────────────────────────────────────────┐"
    echo "  │          ALL TESTS PASSED SUCCESSFULLY              │"
    echo "  └─────────────────────────────────────────────────────┘"
    echo -e "  ${NC}"
else
    echo -e "  ${RED}${BOLD}"
    echo "  ┌─────────────────────────────────────────────────────┐"
    echo "  │            SOME TESTS FAILED                        │"
    echo "  └─────────────────────────────────────────────────────┘"
    echo -e "  ${NC}"
fi

# ── 5. Запуск сервисов ───────────────────────────────────────────

section "Starting ForestGuard Services" "🚀" "$GREEN"

step "Starting cloud, edge, lora_gateway..."
docker compose up -d
echo ""

sleep 2

step "Service status:"
separator
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null \
    | while IFS= read -r line; do
        if echo "$line" | grep -q "Up"; then
            echo -e "  ${GREEN}│${NC} ${line}"
        else
            echo -e "  ${YELLOW}│${NC} ${line}"
        fi
    done
separator

# ── 6. Итог ──────────────────────────────────────────────────────

section "Deployment Complete" "🌲" "$GREEN"

echo -e "  ${BOLD}ForestGuard is ready!${NC}"
echo ""
echo -e "  ${CYAN}Dashboard:${NC}    http://localhost:8000"
echo -e "  ${CYAN}API Docs:${NC}     http://localhost:8000/docs"
echo -e "  ${CYAN}WebSocket:${NC}    ws://localhost:8000/ws"
echo -e "  ${CYAN}LoRa Gateway:${NC} tcp://localhost:9000"
echo ""
echo -e "  ${DIM}To trigger demo:${NC}"
echo -e "  ${BOLD}  curl -X POST http://localhost:8000/api/v1/demo -H 'Content-Type: application/json' -d '{\"scenario\":\"chainsaw\"}'${NC}"
echo ""
echo -e "  ${DIM}To view logs:${NC}"
echo -e "  ${BOLD}  docker compose logs -f${NC}"
echo ""
echo -e "  ${DIM}To stop:${NC}"
echo -e "  ${BOLD}  docker compose down${NC}"
echo ""
