#!/bin/bash

# Simple Joy-Con Connection Script with Mode Selection
# Usage: ./joycon_connect.sh [right|left|both]
# This uses expect if available for better reliability

# Joy-Con MAC addresses
JOYCON_LEFT="A0:5A:5E:EA:C9:88"
JOYCON_RIGHT="A0:5A:5E:EB:45:D4"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"
}

show_help() {
    echo "Usage: $0 [MODE]"
    echo ""
    echo "MODE options:"
    echo "  right    - Connect right Joy-Con only"
    echo "  left     - Connect left Joy-Con only"
    echo "  both     - Connect both Joy-Cons (default)"
    echo ""
    echo "Examples:"
    echo "  $0 right        # Connect right Joy-Con only"
    echo "  $0 left         # Connect left Joy-Con only"
    echo "  $0 both         # Connect both Joy-Cons"
    echo "  $0              # Same as 'both'"
    echo ""
    exit 0
}

# Parse arguments
MODE="${1:-both}"  # Default to 'both' if no argument provided

case "$MODE" in
    right|left|both)
        # Valid mode
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        error "Invalid mode: $MODE"
        echo ""
        show_help
        ;;
esac

# Check if connected
is_connected() {
    bluetoothctl info "$1" 2>/dev/null | grep -q "Connected: yes"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "╔═══════════════════════════════════╗"
echo "║   Joy-Con Connection Manager      ║"
echo "╚═══════════════════════════════════╝"
echo ""

# Check for expect
if command -v expect &> /dev/null && [ -f "$SCRIPT_DIR/joycon_pair.expect" ]; then
    # log "Using expect-based pairing (more reliable)"
    USE_EXPECT=1
else
    # log "expect not found, using basic method"
    # log "Install expect for better reliability: sudo apt-get install expect"
    USE_EXPECT=0
fi

echo ""

# Connect function
connect_joycon() {
    local mac=$1
    local name=$2

    log "Connecting $name..."

    if is_connected "$mac"; then
        log "✓ $name already connected"
        return 0
    fi

    if [ $USE_EXPECT -eq 1 ]; then
        # Use expect script
        if "$SCRIPT_DIR/joycon_pair.expect" "$mac" "$name"; then
            return 0
        else
            error "Failed to connect $name with expect"
            return 1
        fi
    else
        # Basic method - just try to connect
        log "Press any button on $name now!"

        # Try direct connection
        timeout 10 bluetoothctl connect "$mac" &>/dev/null
        sleep 2

        if is_connected "$mac"; then
            log "✓ $name connected"
            return 0
        else
            error "✗ $name connection failed"
            error "Tip: You may need to pair manually using bluetoothctl:"
            error "  1. Run: bluetoothctl"
            error "  2. Type: scan on"
            error "  3. Press button on Joy-Con"
            error "  4. Type: pair $mac"
            error "  5. Type: trust $mac"
            error "  6. Type: connect $mac"
            return 1
        fi
    fi
}

# Determine which Joy-Cons to connect based on MODE
CONNECT_LEFT=0
CONNECT_RIGHT=0

case "$MODE" in
    left)
        CONNECT_LEFT=1
        ;;
    right)
        CONNECT_RIGHT=1
        ;;
    both)
        CONNECT_LEFT=1
        CONNECT_RIGHT=1
        ;;
esac

# Connect requested Joy-Cons
log "=== Connecting Joy-Cons (mode: $MODE) ==="
echo ""

if [ $CONNECT_LEFT -eq 1 ]; then
    connect_joycon "$JOYCON_LEFT" "Joy-Con (L)"
    echo ""
fi

if [ $CONNECT_RIGHT -eq 1 ]; then
    connect_joycon "$JOYCON_RIGHT" "Joy-Con (R)"
    echo ""
fi

# Summary
log "=== Connection Summary ==="

if [ $CONNECT_LEFT -eq 1 ]; then
    if is_connected "$JOYCON_LEFT"; then
        log "✓ Left Joy-Con:  Connected"
    else
        error "✗ Left Joy-Con:  Not connected"
    fi
fi

if [ $CONNECT_RIGHT -eq 1 ]; then
    if is_connected "$JOYCON_RIGHT"; then
        log "✓ Right Joy-Con: Connected"
    else
        error "✗ Right Joy-Con: Not connected"
    fi
fi

echo ""

log "Monitoring connections (Ctrl+C to stop)..."
echo ""

while true; do
    status_parts=()
    needs_reconnect=0

    if [ $CONNECT_LEFT -eq 1 ]; then
        if is_connected "$JOYCON_LEFT"; then
            left_status="✓"
        else
            left_status="✗"
            needs_reconnect=1
        fi
        status_parts+=("L[$left_status]")
    fi

    if [ $CONNECT_RIGHT -eq 1 ]; then
        if is_connected "$JOYCON_RIGHT"; then
            right_status="✓"
        else
            right_status="✗"
            needs_reconnect=1
        fi
        status_parts+=("R[$right_status]")
    fi

    echo -ne "\r${GREEN}Status:${NC} ${status_parts[*]} | $(date '+%H:%M:%S')  "

    # If any Joy-Con disconnected, attempt to reconnect (blocking)
    if [ $needs_reconnect -eq 1 ]; then
        echo ""  # New line after status

        if [ $CONNECT_LEFT -eq 1 ] && ! is_connected "$JOYCON_LEFT"; then
            log "Left disconnected! Reconnecting..."
            connect_joycon "$JOYCON_LEFT" "Joy-Con (L)"
        fi

        if [ $CONNECT_RIGHT -eq 1 ] && ! is_connected "$JOYCON_RIGHT"; then
            log "Right disconnected! Reconnecting..."
            connect_joycon "$JOYCON_RIGHT" "Joy-Con (R)"
        fi
    fi

    sleep 3
done


log "Done!"
