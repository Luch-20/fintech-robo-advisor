#!/bin/bash

# Script để deploy Agent Service
# Usage: ./deploy_agent_service.sh [start|stop|restart|status]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=5002
SERVICE_NAME="agent_service"
LOG_FILE="agent_service.log"
PID_FILE="agent_service.pid"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
start_service() {
    echo "🚀 Starting Agent Service..."
    
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  Service already running on port $PORT (PID: $PID)${NC}"
            return 1
        else
            rm -f "$PID_FILE"
        fi
    fi
    
    # Check if port is in use
    if lsof -ti:$PORT > /dev/null 2>&1; then
        echo -e "${RED}❌ Port $PORT is already in use${NC}"
        echo "   Trying to kill existing process..."
        lsof -ti:$PORT | xargs kill -9 2>/dev/null
        sleep 2
    fi
    
    # Start service in background
    nohup python3 agent_service.py > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    # Wait a bit and check if it's running
    sleep 3
    
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service started successfully!${NC}"
        echo "   PID: $PID"
        echo "   Port: $PORT"
        echo "   Log: $LOG_FILE"
        echo ""
        echo "📡 Endpoints:"
        echo "   Health: http://localhost:$PORT/health"
        echo "   API: http://localhost:$PORT/api"
        echo "   Agent: http://localhost:$PORT/agent/analyze"
        
        # Test health check
        sleep 2
        if curl -s http://localhost:$PORT/health > /dev/null; then
            echo -e "\n${GREEN}✅ Health check passed!${NC}"
        else
            echo -e "\n${YELLOW}⚠️  Health check failed, but service might still be starting...${NC}"
        fi
    else
        echo -e "${RED}❌ Service failed to start${NC}"
        echo "   Check log: tail -20 $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_service() {
    echo "🛑 Stopping Agent Service..."
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID 2>/dev/null
            sleep 2
            if ps -p $PID > /dev/null 2>&1; then
                kill -9 $PID 2>/dev/null
            fi
            echo -e "${GREEN}✅ Service stopped${NC}"
        else
            echo -e "${YELLOW}⚠️  Service not running (PID file exists but process not found)${NC}"
        fi
        rm -f "$PID_FILE"
    else
        # Try to kill by port
        if lsof -ti:$PORT > /dev/null 2>&1; then
            lsof -ti:$PORT | xargs kill -9 2>/dev/null
            echo -e "${GREEN}✅ Service stopped (killed by port)${NC}"
        else
            echo -e "${YELLOW}⚠️  Service not running${NC}"
        fi
    fi
}

restart_service() {
    echo "🔄 Restarting Agent Service..."
    stop_service
    sleep 2
    start_service
}

status_service() {
    echo "📊 Agent Service Status"
    echo "======================"
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "Status: ${GREEN}Running${NC}"
            echo "PID: $PID"
            echo "Port: $PORT"
            
            # Test health
            if curl -s http://localhost:$PORT/health > /dev/null; then
                echo -e "Health: ${GREEN}Healthy${NC}"
                echo ""
                echo "Response:"
                curl -s http://localhost:$PORT/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:$PORT/health
            else
                echo -e "Health: ${RED}Unhealthy${NC}"
            fi
        else
            echo -e "Status: ${RED}Not Running (stale PID file)${NC}"
            rm -f "$PID_FILE"
        fi
    else
        if lsof -ti:$PORT > /dev/null 2>&1; then
            PID=$(lsof -ti:$PORT | head -1)
            echo -e "Status: ${YELLOW}Running (no PID file, PID: $PID)${NC}"
            echo "Port: $PORT"
        else
            echo -e "Status: ${RED}Not Running${NC}"
        fi
    fi
    
    echo ""
    echo "Recent logs:"
    tail -5 "$LOG_FILE" 2>/dev/null || echo "No log file"
}

# Main
case "${1:-start}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the service"
        echo "  stop    - Stop the service"
        echo "  restart - Restart the service"
        echo "  status  - Show service status"
        exit 1
        ;;
esac

