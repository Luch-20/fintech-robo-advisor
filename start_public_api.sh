#!/bin/bash
# Script để start Flask và expose public URL

echo "======================================================================"
echo "🚀 STARTING PUBLIC API"
echo "======================================================================"

# Check if Flask is running
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 5001 is already in use"
    echo "   Please stop the existing Flask server first"
    exit 1
fi

# Start Flask in background
echo ""
echo "1️⃣  Starting Flask server..."
python3 app.py &
FLASK_PID=$!
echo "   ✅ Flask started (PID: $FLASK_PID)"
echo "   Waiting for Flask to be ready..."
sleep 3

# Check if Flask started successfully
if ! kill -0 $FLASK_PID 2>/dev/null; then
    echo "   ❌ Flask failed to start"
    exit 1
fi

echo ""
echo "2️⃣  Choose tunneling method:"
echo "   1) Serveo (SSH tunnel - no install needed)"
echo "   2) Cloudflare Tunnel (need: brew install cloudflared)"
echo ""
read -p "   Enter choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "3️⃣  Starting Serveo tunnel..."
        echo "   URL will be displayed below:"
        echo ""
        ssh -R 80:localhost:5001 serveo.net
        ;;
    2)
        if ! command -v cloudflared &> /dev/null; then
            echo ""
            echo "❌ cloudflared not installed"
            echo "   Install with: brew install cloudflared"
            kill $FLASK_PID
            exit 1
        fi
        echo ""
        echo "3️⃣  Starting Cloudflare tunnel..."
        echo "   URL will be displayed below:"
        echo ""
        cloudflared tunnel --url http://localhost:5001
        ;;
    *)
        echo "❌ Invalid choice"
        kill $FLASK_PID
        exit 1
        ;;
esac

# Cleanup on exit
trap "kill $FLASK_PID 2>/dev/null" EXIT
