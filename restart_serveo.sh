#!/bin/bash
# Script để restart Serveo tunnel

echo "======================================================================"
echo "🌐 RESTARTING SERVEO TUNNEL"
echo "======================================================================"

# Kill existing Serveo tunnels
echo ""
echo "🛑 Stopping existing Serveo tunnels..."
pkill -f "serveo.net" 2>/dev/null
sleep 2

# Check if service is running
if ! curl -s http://localhost:5002/health >/dev/null 2>&1; then
    echo ""
    echo "❌ Agent service is not running on port 5002"
    echo "   Please start it first:"
    echo "   source ~/miniforge3/etc/profile.d/conda.sh"
    echo "   conda activate base"
    echo "   python3 agent_service.py"
    exit 1
fi

echo "✅ Agent service is running"

# Start new tunnel
echo ""
echo "🚀 Starting Serveo tunnel..."
echo "   Forwarding: https://xxxx.serveo.net -> http://localhost:5002"
echo ""
echo "📝 Your public URL will be displayed below:"
echo "   (Press Ctrl+C to stop)"
echo ""

ssh -R 80:localhost:5002 serveo.net

