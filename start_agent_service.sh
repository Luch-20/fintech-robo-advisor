#!/bin/bash
# Script để start agent_service.py với đầy đủ dependencies

echo "======================================================================"
echo "🤖 STARTING AGENT SERVICE"
echo "======================================================================"

# Check if service is already running
if lsof -ti:5002 >/dev/null 2>&1; then
    echo "⚠️  Port 5002 is already in use"
    echo "   Killing existing process..."
    lsof -ti:5002 | xargs kill -9
    sleep 2
fi

# Try to activate conda environment if exists
if command -v conda &> /dev/null; then
    # Check for common conda env names
    if conda env list | grep -q "miniforge3\|base\|nckh"; then
        echo ""
        echo "📦 Activating conda environment..."
        eval "$(conda shell.bash hook)"
        # Try common environment names
        if conda env list | grep -q "miniforge3"; then
            conda activate miniforge3 2>/dev/null || true
        elif conda env list | grep -q "base"; then
            conda activate base 2>/dev/null || true
        fi
    fi
fi

# Check if required packages are installed
echo ""
echo "🔍 Checking dependencies..."
python3 -c "import flask" 2>/dev/null && echo "   ✅ Flask" || echo "   ❌ Flask - installing..."
python3 -c "import torch" 2>/dev/null && echo "   ✅ PyTorch" || echo "   ❌ PyTorch - missing"
python3 -c "import pandas" 2>/dev/null && echo "   ✅ Pandas" || echo "   ❌ Pandas - missing"

# Install missing basic packages if needed
if ! python3 -c "import flask" 2>/dev/null; then
    echo ""
    echo "📦 Installing Flask..."
    python3 -m pip install flask --quiet
fi

# Check if we can import from app.py
echo ""
echo "🔍 Checking imports..."
if python3 -c "from app import generate_recommendation" 2>&1 | grep -q "Error\|ModuleNotFoundError"; then
    echo "   ⚠️  Cannot import from app.py"
    echo "   → This is OK if dependencies are missing"
    echo "   → Service will start but may fail on first request"
else
    echo "   ✅ Imports OK"
fi

# Start service
echo ""
echo "🚀 Starting agent_service.py..."
echo "   Port: 5002"
echo "   Log: agent_service.log"
echo ""

cd "$(dirname "$0")"
nohup python3 agent_service.py > agent_service.log 2>&1 &
SERVICE_PID=$!

sleep 3

# Check if service started
if kill -0 $SERVICE_PID 2>/dev/null; then
    echo "✅ Service started (PID: $SERVICE_PID)"
    
    # Test health endpoint
    sleep 2
    if curl -s http://localhost:5002/health >/dev/null 2>&1; then
        echo "✅ Health check passed"
        echo ""
        echo "📡 Service is running at:"
        echo "   http://localhost:5002"
        echo "   http://localhost:5002/health"
        echo "   http://localhost:5002/api (POST)"
        echo "   http://localhost:5002/agent/analyze (POST)"
    else
        echo "⚠️  Service started but health check failed"
        echo "   Check agent_service.log for errors:"
        tail -10 agent_service.log
    fi
else
    echo "❌ Service failed to start"
    echo "   Check agent_service.log for errors:"
    tail -20 agent_service.log
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ AGENT SERVICE STARTED"
echo "======================================================================"
echo ""
echo "📝 Next steps:"
echo "   1. Test: curl http://localhost:5002/health"
echo "   2. Expose with Serveo: ssh -R 80:localhost:5002 serveo.net"
echo ""

