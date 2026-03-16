#!/bin/bash
# Script để start ngrok và hiển thị public URL

echo "🔍 Checking if service is running..."
if ! curl -s http://localhost:5002/health > /dev/null; then
    echo "❌ Service not running. Starting service..."
    ./deploy_agent_service.sh start
    sleep 3
fi

echo "✅ Service is running"
echo ""
echo "🌐 Starting ngrok..."
echo "   Public URL will be shown below"
echo "   Press Ctrl+C to stop"
echo ""

# Start ngrok
ngrok http 5002
