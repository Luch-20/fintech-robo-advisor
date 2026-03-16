#!/bin/bash

echo "🛑 Dừng service cũ..."
if lsof -ti:5002 >/dev/null 2>&1; then
    lsof -ti:5002 | xargs kill -9 2>/dev/null
    sleep 2
    echo "✅ Đã dừng process trên port 5002"
else
    echo "ℹ️  Không có process nào đang chạy trên port 5002"
fi
echo ""
echo "🚀 Khởi động lại service..."
echo ""

# Start service trong background
nohup python3 agent_service.py > agent_service.log 2>&1 &

# Đợi service khởi động
sleep 3

echo "🔍 Kiểm tra service..."
if curl -s http://localhost:5002/health > /dev/null 2>&1; then
    echo "✅ Service đã chạy thành công!"
    echo ""
    echo "📊 Thông tin service:"
    echo "   URL: http://localhost:5002"
    echo "   Health: http://localhost:5002/health"
    echo "   API: http://localhost:5002/api"
    echo ""
    echo "📝 Xem log: tail -f agent_service.log"
    echo ""
    echo "🧪 Test API:"
    echo "   curl -X POST http://localhost:5002/api \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"investing\": [{\"ticker\": \"ACB\", \"amount\": 1000000}, {\"ticker\": \"BCM\", \"amount\": 2000000}]}'"
else
    echo "❌ Service chưa chạy được. Kiểm tra log:"
    echo "   tail -20 agent_service.log"
fi

