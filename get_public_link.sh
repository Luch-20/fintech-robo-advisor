#!/bin/bash

# Script để lấy public URL và tạo message share

echo "🌐 Tạo Public URL để Share"
echo "=========================="
echo ""

# Check service
if ! curl -s http://localhost:5002/health > /dev/null; then
    echo "❌ Service chưa chạy. Đang start..."
    ./deploy_agent_service.sh start
    sleep 3
fi

echo "✅ Service đang chạy"
echo ""
echo "🚀 Đang start ngrok..."
echo ""
echo "📋 Sau khi ngrok chạy, bạn sẽ thấy URL như:"
echo "   Forwarding: https://abc123.ngrok.io -> http://localhost:5002"
echo ""
echo "💡 Copy URL đó và share cho bạn bè!"
echo ""
echo "📝 Ví dụ message để gửi:"
echo "   Base URL: https://abc123.ngrok.io/api"
echo ""
echo "   curl -X POST https://abc123.ngrok.io/api \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"investing\": [{\"ticker\": \"ACB\", \"amount\": 1000000}, {\"ticker\": \"BCM\", \"amount\": 2000000}]}'"
echo ""
echo "⚠️  Lưu ý: Nhấn Ctrl+C để dừng ngrok"
echo ""
echo "=========================="
echo ""

# Start ngrok
ngrok http 5002

