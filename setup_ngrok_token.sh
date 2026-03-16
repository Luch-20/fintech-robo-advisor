#!/bin/bash

# Script để setup ngrok authtoken

echo "🔐 Setup Ngrok Authtoken"
echo "========================"
echo ""
echo "📋 Cách lấy token:"
echo "1. Đăng ký/Login: https://dashboard.ngrok.com"
echo "2. Vào: https://dashboard.ngrok.com/get-started/your-authtoken"
echo "3. Copy authtoken"
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ Ngrok chưa được cài đặt"
    echo ""
    echo "Cài đặt ngrok:"
    echo "  brew install ngrok  # macOS"
    echo "  hoặc download từ: https://ngrok.com/download"
    exit 1
fi

echo "✅ Ngrok đã được cài đặt"
echo ""

# Ask for token
read -p "📝 Paste ngrok authtoken của bạn: " token

if [ -z "$token" ]; then
    echo "❌ Token không được để trống"
    exit 1
fi

# Add authtoken
echo ""
echo "🔧 Đang lưu token..."
ngrok config add-authtoken "$token"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Token đã được lưu thành công!"
    echo ""
    echo "🚀 Bây giờ bạn có thể chạy:"
    echo "   ./get_public_link.sh"
    echo "   hoặc"
    echo "   ngrok http 5002"
else
    echo ""
    echo "❌ Lỗi khi lưu token. Vui lòng thử lại."
    exit 1
fi

