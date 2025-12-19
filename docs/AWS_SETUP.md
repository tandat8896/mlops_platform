# AWS EC2 Security Group Setup

## Mở Port 8000 cho API

### Cách 1: Từ EC2 Instance Details

1. Vào **EC2 Console** → **Instances**
2. Chọn instance của bạn (`i-03f89a0b86d556c17`)
3. Click tab **Security**
4. Click vào Security Group name: **`launch-wizard-2`** (hoặc ID `sg-0a8178a857143bdde`)
5. Chọn tab **Inbound rules**
6. Click **Edit inbound rules**
7. Click **Add rule**:
   - **Type**: Custom TCP
   - **Port range**: 8000
   - **Source**: `0.0.0.0/0` (cho phép từ mọi nơi) hoặc IP cụ thể của bạn
   - **Description**: "FastAPI Application Port"
8. Click **Save rules**

### Cách 2: Từ Security Groups Menu

1. Vào **EC2 Console** → **Security Groups** (menu bên trái)
2. Tìm và chọn Security Group: **`launch-wizard-2`** (ID: `sg-0a8178a857143bdde`)
3. Chọn tab **Inbound rules**
4. Click **Edit inbound rules**
5. Thêm rule như trên
6. Click **Save rules**

## Kiểm tra

Sau khi thêm rule, test API:

```bash
# Health check
curl http://13.212.160.80:8000/health

# Hoặc mở trình duyệt
http://13.212.160.80:8000/health
```

## Security Best Practices

⚠️ **Lưu ý**: Mở port `0.0.0.0/0` cho phép truy cập từ mọi nơi. Để bảo mật hơn:

- Chỉ mở cho IP cụ thể của bạn
- Hoặc sử dụng VPN/SSH tunnel
- Hoặc thêm authentication layer (API key, JWT, etc.)

## Ports cần mở

- **Port 22**: SSH (đã có)
- **Port 80**: HTTP (đã có)
- **Port 8000**: FastAPI Application (cần thêm)
