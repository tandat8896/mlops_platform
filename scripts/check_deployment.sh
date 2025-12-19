#!/bin/bash
# Script to check deployment status on EC2

echo "=== Checking Docker installation ==="
if command -v docker &> /dev/null; then
    echo "✓ Docker found: $(docker --version)"
else
    echo "✗ Docker not found"
    exit 1
fi

echo ""
echo "=== Checking container status ==="
docker ps -a | grep app-container || echo "✗ Container 'app-container' not found"

echo ""
echo "=== Checking if container is running ==="
if docker ps | grep -q app-container; then
    echo "✓ Container is running"
else
    echo "✗ Container is NOT running"
    echo "Attempting to start container..."
    docker start app-container || echo "Failed to start container"
fi

echo ""
echo "=== Checking container logs (last 20 lines) ==="
docker logs --tail 20 app-container 2>&1 || echo "Could not get logs"

echo ""
echo "=== Checking port 8000 ==="
if sudo netstat -tlnp 2>/dev/null | grep -q ":8000"; then
    echo "✓ Port 8000 is listening"
    sudo netstat -tlnp 2>/dev/null | grep ":8000"
elif sudo ss -tlnp 2>/dev/null | grep -q ":8000"; then
    echo "✓ Port 8000 is listening"
    sudo ss -tlnp 2>/dev/null | grep ":8000"
else
    echo "✗ Port 8000 is NOT listening"
fi

echo ""
echo "=== Testing local connection ==="
if curl -s -f http://localhost:8000/health > /dev/null; then
    echo "✓ Local health check passed"
    curl -s http://localhost:8000/health
else
    echo "✗ Local health check failed"
fi

echo ""
echo "=== Checking Security Group (requires AWS CLI) ==="
if command -v aws &> /dev/null; then
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)
    if [ ! -z "$INSTANCE_ID" ]; then
        echo "Instance ID: $INSTANCE_ID"
        aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].SecurityGroups[*].[GroupId,GroupName]' --output table 2>/dev/null || echo "Could not get Security Group info"
    fi
else
    echo "AWS CLI not installed, skipping Security Group check"
fi

echo ""
echo "=== Summary ==="
echo "1. Use HTTP (not HTTPS): http://13.212.160.80:8000/health"
echo "2. Ensure Security Group allows port 8000 (TCP) from your IP or 0.0.0.0/0"
echo "3. Check container logs above for any errors"
