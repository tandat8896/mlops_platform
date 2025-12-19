#!/bin/bash
# Debug script for DVC push issues
# Usage: bash debug_dvc_push.sh

echo "=== Debugging DVC Push ==="
echo ""

# Load credentials
if [ -f .env ]; then
    echo "1. Loading credentials..."
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | tr -d '\r')
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        if [[ "$line" =~ ^[[:space:]]*[A-Z_]+= ]]; then
            line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            export "$line"
        fi
    done < .env
    export AWS_DEFAULT_REGION="us-east-1"
    echo "   ✓ Credentials loaded"
    echo ""
fi

# Check credentials
echo "2. Checking credentials..."
echo "   Access Key: ${AWS_ACCESS_KEY_ID:0:15}..."
echo "   Region: $AWS_DEFAULT_REGION"
echo ""

# Check DVC config
echo "3. Checking DVC config..."
dvc remote list
echo ""

# Check what needs to be pushed
echo "4. Checking DVC status..."
dvc status
echo ""

# Test S3 permissions in detail
echo "5. Testing S3 permissions..."
echo "   Testing ListBucket..."
aws s3 ls s3://s3-cicd-1-demo/ --region us-east-1
echo ""

echo "   Testing PutObject (creating test file)..."
echo "test" | aws s3 cp - s3://s3-cicd-1-demo/dvc-storage/test-permissions.txt --region us-east-1 2>&1
PUT_EXIT=$?
if [ $PUT_EXIT -eq 0 ]; then
    echo "   ✓ PutObject permission OK"
    # Clean up
    aws s3 rm s3://s3-cicd-1-demo/dvc-storage/test-permissions.txt --region us-east-1 > /dev/null 2>&1
else
    echo "   ✗ PutObject permission FAILED"
fi
echo ""

echo "   Testing HeadObject (checking if file exists)..."
aws s3api head-object --bucket s3-cicd-1-demo --key dvc-storage/.gitignore --region us-east-1 2>&1 | head -5
HEAD_EXIT=$?
if [ $HEAD_EXIT -eq 0 ] || [[ $(aws s3api head-object --bucket s3-cicd-1-demo --key dvc-storage/.gitignore --region us-east-1 2>&1) == *"404"* ]]; then
    echo "   ✓ HeadObject permission OK (404 is OK if file doesn't exist)"
else
    echo "   ✗ HeadObject permission FAILED"
fi
echo ""

# Try DVC push with verbose output
echo "6. Attempting DVC push (verbose)..."
dvc push -v 2>&1 | head -30

