#!/bin/bash
# Setup script for DVC and pipeline configuration
# Run this script to initialize DVC and install hooks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "DVC Pipeline Setup Script"
echo "========================================"

cd "$PROJECT_ROOT"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed"
    exit 1
fi

echo "📦 Installing DVC and dependencies..."
pip install dvc dvc-s3 requests python-dotenv

# Initialize DVC if not already done
if [ ! -f ".dvc/.gitignore" ]; then
    echo "🔧 Initializing DVC..."
    dvc init
else
    echo "✅ DVC already initialized"
fi

# Track the data directory
echo "📊 Setting up data tracking..."
if [ ! -f "data.dvc" ]; then
    # Create a .dvc file to track the data directory
    dvc add data/train data/valid data/test 2>/dev/null || true
fi

# Install git hooks
echo "🪝 Installing git hooks..."
mkdir -p .git/hooks

if [ -f "scripts/hooks/post-commit" ]; then
    cp scripts/hooks/post-commit .git/hooks/post-commit
    chmod +x .git/hooks/post-commit
    echo "✅ Post-commit hook installed"
fi

# Create environment file template
if [ ! -f ".env" ]; then
    echo "📝 Creating .env template..."
    cat > .env.dvc << 'EOF'
# DVC and Jenkins Configuration
JENKINS_URL=http://localhost:8081
JENKINS_TOKEN=dvc-data-change-trigger
JENKINS_USER=admin
JENKINS_API_TOKEN=your-jenkins-api-token

# AWS S3 Configuration (for DVC remote storage)
# AWS_ACCESS_KEY_ID=your-access-key
# AWS_SECRET_ACCESS_KEY=your-secret-key
# S3_BUCKET=your-bucket-name
EOF
    echo "✅ Created .env.dvc template - configure your settings"
fi

# Verify DVC installation
echo ""
echo "========================================"
echo "Verification"
echo "========================================"
echo "DVC version: $(dvc --version)"
echo ""

# Show DVC status
echo "📈 DVC Status:"
dvc status || echo "No DVC pipeline configured yet"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Configure your .env.dvc file with Jenkins and AWS settings"
echo "2. Start Jenkins: cd jenkins && docker-compose up -d"
echo "3. Install 'Generic Webhook Trigger' plugin in Jenkins"
echo "4. Create a Jenkins pipeline job pointing to Jenkinsfile"
echo "5. Configure Jenkins credentials (github-token, aws-access-key-id, etc.)"
echo ""
echo "To test the pipeline:"
echo "  python scripts/trigger_jenkins.py --force"
echo ""
echo "To track data changes:"
echo "  dvc add data/train data/valid data/test"
echo "  git add data/*.dvc .gitignore"
echo "  git commit -m 'Track data with DVC'"
echo ""
