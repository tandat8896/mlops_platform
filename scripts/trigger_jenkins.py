#!/usr/bin/env python3
"""
DVC Data Change Webhook Trigger

This script monitors DVC data changes and sends webhooks to Jenkins.
Can be used as a standalone trigger or integrated into CI/CD.

Usage:
    python scripts/trigger_jenkins.py [--force]
"""

import os
import sys
import json
import subprocess
import requests
from datetime import datetime
from pathlib import Path

# Configuration
JENKINS_URL = os.getenv("JENKINS_URL", "http://localhost:8081")
JENKINS_TOKEN = os.getenv("JENKINS_TOKEN", "dvc-data-change-trigger")
JENKINS_USER = os.getenv("JENKINS_USER", "admin")
JENKINS_API_TOKEN = os.getenv("JENKINS_API_TOKEN", "")

def get_git_info():
    """Get current git commit information."""
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            text=True
        ).strip()
        
        commit_message = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%B"], 
            text=True
        ).strip().split('\n')[0]
        
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
            text=True
        ).strip()
        
        return {
            "sha": commit_sha,
            "message": commit_message,
            "branch": branch
        }
    except subprocess.CalledProcessError as e:
        print(f"Error getting git info: {e}")
        return None


def check_dvc_status():
    """Check if DVC detects any changes."""
    try:
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True
        )
        
        # If output contains file paths, there are changes
        has_changes = bool(result.stdout.strip()) and "changed" in result.stdout.lower()
        return has_changes, result.stdout
    except FileNotFoundError:
        print("DVC is not installed. Please install with: pip install dvc")
        return False, ""


def check_data_files_changed():
    """Check if data files have changed in the last commit."""
    try:
        result = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
            text=True
        )
        
        changed_files = result.strip().split('\n')
        
        # Check for data-related file changes
        data_patterns = [
            'data/',
            '.dvc',
            'params.yaml',
            'dvc.yaml',
            'dvc.lock'
        ]
        
        for file in changed_files:
            for pattern in data_patterns:
                if pattern in file:
                    return True, changed_files
        
        return False, changed_files
    except subprocess.CalledProcessError:
        return False, []


def trigger_jenkins(git_info, force=False):
    """Send webhook to Jenkins to trigger training pipeline."""
    
    # Prepare payload
    payload = {
        "ref": f"refs/heads/{git_info['branch']}",
        "data_changed": "true",
        "commits": [
            {
                "id": git_info["sha"],
                "message": git_info["message"],
                "timestamp": datetime.now().isoformat()
            }
        ],
        "repository": {
            "full_name": "ThuanNaN/aio2025-mlops-cicd"
        },
        "forced": force
    }
    
    # Jenkins Generic Webhook Trigger URL
    webhook_url = f"{JENKINS_URL}/generic-webhook-trigger/invoke"
    
    headers = {
        "Content-Type": "application/json",
        "token": JENKINS_TOKEN
    }
    
    try:
        print(f"🚀 Sending webhook to Jenkins at {webhook_url}")
        response = requests.post(
            webhook_url,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json() if response.text else {}
            if result.get("triggered", False) or "triggered" in response.text.lower():
                print("✅ Jenkins pipeline triggered successfully!")
                return True
            else:
                print(f"⚠️ Jenkins response: {response.text}")
                return False
        else:
            print(f"❌ Jenkins trigger failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Jenkins at {JENKINS_URL}")
        print("   Make sure Jenkins is running and accessible")
        return False
    except Exception as e:
        print(f"❌ Error triggering Jenkins: {e}")
        return False


def main():
    """Main function to check for data changes and trigger Jenkins."""
    force = "--force" in sys.argv
    
    print("=" * 50)
    print("DVC Data Change Trigger")
    print("=" * 50)
    
    # Get git info
    git_info = get_git_info()
    if not git_info:
        print("❌ Failed to get git information")
        sys.exit(1)
    
    print(f"📌 Branch: {git_info['branch']}")
    print(f"📝 Commit: {git_info['sha'][:8]} - {git_info['message']}")
    
    if force:
        print("⚡ Force trigger enabled")
        trigger_jenkins(git_info, force=True)
        sys.exit(0)
    
    # Check for data changes
    data_changed, changed_files = check_data_files_changed()
    dvc_changed, dvc_status = check_dvc_status()
    
    if data_changed:
        print("📊 Data-related files changed in last commit:")
        for f in changed_files:
            if any(p in f for p in ['data/', '.dvc', 'params.yaml', 'dvc.yaml', 'dvc.lock']):
                print(f"   - {f}")
    
    if dvc_changed:
        print("📈 DVC status shows changes:")
        print(dvc_status)
    
    if data_changed or dvc_changed:
        success = trigger_jenkins(git_info)
        sys.exit(0 if success else 1)
    else:
        print("ℹ️ No data changes detected, skipping Jenkins trigger")
        sys.exit(0)


if __name__ == "__main__":
    main()
