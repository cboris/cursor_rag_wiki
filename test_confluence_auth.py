#!/usr/bin/env python3
"""
Test script to debug Confluence authentication issues
"""

import requests
import yaml
import sys
from urllib.parse import urljoin

def test_basic_auth(base_url, username, password):
    """Test basic username/password authentication"""
    print(f"🔐 Testing basic authentication for {username}...")
    
    # Test basic auth
    auth = (username, password)
    test_url = urljoin(base_url, "/rest/api/space")
    
    try:
        response = requests.get(test_url, auth=auth, timeout=10)
        print(f"✅ Basic auth response: {response.status_code}")
        if response.status_code == 200:
            print("🎉 Basic authentication successful!")
            return True
        else:
            print(f"❌ Basic auth failed: {response.status_code} - {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Basic auth error: {e}")
        return False

def test_api_token(base_url, username, api_token):
    """Test API token authentication"""
    print(f"🔑 Testing API token authentication for {username}...")
    
    # Test API token auth
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }
    test_url = urljoin(base_url, "/rest/api/space")
    
    try:
        response = requests.get(test_url, headers=headers, timeout=10)
        print(f"✅ API token response: {response.status_code}")
        if response.status_code == 200:
            print("🎉 API token authentication successful!")
            return True
        else:
            print(f"❌ API token failed: {response.status_code} - {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ API token error: {e}")
        return False

def test_personal_access_token(base_url, username, pat):
    """Test Personal Access Token authentication"""
    print(f"🔐 Testing Personal Access Token for {username}...")
    
    # Test PAT auth (usually sent as Authorization header)
    headers = {
        'Authorization': f'Bearer {pat}',
        'Content-Type': 'application/json'
    }
    test_url = urljoin(base_url, "/rest/api/space")
    
    try:
        response = requests.get(test_url, headers=headers, timeout=10)
        print(f"✅ PAT response: {response.status_code}")
        if response.status_code == 200:
            print("🎉 Personal Access Token authentication successful!")
            return True
        else:
            print(f"❌ PAT failed: {response.status_code} - {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ PAT error: {e}")
        return False

def test_confluence_info(base_url):
    """Test if we can reach Confluence at all"""
    print(f"🌐 Testing connection to {base_url}...")
    
    try:
        response = requests.get(base_url, timeout=10)
        print(f"✅ Connection response: {response.status_code}")
        if response.status_code == 200:
            print("🎉 Can reach Confluence!")
            return True
        else:
            print(f"⚠️  Confluence responded with: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    # Load config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ config.yaml not found!")
        return
    
    confluence_config = config.get('confluence', {})
    base_url = confluence_config.get('base_url')
    username = confluence_config.get('username')
    api_token = confluence_config.get('api_token')
    
    print("🚀 Confluence Authentication Test")
    print("=" * 50)
    print(f"Base URL: {base_url}")
    print(f"Username: {username}")
    print(f"API Token: {api_token[:10]}..." if api_token else "No API token")
    print()
    
    # Test 1: Can we reach Confluence?
    if not test_confluence_info(base_url):
        print("\n❌ Cannot reach Confluence. Check your base_url and network connection.")
        return
    
    print()
    
    # Test 2: Try API token
    if api_token:
        test_api_token(base_url, username, api_token)
        print()
    
    # Test 3: Try Personal Access Token (if API token failed)
    if api_token:
        test_personal_access_token(base_url, username, api_token)
        print()
    
    print("\n💡 Troubleshooting Tips:")
    print("1. For on-prem Confluence, try using username/password instead of API token")
    print("2. Check if you need a Personal Access Token (PAT) instead of API token")
    print("3. Verify your user has 'Confluence Users' permission")
    print("4. Check if your Confluence instance requires specific authentication headers")
    print("5. Try accessing the Confluence web interface with the same credentials")

if __name__ == "__main__":
    main()
