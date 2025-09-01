#!/usr/bin/env python3
"""
Detailed Confluence API test to identify specific 401 issues
"""

import requests
import yaml
import json
from urllib.parse import urljoin

def test_endpoint(base_url, endpoint, auth_method, auth_value, description):
    """Test a specific Confluence endpoint"""
    print(f"🔍 Testing: {description}")
    print(f"   Endpoint: {endpoint}")
    
    if auth_method == "token":
        headers = {
            'Authorization': f'Bearer {auth_value}',
            'Content-Type': 'application/json'
        }
        response = requests.get(urljoin(base_url, endpoint), headers=headers, timeout=10)
    elif auth_method == "basic":
        auth = (auth_value['username'], auth_value['password'])
        response = requests.get(urljoin(base_url, endpoint), auth=auth, timeout=10)
    
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        print("   ✅ Success!")
        try:
            data = response.json()
            if 'results' in data:
                print(f"   📊 Results count: {len(data.get('results', []))}")
        except:
            pass
    elif response.status_code == 401:
        print("   ❌ 401 Unauthorized")
        print(f"   📝 Response: {response.text[:200]}")
    elif response.status_code == 403:
        print("   ❌ 403 Forbidden")
        print(f"   📝 Response: {response.text[:200]}")
    else:
        print(f"   ⚠️  Unexpected status: {response.status_code}")
        print(f"   📝 Response: {response.text[:200]}")
    
    print()

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    confluence_config = config.get('confluence', {})
    base_url = confluence_config.get('base_url')
    username = confluence_config.get('username')
    api_token = confluence_config.get('api_token')
    space_key = confluence_config.get('space_key')
    
    print("🔍 Detailed Confluence API Test")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print(f"Username: {username}")
    print(f"Space Key: {space_key}")
    print()
    
    # Test various endpoints that might be used by the RAG system
    endpoints_to_test = [
        # Basic space info
        (f"/rest/api/space/{space_key}", "token", api_token, f"Space info for {space_key}"),
        
        # Space content
        (f"/rest/api/content?spaceKey={space_key}&type=page&limit=5", "token", api_token, f"Pages in space {space_key}"),
        
        # All spaces (to check general access)
        ("/rest/api/space?limit=5", "token", api_token, "List all accessible spaces"),
        
        # User info
        (f"/rest/api/user?username={username}", "token", api_token, f"User info for {username}"),
        
        # Content by type
        (f"/rest/api/content?spaceKey={space_key}&type=page&status=current&limit=5", "token", api_token, f"Current pages in {space_key}"),
        
        # Space permissions
        (f"/rest/api/space/{space_key}/permission", "token", api_token, f"Permissions for space {space_key}"),
        
        # Content search
        (f"/rest/api/content/search?cql=space={space_key}&limit=5", "token", api_token, f"Search content in {space_key}"),
    ]
    
    print("🧪 Testing various Confluence endpoints...")
    print()
    
    for endpoint, auth_method, auth_value, description in endpoints_to_test:
        if auth_method == "token":
            test_endpoint(base_url, endpoint, auth_method, auth_value, description)
        else:
            test_endpoint(base_url, endpoint, auth_method, auth_value, description)
    
    print("💡 Analysis:")
    print("• If space-specific endpoints return 401, check your space permissions")
    print("• If general endpoints work but specific ones don't, it's a permission issue")
    print("• If all endpoints return 401, check your authentication method")
    print("• For on-prem Confluence, you might need different authentication headers")

if __name__ == "__main__":
    main()
