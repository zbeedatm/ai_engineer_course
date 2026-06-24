"""
Test script for Glassdoor API Server
Demonstrates how to interact with the API endpoints
This version tests the real RapidAPI integration
"""

import requests
import json
import sys

# Configuration
BASE_URL = "http://localhost:5000/api/v1"
API_KEY = "your-api-key-here"  # Your Flask server API key

# Headers with API key
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Warning about real API usage
print("=" * 60)
print("⚠️  WARNING: This will make real API calls to RapidAPI")
print("=" * 60)
print("Make sure you have:")
print("1. A valid RapidAPI key set in your .env file")
print("2. An active subscription to Real-Time Glassdoor Data API")
print("3. The Flask server running (python glassdoor_api_server.py)")
print("=" * 60)
print()


def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get("http://localhost:5000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_search_companies(query="Google"):
    """Test company search"""
    print(f"\n=== Searching for: {query} ===")
    response = requests.get(
        f"{BASE_URL}/companies/search",
        headers=headers,
        params={"q": query, "limit": 5}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()


def test_get_company(company_id="1234"):
    """Test getting company details"""
    print(f"\n=== Getting Company Details: {company_id} ===")
    response = requests.get(
        f"{BASE_URL}/companies/{company_id}",
        headers=headers
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_get_reviews(company_id="1234"):
    """Test getting company reviews"""
    print(f"\n=== Getting Reviews for Company: {company_id} ===")
    response = requests.get(
        f"{BASE_URL}/companies/{company_id}/reviews",
        headers=headers,
        params={"limit": 3, "sort": "helpful"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_get_salaries(company_id="1234"):
    """Test getting salary information"""
    print(f"\n=== Getting Salaries for Company: {company_id} ===")
    response = requests.get(
        f"{BASE_URL}/companies/{company_id}/salaries",
        headers=headers,
        params={"job_title": "Software Engineer", "limit": 3}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_get_interviews(company_id="1234"):
    """Test getting interview experiences"""
    print(f"\n=== Getting Interviews for Company: {company_id} ===")
    response = requests.get(
        f"{BASE_URL}/companies/{company_id}/interviews",
        headers=headers,
        params={"limit": 3}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_get_benefits(company_id="1234"):
    """Test getting benefits information"""
    print(f"\n=== Getting Benefits for Company: {company_id} ===")
    response = requests.get(
        f"{BASE_URL}/companies/{company_id}/benefits",
        headers=headers
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_get_ceo(company_id="1234"):
    """Test getting CEO information"""
    print(f"\n=== Getting CEO Info for Company: {company_id} ===")
    response = requests.get(
        f"{BASE_URL}/companies/{company_id}/ceo",
        headers=headers
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_invalid_api_key():
    """Test with invalid API key"""
    print("\n=== Testing Invalid API Key ===")
    invalid_headers = {"X-API-Key": "invalid-key"}
    response = requests.get(
        f"{BASE_URL}/companies/1234",
        headers=invalid_headers
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_missing_api_key():
    """Test with missing API key"""
    print("\n=== Testing Missing API Key ===")
    response = requests.get(f"{BASE_URL}/companies/1234")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("GLASSDOOR API SERVER TESTS")
    print("=" * 60)
    print(f"\nBase URL: {BASE_URL}")
    print(f"API Key: {API_KEY}")
    
    try:
        # Test health check (no auth required)
        test_health_check()
        
        # Test authentication errors
        test_missing_api_key()
        test_invalid_api_key()
        
        # Test GET endpoints - these make REAL API calls to RapidAPI
        test_search_companies("Google")
        test_get_company()
        test_get_reviews()
        test_get_salaries()
        test_get_interviews()
        test_get_benefits()
        test_get_ceo()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        print("\n✅ All endpoints are now using REAL Glassdoor data from RapidAPI!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server.")
        print("Make sure the Flask server is running: python glassdoor_api_server.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    print("Make sure the server is running before executing tests!")
    print("Run: python glassdoor_api_server.py\n")
    
    response = input("Press Enter to run tests (or Ctrl+C to cancel)...")
    run_all_tests()
