"""
Flask Server for Glassdoor Real-Time API
Implements endpoints for company reviews, salaries, interviews, and more
Uses RapidAPI's Real-Time Glassdoor Data API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import os
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # allow WebUI (in the browser) to call us

# Configuration
RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY', '')
RAPIDAPI_HOST = "glassdoor-real-time.p.rapidapi.com"


# Middleware for API key validation
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')

        if not api_key:
            return jsonify({
                'error': 'No API key provided',
                'message': 'Please provide an API key via X-API-Key header or api_key query parameter'
            }), 401

        if api_key != RAPIDAPI_KEY:
            return jsonify({
                'error': 'Invalid API key',
                'message': 'The provided API key is not valid'
            }), 403

        return f(*args, **kwargs)

    return decorated_function


# Helper function to make RapidAPI requests
def make_rapidapi_request(endpoint, params=None):
    """
    Make a request to the RapidAPI Glassdoor Real-Time Data API
    """
    url = f"https://{RAPIDAPI_HOST}/companies/{endpoint}"
    # example: https://glassdoor-real-time.p.rapidapi.com/companies/search?query=Meta

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        # response.raise_for_status()
        return response.json(), None
    except requests.exceptions.Timeout:
        return None, {'error': 'Request timeout', 'message': 'The API request timed out'}
    except requests.exceptions.HTTPError as e:
        return None, {'error': 'API error', 'message': f'API returned status code {e.response.status_code}'}
    except requests.exceptions.RequestException as e:
        return None, {'error': 'Connection error', 'message': str(e)}
    except ValueError:
        return None, {'error': 'Invalid response', 'message': 'Failed to parse API response'}


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200


# Company Search
@app.route('/api/v1/companies/search', methods=['GET'])
# @require_api_key
def search_companies():
    """
    Search for companies by name
    Query params: q (search query), limit, offset
    """
    query = request.args.get('q', '')
    limit = request.args.get('limit', 10, type=int)
    offset = request.args.get('offset', 0, type=int)

    if not query:
        return jsonify({
            'error': 'Missing query parameter',
            'message': 'Please provide a search query using the "q" parameter'
        }), 400

    # Make request to RapidAPI
    params = {
        'query': query,
        'limit': limit,
        'page': offset // limit + 1  # Convert offset to page number
    }

    data, error = make_rapidapi_request('search', params)

    if error:
        return jsonify(error), 500

    # Transform response to match our API format
    results = data.get('data', []) if data else []

    # return data
    return jsonify({
        'query': query,
        'results': results,
        'company_id': results["employerResults"][0]["employer"]["id"],
        'total': len(results),
        'limit': limit,
        'offset': offset
    }), 200


# Company Details
@app.route('/api/v1/companies/base-info', methods=['GET'])
# @require_api_key
def get_company():
    """
    Get detailed information about a specific company
    """
    company_id = request.args.get('companyId', '')

    # Make request to RapidAPI
    params = {'companyId': company_id}

    data, error = make_rapidapi_request('base-info', params)

    if error:
        return jsonify(error), 500

    # if data["errors"]:
    #     return jsonify(data["errors"]), 500

    return jsonify(data.get('data', {}) if data else {}), 200


# Company Reviews
@app.route('/api/v1/companies/reviews', methods=['GET'])
# @require_api_key
def get_company_reviews():
    """
    Get reviews for a specific company
    Query params: limit, offset, sort (helpful, recent, rating), language, employment_status
    """
    company_id = request.args.get('companyId', '')
    limit = request.args.get('limit', 5, type=int)
    offset = request.args.get('page', 1, type=int)
    sort = request.args.get('sort', 'RELEVANCE')
    # offset = request.args.get('offset', 0, type=int)
    # language = request.args.get('language', 'eng')
    # employment_status = request.args.get('employment_status', '')

    # Calculate page from offset
    page = offset // limit + 1

    # Make request to RapidAPI
    params = {
        'companyId': company_id,
        'page': page,
        'limit': limit,
        'sort': sort
    }

    # if employment_status:
    #     params['employment_status'] = employment_status

    data, error = make_rapidapi_request('reviews', params)

    if error:
        return jsonify(error), 500

    # Extract reviews and metadata
    demographic_reviews = data.get('data', {}).get('demographicRatingsRG', []) if data else []
    employer_reviews = data.get('data', {}).get('employerReviewsRG', []) if data else []
    # total_count = data.get('data', {}).get('review_count', 0) if data else 0

    return jsonify({
        'company_id': company_id,
        'demographic_reviews': demographic_reviews,
        'employer_reviews': employer_reviews,
        # 'total': total_count,
        'limit': limit,
        'offset': offset,
        'sort': sort,
        # 'language': language
    }), 200


# Company Interviews
@app.route('/api/v1/companies/interviews', methods=['GET'])
# @require_api_key
def get_company_interviews(company_id):
    """
    Get interview experiences for a specific company
    Query params: page, job_title
    """
    company_id = request.args.get('companyId', '')
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 5, type=int)
    sort = request.args.get('sort', 'RELEVANCE')
    # job_title = request.args.get('job_title', '')

    # Make request to RapidAPI
    params = {
        'companyId': company_id,
        'page': page,
        'limit': limit,
        'sort': sort
    }

    # if job_title:
    #     params['job_title'] = job_title

    data, error = make_rapidapi_request('interviews', params)

    if error:
        return jsonify(error), 500

    # Extract interviews data
    interviews_data = data.get('data', {}) if data else {}
    interviews = interviews_data.get('employerInterviews', [])

    return jsonify({
        'company_id': company_id,
        'interviews': interviews,
        'total': len(interviews),
        'page': page
    }), 200


#############################   Not in use  ##############################
# # Company Benefits
# @app.route('/api/v1/companies/benefits', methods=['GET'])
# # @require_api_key
# def get_company_benefits(company_id):
#     """
#     Get benefits information for a specific company
#     Note: This endpoint may not be directly supported by the upstream API
#     """
#     # Try to get company details which may include benefits info
#     params = {'company_id': company_id}
#
#     data, error = make_rapidapi_request('company', params)
#
#     if error:
#         return jsonify(error), 500
#
#     # Extract benefits if available in company data
#     company_data = data.get('data', {}) if data else {}
#     benefits = company_data.get('benefits', {})
#
#     # If no benefits data, return a message
#     if not benefits:
#         return jsonify({
#             'company_id': company_id,
#             'message': 'Benefits information not available for this company',
#             'benefits': {}
#         }), 200
#
#     return jsonify({
#         'company_id': company_id,
#         'benefits': benefits
#     }), 200
#
#
# # Company Photos
# @app.route('/api/v1/companies/photos', methods=['GET'])
# # @require_api_key
# def get_company_photos(company_id):
#     """
#     Get photos for a specific company
#     Note: This endpoint may not be directly supported by the upstream API
#     """
#     page = request.args.get('page', 1, type=int)
#
#     # Try to get company details which may include photo URLs
#     params = {'company_id': company_id}
#
#     data, error = make_rapidapi_request('company', params)
#
#     if error:
#         return jsonify(error), 500
#
#     # Extract photos if available
#     company_data = data.get('data', {}) if data else {}
#     photos = company_data.get('photos', [])
#     company_logo = company_data.get('company_logo', '')
#
#     # If we have a company logo, include it
#     result_photos = []
#     if company_logo:
#         result_photos.append({
#             'id': 'logo',
#             'url': company_logo,
#             'caption': 'Company Logo',
#             'type': 'logo'
#         })
#
#     # Add any additional photos
#     result_photos.extend(photos)
#
#     return jsonify({
#         'company_id': company_id,
#         'photos': result_photos,
#         'total': len(result_photos),
#         'page': page
#     }), 200
#
#
# # CEO Information
# @app.route('/api/v1/companies/ceo', methods=['GET'])
# # @require_api_key
# def get_company_ceo(company_id):
#     """
#     Get CEO information and ratings for a specific company
#     """
#     # Make request to RapidAPI for company details
#     params = {'company_id': company_id}
#
#     data, error = make_rapidapi_request('company', params)
#
#     if error:
#         return jsonify(error), 500
#
#     # Extract CEO information from company data
#     company_data = data.get('data', {}) if data else {}
#
#     ceo_info = {
#         'company_id': company_id,
#         'ceo': {
#             'name': company_data.get('ceo', 'Unknown'),
#             'approval_rating': company_data.get('ceo_rating', 0),
#             'title': 'CEO'
#         }
#     }
#
#     return jsonify(ceo_info), 200
#
#
# # Company Salaries
# @app.route('/api/v1/companies/salaries/search', methods=['GET'])
# # @require_api_key
# def get_company_salaries(company_id):
#     """
#     Get salary information for a specific company
#     Query params: job_title, location, page
#     """
#     job_title = request.args.get('job_title', '')
#     location = request.args.get('location', '')
#     page = request.args.get('page', 1, type=int)
#
#     # Make request to RapidAPI
#     params = {
#         'company_id': company_id,
#         'page': page
#     }
#
#     if job_title:
#         params['job_title'] = job_title
#     if location:
#         params['location'] = location
#
#     data, error = make_rapidapi_request('salaries', params)
#
#     if error:
#         return jsonify(error), 500
#
#     # Extract salaries data
#     salaries = data.get('data', []) if data else []
#
#     return jsonify({
#         'company_id': company_id,
#         'salaries': salaries,
#         'total': len(salaries),
#         'page': page
#     }), 200


# API Documentation endpoint
@app.route('/api/v1/docs', methods=['GET'])
def api_docs():
    """
    Return API documentation
    """
    return jsonify({
        'version': '1.0.0',
        'base_url': request.host_url + 'api/v1',
        'authentication': 'API Key via X-API-Key header or api_key query parameter',
        'data_source': 'RapidAPI Real-Time Glassdoor Data API (Read-Only)',
        'endpoints': {
            'GET /companies/search': 'Search for companies by name',
            'GET /companies/{id}': 'Get company details',
            'GET /companies/{id}/reviews': 'Get company reviews',
            'GET /companies/{id}/salaries': 'Get salary information',
            'GET /companies/{id}/interviews': 'Get interview experiences',
            'GET /companies/{id}/benefits': 'Get benefits information',
            'GET /companies/{id}/photos': 'Get company photos',
            'GET /companies/{id}/ceo': 'Get CEO information'
        },
        'note': 'This API provides read-only access to Glassdoor data. Write operations (POST) are not supported.'
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    # IMPORTANT: 0.0.0.0 so Docker can reach it
    app.run(host='0.0.0.0', port=port, debug=debug)
    # Local testing
    # app.run(host='127.0.0.1', port=port, debug=True)