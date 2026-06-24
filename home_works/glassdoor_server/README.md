# Glassdoor Real-Time API Flask Server

A Flask-based REST API server implementing the Glassdoor Real-Time API endpoints for company information, reviews, salaries, interviews, and more. This server integrates with RapidAPI's Real-Time Glassdoor Data API to provide live data from Glassdoor.

## Features

- **Company Search**: Search for companies by name with real-time data
- **Company Details**: Get comprehensive company information including ratings, CEO info, and more
- **Reviews**: Retrieve company reviews with filtering options
- **Salaries**: Access salary information by job title and location
- **Interviews**: View interview experiences and statistics
- **CEO Ratings**: Access CEO approval ratings and information
- **API Key Authentication**: Secure endpoints with API key validation
- **Real-Time Data**: Powered by RapidAPI's Glassdoor Real-Time Data API

**Note**: This API provides **read-only** access to Glassdoor data. The RapidAPI Glassdoor Real-Time Data API does not support write operations (submitting reviews, salaries, or interviews).

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. **Get a RapidAPI Key**:
   - Visit [RapidAPI Glassdoor Real-Time Data](https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-glassdoor-data)
   - Sign up or log in to RapidAPI
   - Subscribe to the Real-Time Glassdoor Data API (free tier available)
   - Copy your RapidAPI key from the API dashboard

2. Clone or download the project files

3. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and set your API keys:
# - GLASSDOOR_API_KEY: Your own API key for authenticating requests to this server
# - RAPIDAPI_KEY: Your RapidAPI key for accessing Glassdoor data
```

6. Run the server:
```bash
python glassdoor_api_server.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Authentication

All endpoints (except `/health` and `/api/v1/docs`) require authentication via API key.

**Methods:**
- Header: `X-API-Key: your-api-key`
- Query parameter: `?api_key=your-api-key`

### Available Endpoints

#### General

- `GET /health` - Health check endpoint
- `GET /api/v1/docs` - API documentation

#### Company Endpoints

- `GET /api/v1/companies/search` - Search companies
  - Query params: `q` (required), `limit`, `offset`
  
- `GET /api/v1/companies/{company_id}` - Get company details
  - Returns: company info, ratings, CEO, headquarters, revenue, etc.

- `GET /api/v1/companies/{company_id}/reviews` - Get company reviews
  - Query params: `limit`, `offset`, `language` (eng, spa, fra, etc.), `employment_status`

- `GET /api/v1/companies/{company_id}/salaries` - Get salary information
  - Query params: `job_title`, `location`, `page`

- `GET /api/v1/companies/{company_id}/interviews` - Get interview experiences
  - Query params: `page`, `job_title`

- `GET /api/v1/companies/{company_id}/benefits` - Get benefits information
  - Note: Available if included in company data

- `GET /api/v1/companies/{company_id}/photos` - Get company photos
  - Returns company logo and any available photos

- `GET /api/v1/companies/{company_id}/ceo` - Get CEO information
  - Returns CEO name and approval rating

## Usage Examples

### Search for Companies

```bash
curl -X GET "http://localhost:5000/api/v1/companies/search?q=Google" \
  -H "X-API-Key: your-api-key"
```

### Get Company Details

```bash
curl -X GET "http://localhost:5000/api/v1/companies/1234" \
  -H "X-API-Key: your-api-key"
```

### Get Company Reviews

```bash
curl -X GET "http://localhost:5000/api/v1/companies/1234/reviews?limit=5&sort=helpful" \
  -H "X-API-Key: your-api-key"
```

### Get Interview Experiences

```bash
curl -X GET "http://localhost:5000/api/v1/companies/1234/interviews?page=1" \
  -H "X-API-Key: your-api-key"
```

### Get CEO Information

```bash
curl -X GET "http://localhost:5000/api/v1/companies/1234/ceo" \
  -H "X-API-Key: your-api-key"
```

## Response Format

All endpoints return JSON responses with appropriate HTTP status codes.

### Success Response (200/201)

```json
{
  "data": { ... },
  "success": true
}
```

### Error Response (4xx/5xx)

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

## Configuration

Environment variables (set in `.env` file):

- `GLASSDOOR_API_KEY`: Your API key for authenticating requests to this Flask server
- `RAPIDAPI_KEY`: Your RapidAPI key for accessing the Glassdoor Real-Time Data API
- `PORT`: Server port (default: 5000)
- `DEBUG`: Debug mode (default: False)

## Data Source

This server uses the [RapidAPI Real-Time Glassdoor Data API](https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-glassdoor-data) to fetch live data from Glassdoor. You'll need a RapidAPI account and subscription to use this service.

### RapidAPI Pricing
- **Free Tier**: Available with limited requests
- **Pro Tiers**: Higher rate limits and volume options

Visit the API page on RapidAPI for current pricing and plans.

## Development

### Running in Debug Mode

```bash
export DEBUG=True
python glassdoor_api_server.py
```

### Testing

You can test endpoints using curl, Postman, or any HTTP client.

## Notes

- This implementation makes real API calls to RapidAPI's Glassdoor Real-Time Data API
- All GET endpoints return live data from Glassdoor
- POST endpoints (submit review, salary, interview) are included but require additional integration with a data submission service
- Rate limiting is handled by RapidAPI based on your subscription plan
- Response times vary between 0.5s - 4s depending on the endpoint
- Some endpoints may have limited data availability depending on the company

## API Response Examples

The RapidAPI returns data in the following formats:

### Company Search
```json
{
  "data": [
    {
      "company_id": 9079,
      "name": "Google",
      "website": "https://www.google.com",
      "rating": 4.5
    }
  ]
}
```

### Company Reviews
```json
{
  "data": {
    "reviews": [...],
    "review_count": 50000,
    "rating": 4.4
  }
}
```

## Security Considerations

- Store API keys securely (never commit `.env` file)
- Use HTTPS in production
- Implement rate limiting
- Add request validation and sanitization
- Consider adding CORS headers for web client access

## License

This is a template implementation. Adjust according to your needs and Glassdoor's API terms of service.
