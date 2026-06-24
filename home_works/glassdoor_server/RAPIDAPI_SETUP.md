# RapidAPI Configuration Guide

## Getting Your RapidAPI Key

1. **Sign Up for RapidAPI**
   - Visit https://rapidapi.com/
   - Click "Sign Up" and create an account (free)
   - Verify your email address

2. **Subscribe to Real-Time Glassdoor Data API**
   - Go to: https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-glassdoor-data
   - Click "Subscribe to Test" button
   - Choose a pricing plan:
     - **Basic (Free)**: Limited requests per month - good for testing
     - **Pro**: Higher limits for production use
     - **Ultra/Mega**: Enterprise-level access
   - Click "Subscribe"

3. **Get Your API Key**
   - After subscribing, you'll see your API key in the "X-RapidAPI-Key" header
   - Copy this key
   - Add it to your `.env` file as `RAPIDAPI_KEY=your-key-here`

## Available Endpoints on RapidAPI

The Real-Time Glassdoor Data API provides the following endpoints:

### 1. Company Search
```
GET /search
Parameters:
  - query: Company name to search for
  - page: Page number (optional, default: 1)
```

### 2. Company Details
```
GET /company
Parameters:
  - company_id: Glassdoor company ID
```

### 3. Company Reviews
```
GET /reviews
Parameters:
  - company_id: Glassdoor company ID
  - page: Page number (optional)
  - language: Language code (eng, spa, fra, etc.)
  - employment_status: Filter by employment status
```

### 4. Salaries
```
GET /salaries
Parameters:
  - company_id: Glassdoor company ID
  - job_title: Job title filter (optional)
  - location: Location filter (optional)
  - page: Page number (optional)
```

### 5. Interviews
```
GET /interviews
Parameters:
  - company_id: Glassdoor company ID
  - job_title: Job title filter (optional)
  - page: Page number (optional)
```

## Testing Your Setup

1. **Test in RapidAPI Console First**
   - Use the built-in "Test Endpoint" feature on RapidAPI
   - Try searching for a company like "Google" or "Microsoft"
   - Verify you get valid responses

2. **Test Your Flask Server**
   ```bash
   # Make sure your .env file has both keys set
   python glassdoor_api_server.py
   
   # In another terminal, test the search endpoint
   curl -X GET "http://localhost:5000/api/v1/companies/search?q=Google" \
     -H "X-API-Key: your-glassdoor-api-key"
   ```

## Rate Limits

Rate limits depend on your RapidAPI subscription plan:

- **Basic (Free)**: ~500 requests/month
- **Pro**: ~10,000 requests/month
- **Ultra**: ~100,000 requests/month
- **Mega**: ~1,000,000 requests/month

Check your current usage in the RapidAPI dashboard.

## Troubleshooting

### Error: "You are not subscribed to this API"
- Make sure you've subscribed to the API on RapidAPI
- Verify your subscription is active

### Error: "Invalid API Key"
- Check that RAPIDAPI_KEY in .env matches your RapidAPI key
- Make sure there are no extra spaces or quotes

### Error: "Rate limit exceeded"
- You've exceeded your plan's request limit
- Upgrade your plan or wait for the limit to reset

### Error: "Timeout"
- The API request took too long
- Try again or check RapidAPI status page

## Best Practices

1. **Cache Responses**: Store frequently accessed data to reduce API calls
2. **Handle Errors Gracefully**: Always check for error responses
3. **Monitor Usage**: Keep track of your API usage in RapidAPI dashboard
4. **Use Pagination**: For large result sets, use pagination parameters
5. **Implement Rate Limiting**: Add your own rate limiting on the Flask server

## Support

- RapidAPI Support: https://rapidapi.com/support
- API Provider Support: Check the API page on RapidAPI for provider contact info
- This Flask Server: See README.md for documentation
