# Gemini AI Integration for Heart Disease Recommendations

This document explains how to integrate Google's Gemini AI for generating personalized heart disease prevention recommendations.

## Setup Instructions

### 1. Install Dependencies
The required dependencies have been added to `requirements.txt`:
- `google-generativeai==0.3.2`

### 2. Get Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 3. Configure Environment Variables

Set the following environment variables to enable Gemini recommendations:

#### Windows (Command Prompt):
```cmd
set GEMINI_API_KEY=your_api_key_here
set USE_GEMINI_RECOMMENDATIONS=true
```

#### Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
$env:USE_GEMINI_RECOMMENDATIONS="true"
```

#### Linux/macOS:
```bash
export GEMINI_API_KEY=your_api_key_here
export USE_GEMINI_RECOMMENDATIONS=true
```

### 4. Restart the Application
After setting the environment variables, restart both the API server and Streamlit app:

```bash
# Terminal 1 - API Server
uvicorn Heart_Disease.api:app --reload

# Terminal 2 - Streamlit App
streamlit run Heart_Disease/app.py
```

## How It Works

### Fallback System
- If Gemini is not configured or unavailable, the system automatically falls back to static recommendations
- No disruption to existing functionality

### Gemini Integration
- Uses `gemini-1.5-flash` model for fast, cost-effective responses
- Generates 6-8 personalized recommendations based on patient data
- Considers all risk factors: age, blood pressure, cholesterol, heart rate, etc.

### Recommendation Categories
The AI generates recommendations for:
1. **Immediate medical actions** (for high/critical risk)
2. **Lifestyle modifications**
3. **Dietary changes**
4. **Exercise recommendations**
5. **Monitoring and follow-up**
6. **Preventive measures**

## Testing

### Without Gemini (Default)
- Recommendations are generated from predefined rules
- Fast response, no API calls

### With Gemini Enabled
- AI generates personalized recommendations
- May take 1-2 seconds longer due to API call
- More nuanced and specific recommendations

## Troubleshooting

### Common Issues

1. **"Google Generative AI not available"**
   - Install the package: `pip install google-generativeai==0.3.2`

2. **"Failed to configure Gemini AI"**
   - Check your API key is valid
   - Ensure internet connection
   - Verify API key has sufficient quota

3. **Slow responses**
   - Gemini API calls add latency
   - Consider caching frequent requests

### Configuration Options

In `api.py`, you can modify:
- `GEMINI_MODEL`: Change to different Gemini model (e.g., "gemini-1.5-pro")
- `USE_GEMINI_RECOMMENDATIONS`: Toggle AI recommendations on/off

## Cost Considerations

- Gemini 1.5 Flash has very low cost per request
- Typical recommendation generation costs <$0.01 per request
- Monitor usage in Google Cloud Console

## Security Notes

- API key is stored in environment variables (not in code)
- No patient data is sent to external services for recommendations
- Only aggregated risk factors are used for AI prompts