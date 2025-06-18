# 🌟 Vibe Mapping Shopping Assistant

A conversational AI shopping assistant that recommends fashion products based on customer's style "vibes" and preferences.

## ✨ Features

- **Natural Language Processing**: Understands style "vibes" like "casual", "elevated", "flowy" and converts them to concrete product attributes
- **Smart Recommendations**: Provides tailored product recommendations based on user preferences
- **Conversational UI**: Chat-based interface that feels natural and engaging
- **Personalized Justifications**: Explains why products were recommended in a human-like way using Gemini API
- **Smart Follow-up Questions**: Dynamically generates contextual questions to gather missing information

## 🛠️ Tech Stack

- **Streamlit**: Powers the interactive web interface
- **Pandas**: Handles product catalog data processing
- **Google Gemini AI**: Generates personalized responses and follow-up questions
- **Python-dotenv**: Manages API keys and environment variables

## 💻 Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure your environment variables in `.env` file:
   ```
   GEMINI_API_KEY="your_gemini_api_key"
   GOOGLE_API_KEY="your_google_api_key"
   ```

3. Run the application:
   ```
   streamlit run pp.py
   ```

## 👩‍💻 Usage

The shopping assistant will:
1. Ask about your style preferences in a conversational manner
2. Request any missing critical information about size, budget, etc.
3. Display product recommendations that match your style "vibes"
4. Explain why each product was recommended

## 📊 Project Structure

- `pp.py`: Main application with UI components and business logic
  - `VibeMapper`: Maps style vibes to product attributes
  - `FollowUpEngine`: Generates contextual follow-up questions
  - `RecommendationEngine`: Handles product filtering and ranking
  - `GeminiAgent`: Integrates with Google Gemini API
  - `ShoppingAgent`: Main agent coordinating the conversation flow
- `requirements.txt`: Dependencies list
- `.env`: Configuration file for API keys