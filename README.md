# Journal Finder

A Streamlit web app that suggests scientific journals for a research paper using the
**Google Gemini API** (free tier, model `gemini-3.8-flash`).

## Getting Started

1. Get a free Gemini API key from https://aistudio.google.com/apikey
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Enter your API key, paper title, abstract and (optionally) keywords
5. Choose optional filters for indexes and publishers, then click **Find Journals**
6. Download the results as CSV if needed

## How It Works

The app builds a prompt from your title, abstract, keywords and filters, sends it to
Gemini, and asks for the answer as JSON. The results are shown in a table with
publisher, indexes, impact factor, acceptance rate, review speed and journal URL.

## Limitations

- The app does **not** have a live database of all journals. Results come from the
  model's knowledge, so impact factors, acceptance rates and index status can be
  outdated or wrong. Always verify on the journal's official site or in Web of Science / Scopus.
- Free-tier Gemini has rate limits, so very frequent searches may fail for a while.
- Leaving the index and publisher filters empty searches all journals the model knows about,
  but it cannot list every journal in existence.

## Changes from the original

- Replaced the OpenAI GPT-3 calls (`text-davinci-002`, `gpt-3.5-turbo`) with the Gemini API.
- Results are requested as structured JSON and shown as a sortable table.
- Added optional filters for any index and any publisher, a result count slider, and CSV export.

## Hosting notes

This copy does not save API keys. Each visitor enters their own free Gemini key, used only for that session.
