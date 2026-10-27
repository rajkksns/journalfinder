import json
import re
import time

import pandas as pd
import requests
import streamlit as st

GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PUBLISHERS = [
    "Elsevier (ScienceDirect)", "MDPI", "IEEE", "Wiley", "PeerJ", "Emerald",
    "PLOS", "Taylor & Francis", "ProQuest", "Springer Nature", "Nature Portfolio",
    "Oxford University Press", "Cambridge University Press", "SAGE", "ACS",
    "RSC", "AIP", "APS", "ACM", "Frontiers", "Hindawi", "BMJ", "Cell Press",
    "Other / any publisher",
]


def gemini_request(api_key, prompt, use_search=False, temperature=0.3):
    """Call the Gemini API and return the text response.

    use_search=True turns on Google Search grounding so the model can read
    current journal pages. Grounding does not allow forcing JSON output, so the
    parser handles plain text that contains JSON.
    """
    generation = {"temperature": temperature, "maxOutputTokens": 4096}
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": generation}
    if use_search:
        payload["tools"] = [{"google_search": {}}]
    else:
        generation["responseMimeType"] = "application/json"

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    url = GEMINI_URL.format(model=GEMINI_MODEL)
    response = None
    # Retry on busy server (503), rate limit (429) and slow replies (timeouts).
    for attempt in range(4):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.exceptions.Timeout:
            if attempt == 3:
                raise RuntimeError("Gemini did not respond in time after 4 tries. Please try again in a few minutes.")
            time.sleep(5 * (attempt + 1))
            continue
        if response.status_code not in (429, 503):
            break
        if attempt < 3:
            time.sleep(5 * (attempt + 1))
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:300]}")
    data = response.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts).strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response: {str(data)[:300]}")


def build_prompt(title, abstract, keywords, indexes, publishers, top_n, use_search):
    index_part = (
        f"indexed in {', '.join(indexes)}" if indexes
        else "from any indexing database (Web of Science, Scopus, DOAJ, PubMed, etc.)"
    )
    pub_part = (
        f"published by {', '.join(publishers)}" if publishers
        else "from any publisher, including open-access and society journals"
    )
    keywords_part = f"\nKeywords: {keywords}" if keywords else ""
    web_part = (
        "Use web search to check each journal's official website for current APC (article processing charge), "
        "acceptance rate, and impact factor. Prefer figures from the journal's own site or the publisher. "
        "If a figure cannot be confirmed, write \"Not available\" instead of guessing.\n"
        if use_search else
        "Use your best knowledge. Write \"Not available\" for any figure you are not reasonably sure of.\n"
    )
    return f"""You are an expert academic publishing advisor.
Find the top {top_n} journals best suited for the research paper below. Only suggest journals {index_part} and {pub_part}.

{web_part}
Return ONLY a JSON array (no markdown). Each item must have these keys:
- "journal": journal name
- "publisher": publisher name
- "indexes": list of indexes it is in (e.g. "SCIE", "SSCI", "ESCI", "Scopus", "DOAJ")
- "impact_factor": latest impact factor, or "Not available"
- "acceptance_rate": acceptance rate, or "Not available"
- "apc_cost": article processing charge in INR, written as "₹X,XX,XXX" (e.g. "₹1,65,000"). If the journal publishes its fee in USD or another currency, convert it to INR at the current exchange rate and add the USD amount in brackets, e.g. "₹1,65,000 ($2,000)". Use "No APC (free to publish)" for free journals, or "Not available" if unknown
- "review_speed": typical time to first decision, or "Not available"
- "issues_per_year": number of issues published per year (e.g. "12" or "4 per year"), or "Not available"
- "url": official journal homepage URL
- "why_it_fits": one short sentence

Title: {title}

Abstract: {abstract}{keywords_part}
"""


def parse_journals(text):
    """Parse the model's JSON output (plain or inside text/code fences) into rows."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(items, dict):
        items = items.get("journals", [])
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = item.get("indexes", "")
        if isinstance(idx, list):
            idx = ", ".join(map(str, idx))
        rows.append({
            "Journal": item.get("journal", ""),
            "Publisher": item.get("publisher", ""),
            "Indexed in": idx,
            "Impact Factor": item.get("impact_factor", "Not available"),
            "Acceptance Rate": item.get("acceptance_rate", "Not available"),
            "APC Cost (INR)": item.get("apc_cost", "Not available"),
            "Issues per Year": item.get("issues_per_year", "Not available"),
            "Review Speed": item.get("review_speed", "Not available"),
            "Why it fits": item.get("why_it_fits", ""),
            "URL": item.get("url", ""),
        })
    return rows


st.set_page_config(page_title="Journal Finder", page_icon="📚", layout="wide")
st.title("📚 Journal Finder")
st.caption("Powered by Google Gemini (free tier). Suggestions must be verified on each journal's official site.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Your Gemini API key", type="password",
        help="Get a free key at https://aistudio.google.com/apikey. "
             "It is used only for this session and is not stored.",
    )
    st.subheader("Paper details")
    title = st.text_input("Paper title")
    abstract = st.text_area("Abstract", height=180)
    keywords = st.text_input("Keywords (comma separated)")

    st.subheader("Filters (optional)")
    indexes = st.multiselect(
        "Indexed in", ["SCIE", "SSCI", "ESCI", "Scopus", "DOAJ", "PubMed", "Web of Science (any)"],
        help="Leave empty to include journals from all indexes.",
    )
    publishers = st.multiselect(
        "Publishers", PUBLISHERS,
        help="Leave empty to search all publishers.",
    )
    top_n = st.slider("Number of journals", min_value=5, max_value=25, value=15)
    use_web = st.checkbox(
        "Look up current APC and acceptance rate on the web", value=True,
        help="Uses Google Search, so answers are more current but slower. Turn off if you see errors.",
    )

if st.button("Find Journals", type="primary"):
    if not api_key:
        st.error("Please enter your Gemini API key.")
    elif not (title and abstract):
        st.error("Please enter at least the paper title and abstract.")
    else:
        try:
            with st.spinner("Searching for matching journals..."):
                try:
                    prompt = build_prompt(title, abstract, keywords, indexes, publishers, top_n, use_web)
                    raw = gemini_request(api_key, prompt, use_search=use_web)
                except RuntimeError as err:
                    # Some keys/models reject web search; fall back to model knowledge.
                    if use_web and "error 400" in str(err):
                        st.info("Web lookup is not available for this model. Showing results from model knowledge instead.")
                        prompt = build_prompt(title, abstract, keywords, indexes, publishers, top_n, False)
                        raw = gemini_request(api_key, prompt, use_search=False)
                    else:
                        raise
            rows = parse_journals(raw)
            if rows:
                df = pd.DataFrame(rows)
                st.success(f"Found {len(df)} suggested journals.")
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"URL": st.column_config.LinkColumn("URL")},
                )
                st.download_button(
                    "Download as CSV", df.to_csv(index=False).encode("utf-8"),
                    file_name="journal_suggestions.csv", mime="text/csv",
                )
            else:
                st.warning("The model did not return a usable list. Raw output:")
                st.code(raw)
        except Exception as e:
            st.error(str(e))
