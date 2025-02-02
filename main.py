#!/usr/bin/env python3

import os
import sys
import requests
import pdfplumber
import openai
from bs4 import BeautifulSoup

openai.api_key = ""

# A helper prompt template asking for specific financial metrics in JSON
PROMPT_TEMPLATE = """
You are a financial data extraction assistant. 
Given the following text from a company's earnings report, please extract any mention of:
- Revenue
- Earnings
- Operating margin
- Revenue growth rates (Year-over-Year, Quarter-over-Quarter, etc.)
- Guidance or outlook (if present)

Return the extracted data in valid JSON format with the following structure (even if some fields are missing). Return ONLY valid JSON. DO NOT include any explanation or extra text. If you cannot find data, leave the fields blank, but maintain valid JSON format:
{{
  "Revenue": "...",
  "Earnings": "...",
  "OperatingMargin": "...",
  "RevenueGrowthRates": "...",
  "Guidance": "..."
}}

Text to analyze:
{chunk_text}
"""

def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text

def extract_text_from_html(url_or_path: str) -> str:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        response = requests.get(url_or_path)
        response.raise_for_status()
        html_content = response.text
    else:
        # Assume it's a local .html file
        with open(url_or_path, "r", encoding="utf-8") as f:
            html_content = f.read()

    # Parse HTML and extract visible text
    soup = BeautifulSoup(html_content, "html.parser")
    # A simple approach: get all text from <body> or the entire doc
    text = soup.get_text(separator="\n")
    return text

def chunk_text(text: str, max_chars: int = 3000):

    chunks = []
    current_chunk = []
    current_length = 0

    lines = text.split("\n")
    for line in lines:
        line_length = len(line)
        if current_length + line_length > max_chars:
            # Create a new chunk
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_length = line_length
        else:
            current_chunk.append(line)
            current_length += line_length
    
    # Add the last chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def extract_financial_data(text: str):
    text_chunks = chunk_text(text)

    # We'll store partial results from each chunk
    aggregated_data = {
        "Revenue": [],
        "Earnings": [],
        "OperatingMargin": [],
        "RevenueGrowthRates": [],
        "Guidance": []
    }

    for chunk_index, chunk_content in enumerate(text_chunks):
        # Prepare the prompt
        prompt = PROMPT_TEMPLATE.format(chunk_text=chunk_content)

        # Call ChatGPT
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0  # lower temperature => more deterministic response
        )

        # Extract the response content
        chatgpt_reply = response["choices"][0]["message"]["content"].strip()

        print(f"--- ChatGPT raw response for chunk {chunk_index} ---")
        print(chatgpt_reply)
        print("-----------------------------------------------")

        # Try to parse JSON. If that fails, just store raw text.
        import json
        try:
            data = json.loads(chatgpt_reply)
            # Append chunk results to aggregated_data
            for key in aggregated_data.keys():
                if key in data and data[key]:
                    value = data[key]
                    if isinstance(value, dict):
                        value = json.dumps(value)
                    elif not isinstance(value, str):
                        value = str(value)
                    aggregated_data[key].append(value)
                    # aggregated_data[key].append(data[key])
        except json.JSONDecodeError:
            # If ChatGPT didn't return valid JSON, store as raw text
            aggregated_data["Revenue"].append("Failed to parse JSON in chunk {}".format(chunk_index))
            continue

    # Consolidate final results (join multiple chunk outputs, if any)
    final_result = {
        key: "; ".join(val_list) if val_list else ""
        for key, val_list in aggregated_data.items()
    }

    return final_result

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_or_html_path_or_url>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Detect PDF or HTML by extension or by URL
    if input_path.lower().endswith(".pdf"):
        print(f"[*] Detected PDF input: {input_path}")
        text = extract_text_from_pdf(input_path)
    elif input_path.lower().endswith(".html") or input_path.startswith("http"):
        print(f"[*] Detected HTML input: {input_path}")
        text = extract_text_from_html(input_path)
    else:
        print("[!] Unknown file format. Please provide a .pdf file or an HTML (.html) file or a URL.")
        sys.exit(1)

    # Raw text from the doc
    print("[*] Extracted text length:", len(text))

    # Use ChatGPT to extract financial metrics
    print("[*] Extracting financial metrics with ChatGPT...")
    results = extract_financial_data(text)

    # Print final result as JSON
    import json
    print("[*] Final extracted metrics (JSON):")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
