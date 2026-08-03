"""Download scheme records from the myScheme API into the local JSON dataset."""

import json
import os
from pathlib import Path
import time

import requests
from dotenv import load_dotenv

load_dotenv()

SEARCH_URL = "https://api.myscheme.gov.in/search/v6/schemes"
DETAIL_URL = "https://api.myscheme.gov.in/schemes/v6/public/schemes"
OUTPUT_FILE = Path(__file__).resolve().parent / "Education_scheme_details.json"


def main():
    api_key = os.getenv("MY_SCHEME_API_KEY")
    if not api_key:
        raise RuntimeError("MY_SCHEME_API_KEY is required to download scheme data.")
    session = requests.Session()
    session.headers.update({
        "X-Api-Key": api_key,
        "Origin": "https://www.myscheme.gov.in",
        "Referer": "https://www.myscheme.gov.in/",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Indian-Education-Scheme-Assistant/1.0",
    })
    summaries = []
    for offset in range(0, 1500, 100):
        params = {"lang": "en", "q": '[{"identifier":"beneficiaryState","value":"All"}]', "keyword": "", "sort": "multiple_sort", "from": offset, "size": 100}
        try:
            response = session.get(SEARCH_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as error:
            raise SystemExit(f"Unable to download scheme summaries: {error}") from error
        items = response.json().get("data", {}).get("hits", {}).get("items", [])
        if not items:
            break
        summaries.extend(items)
        print(f"Downloaded {len(summaries)} summaries")

    details = []
    for index, scheme in enumerate(summaries, start=1):
        slug = (scheme.get("fields") or {}).get("slug")
        if not slug:
            continue
        try:
            response = session.get(DETAIL_URL, params={"slug": slug, "lang": "en"}, timeout=30)
            response.raise_for_status()
            details.append(response.json())
            print(f"[{index}/{len(summaries)}] Downloaded: {slug}")
        except requests.RequestException as error:
            print(f"[{index}/{len(summaries)}] Failed: {slug} ({error})")
        time.sleep(0.2)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(details, file, ensure_ascii=False, indent=2)
    print(f"Saved {len(details)} scheme details to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
