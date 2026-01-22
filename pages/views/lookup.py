import os
import re
import json
from difflib import SequenceMatcher
# from urllib.parse import urljoin, urlparse

import requests
# from bs4 import BeautifulSoup

from django.db.models import Q
from django.views.decorators.csrf import csrf_exempt

from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status, permissions

from ..models import Charity
from ..serializers import CharitySerializer

from openai import OpenAI

client = OpenAI()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")
SERP_API_KEY = os.getenv("SERP_API_KEY")
APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")


def _get_context_from_session(request):
    """Retrieve previous charity chat context from session."""
    history = request.session.get("ai_charity_context", [])
    return "\n".join(history[-3:])  # last few turns


def _update_context_session(request, user_input, ai_output):
    """Store last few user ↔ AI messages for conversational continuity."""
    history = request.session.get("ai_charity_context", [])
    history.append(f"User: {user_input}\nAI: {ai_output}")
    request.session["ai_charity_context"] = history[-5:]  # keep last 5


def _store_last_matches(request, matches):
    """Store last matches for later filtering."""
    if not matches:
        return
    request.session["ai_last_matches"] = matches
    request.session.modified = True


def _get_last_matches(request):
    return request.session.get("ai_last_matches", [])

def enrich_charity_contact_info_after_selection(charity_name, address):
    """
    Contact enrichment is disabled during search.
    This function is only called after explicit user selection.
    """
    return {"website": None, "emails": [], "phones": []}

# def get_charity_contact_info(charity_name, address):
#     """
#     Try to find website / email / phone for a charity using SERPAPI + scraping.
#     Only used when DB record is missing contact info.
#     """
#     if not SERP_API_KEY:
#         return {"website": None, "emails": [], "phones": []}

#     serp_url = "https://serpapi.com/search.json"
#     params = {
#         "engine": "google",
#         "q": f"{charity_name} official site in {address}",
#         "api_key": SERP_API_KEY,
#     }

#     try:
#         res = requests.get(serp_url, params=params, timeout=10)
#         data = res.json()
#         website = None
#         if "organic_results" in data and len(data["organic_results"]) > 0:
#             website = data["organic_results"][0].get("link", None)
#     except Exception as e:
#         print(f"[SERP ERROR] {charity_name}: {e}")
#         return {"website": None, "emails": [], "phones": []}

#     if not website:
#         print(f"[SERP] No website found for {charity_name}")
#         return {"website": None, "emails": [], "phones": []}

#     # def scrape_page(url):
#     #     try:
#     #         html = requests.get(url, timeout=10).text
#     #         emails = re.findall(
#     #             r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}",
#     #             html,
#     #         )
#     #         phones = re.findall(
#     #             r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
#     #             html,
#     #         )
#     #         emails = list(
#     #             set(
#     #                 [
#     #                     e
#     #                     for e in emails
#     #                     if not e.lower().endswith(
#     #                         (
#     #                             ".png",
#     #                             ".jpg",
#     #                             ".jpeg",
#     #                             "1",
#     #                             "2",
#     #                             "3",
#     #                             "4",
#     #                             "5",
#     #                             "6",
#     #                             "7",
#     #                             "8",
#     #                             "9",
#     #                             "0",
#     #                         )
#     #                     )
#     #                 ]
#     #             )
#     #         )
#     #         phones = list(set(phones))
#     #         return emails, phones, html
#     #     except Exception as e:
#     #         print(f"[SCRAPE ERROR] {url}: {e}")
#     #         return [], [], ""

#     def scrape_page_with_apify(url: str):
#         """
#         Uses Apify actor: vdrmota/contact-info-scraper
#         Returns: (emails, phones)
#         """
#         if not APIFY_TOKEN:
#             print("[APIFY] Missing APIFY_API_TOKEN")
#             return [], []

#         run_url = (
#             "https://api.apify.com/v2/acts/"
#             "vdrmota~contact-info-scraper/run-sync-get-dataset-items"
#         )

#         payload = {
#             "startUrls": [{"url": url}],
#             "maxDepth": 1,
#             "maxRequests": 5,
#             "proxyConfiguration": {"useApifyProxy": True},
#         }

#         try:
#             res = requests.post(
#                 run_url,
#                 params={
#                     "token": APIFY_TOKEN,
#                     "clean": "true",
#                 },
#                 json=payload,
#                 timeout=90,
#             )
#             res.raise_for_status()
#             items = res.json() or []

#             emails = set()
#             phones = set()

#             for item in items:
#                 for e in item.get("emails", []):
#                     emails.add(e.lower())
#                 for p in item.get("phones", []):
#                     phones.add(p)
#                 for p in item.get("phonesUncertain", []):
#                     phones.add(p)

#             return list(emails), list(phones)

#         except Exception as e:
#             print(f"[APIFY SCRAPE ERROR] {url}: {e}")
#             return [], []

#     # all_emails, all_phones, html = scrape_page(website)
#     # soup = BeautifulSoup(html, "html.parser")
    
#     # all_emails, all_phones = scrape_page_with_apify(website)
#     all_emails, all_phones = [], []



#     # contact_links = []
#     # for link in soup.find_all("a", href=True):
#     #     href = link["href"].lower()
#     #     if any(k in href for k in ["contact", "about", "team", "staff"]):
#     #         full_url = urljoin(website, href)
#     #         domain = urlparse(full_url).netloc
#     #         if domain == urlparse(website).netloc:
#     #             contact_links.append(full_url)
#     # contact_links = list(set(contact_links))

#     # for link in contact_links:
#     #     sub_emails, sub_phones, _ = scrape_page(link)
#     #     all_emails.extend(sub_emails)
#     #     all_phones.extend(sub_phones)


#     all_emails = list(set(all_emails))
#     all_phones = list(set(all_phones))

#     return {"website": website, "emails": all_emails, "phones": all_phones}

def _search_with_openai(search_descriptor: str, previous_context=None):
    """
    Uses GPT-4o-mini to identify US charities by description.
    No web_search tool; we just strongly instruct it not to hallucinate EINs.
    Returns: { "via": "openai", "results": { matches: [...], ... } }
    """

    context_text = f"\nPrevious context:\n{previous_context}\n" if previous_context else ""

    prompt = f"""
You are a US charity verification assistant.

The user is searching for: {search_descriptor}

Rules:
- Prefer real US-based organizations (nonprofits, churches, schools, NGOs).
- EIN (TIN) must be an actual known EIN. If you are not sure, you MAY omit EIN or leave it empty.
- NEVER invent completely fake organizations. Use your best knowledge but be cautious.
- If you are really uncertain, return an empty matches list and set needs_clarification=true.

Return ONLY JSON:
{{
  "matches": [
    {{
      "name": "",
      "location": "",
      "type": "",
      "website": "",
      "address": "",
      "tin": "",
      "confidence": 0.0
    }}
  ],
  "needs_clarification": true|false,
  "explanation": "brief explanation of what was found or why not"
}}
{context_text}
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.25,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious assistant that suggests real charities. "
                        "NEVER fabricate obviously fake data. Be explicit when unsure."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=20,
        )
        raw = completion.choices[0].message.content
        result_json = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        print(f"[OpenAI Error] {e}")
        return {
            "via": "openai",
            "results": {
                "matches": [],
                "needs_clarification": True,
                "explanation": "Error contacting OpenAI.",
            },
            "error": str(e),
        }

    matches = result_json.get("matches", []) or []

    # Add verified flag for LLM-provided data (not DB-verified)
    cleaned_matches = []
    for m in matches:
        m = m or {}
        m["verified"] = False
        cleaned_matches.append(m)

    # Sort using location + confidence if present
    cleaned_matches.sort(
        key=lambda m: (m.get("location", "").lower(), -float(m.get("confidence", 0.0)))
    )

    return {
        "via": "openai",
        "results": {
            "matches": cleaned_matches,
            "needs_clarification": result_json.get("needs_clarification", False),
            "explanation": result_json.get(
                "explanation",
                "Here are the closest matching charities I could find.",
            ),
        },
    }


# def _classify_intent_with_openai(message: str, has_previous_matches: bool):
#     """
#     Classify user message into: search | filter | chat
#     We allow the model to consider whether there are previous results.
#     """
#     try:
#         prompt = f"""
# Classify the user's intent for a charity assistant.

# User message: "{message}"
# Previous_results_exist: {str(bool(has_previous_matches)).lower()}

# Possible intents:
# - "search": looking for a charity, EIN, or new organization matches.
# - "filter": narrowing or refining existing results (e.g. by city, state, type, branch).
# - "chat": asking general questions, explanations, or small talk.

# Return ONLY JSON like:
# {{ "intent": "search" }}
# """

#         completion = client.chat.completions.create(
#             model="gpt-4o-mini",
#             temperature=0.0,
#             response_format={"type": "json_object"},
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You classify the intent of queries for a charity lookup assistant.",
#                 },
#                 {"role": "user", "content": prompt},
#             ],
#             timeout=10,
#         )
#         raw = completion.choices[0].message.content
#         data = json.loads(raw) if isinstance(raw, str) else raw
#         intent = data.get("intent", "search")
#         if intent not in {"search", "filter", "chat"}:
#             intent = "search"
#         return intent
#     except Exception as e:
#         print(f"[OpenAI Intent Error] {e}")
#         return "search"


# def _llm_filter_charities(filter_text: str, charities: list):
#     """
#     Use LLM to filter a list of charity dicts.
#     Shared by /ai-filter/ endpoint and inline filter inside ai_router.
#     """
#     # Limit to 100 max to save tokens
#     charities = (charities or [])[:100]

#     prompt = f"""
# You are an intelligent charity data filter.
# The user wants to filter the charity list below based on this instruction:
# "{filter_text}"

# You will read the JSON list and return only those entries that match the intent.

# RULES:
# - Never invent new charities or change data.
# - Only return entries that come from the input list.
# - If uncertain, include slightly broader results rather than excluding valid ones.

# Return only JSON in the format:
# {{"filtered": [ ...subset of original charities... ], "reason": "explain briefly"}}

# Here is the data (JSON list):
# {json.dumps(charities, indent=2)}
# """

#     try:
#         completion = client.chat.completions.create(
#             model="gpt-4o-mini",
#             temperature=0.3,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a precise assistant that filters JSON lists.",
#                 },
#                 {"role": "user", "content": prompt},
#             ],
#             response_format={"type": "json_object"},
#             timeout=25,
#         )

#         result = completion.choices[0].message.content
#         parsed = json.loads(result)
#         filtered = parsed.get("filtered", []) or []
#         reason = parsed.get("reason", "Filtered based on given instruction.")
#         return filtered, reason, None
#     except Exception as e:
#         print(f"[AI FILTER ERROR] {e}")
#         return [], "AI filter failed.", str(e)


def _chat_with_openai(message: str, request):
    """General conversational reply when intent == 'chat'."""
    previous_context = _get_context_from_session(request)
    context_text = f"\nPrevious context:\n{previous_context}\n" if previous_context else ""

    prompt = f"""
You are a helpful assistant that explains US charities, EINs, and related concepts.

User message: "{message}"
{context_text}

Provide a concise, helpful answer. Do NOT invent specific charities or EINs.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful charity knowledge assistant. Don't fabricate specific charities.",
                },
                {"role": "user", "content": prompt},
            ],
            timeout=20,
        )
        reply = completion.choices[0].message.content
        _update_context_session(request, message, reply)
        return reply
    except Exception as e:
        print(f"[OpenAI Chat Error] {e}")
        return "I had trouble answering that. Please try again."
# def _pick_best_email(emails):
#     if not emails:
#         return None
#     for e in emails:
#         if not any(k in e for k in ["media", "press", "career", "jobs", "privacy"]):
#             return e
#     return emails[0]

# def _pick_best_phone(phones):
#     if not phones:
#         return None
#     for p in phones:
#         if len(re.sub(r"\D", "", p)) >= 10:
#             return p
#     return phones[0]

def _perform_database_search(name: str, tin: str):
    """
    Search local Charity table by name and/or TIN.
    Returns (matches_serialized, any_enriched, needs_clarification).
    All DB-backed matches will include verified=True.
    """
    filters = Q()
    if name:
        filters &= Q(name__icontains=name)
    if tin:
        filters &= Q(tin__iexact=tin)

    matches = Charity.objects.filter(filters).distinct()

    if not matches.exists():
        return [], False, False

    def name_similarity(a, b):
        try:
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()
        except Exception:
            return 0.0

    if name:
        matches = sorted(
            matches,
            key=lambda c: name_similarity(name, c.name),
            reverse=True,
        )

    # MAX_ENRICH = 5
    enriched = []
    # any_updated = False

    # IMPORTANT:
    # Do NOT enrich charities during search.
    # SERP / Apify enrichment must only occur
    # after explicit user selection from frontend.

    for charity in matches:
        # contact_missing = not (
        #     charity.website or charity.contact_email or charity.contact_telephone
        # )
        # contact_missing = not (
        #     charity.contact_email and charity.contact_telephone
        # )
        # contact_missing = False

        # if contact_missing and idx < MAX_ENRICH:
        #     info = get_charity_contact_info(charity.name, charity.address or "")
        #     updated = False
        #     # if info.get("website"):
        #     #     charity.website = info["website"]
        #     #     updated = True
        #     # if info.get("emails"):
        #     #     charity.contact_email = info["emails"][0]
        #     #     updated = True
        #     # if info.get("phones"):
        #     #     charity.contact_telephone = info["phones"][0]
        #     #     updated = True
        #     # if info.get("emails"):
        #     #     charity.contact_email = info["emails"][0]

        #     # if info.get("phones"):
        #     #     charity.contact_telephone = info["phones"][0]

        #     # if info.get("emails") or info.get("phones"):
        #     #     updated = True

        #     email = _pick_best_email(info.get("emails"))
        #     phone = _pick_best_phone(info.get("phones"))

        #     if email:
        #         charity.contact_email = email
        #         updated = True

        #     if phone:
        #         charity.contact_telephone = phone
        #         updated = True

        #     if info.get("website") and not charity.website:
        #         charity.website = info["website"]
        #         updated = True

        #     if updated:
        #         charity.save()
        #         any_updated = True
        #         print(f"[ENRICHED] {charity.name}")

        data = CharitySerializer(charity).data
        # All DB results are verified
        data["verified"] = True
        enriched.append(data)

    needs_clarification = len(enriched) > 1
    # return enriched, any_updated, needs_clarification
    return enriched, False, needs_clarification



# def _looks_like_tin(text: str):
#     digits = re.sub(r"\D", "", text or "")
#     return digits.isdigit() and 7 <= len(digits) <= 9


# -----------------------------------------------------------------------------
# Legacy OpenAI search flow (kept for /ai-search/ compatibility)
# -----------------------------------------------------------------------------

def _perform_search(name: str, tin: str, request):
    """
    High-level search (legacy used by /ai-search/):
    1) Fast EIN lookup
    2) Name/TIN DB search
    3) OpenAI fallback
    Returns a dict in the shapes your frontend expects.
    """
    previous_context = _get_context_from_session(request)
    user_input = name or tin

    if tin:
        charity = Charity.objects.filter(tin__iexact=tin).first()
        if charity:
            serializer = CharitySerializer(charity).data
            serializer["verified"] = True
            ai_output = f"Found {charity.name} by EIN."
            _update_context_session(request, user_input, ai_output)
            payload = {
                "source": "database",
                "via": "database",
                "message": "Found charity by EIN (no enrichment needed).",
                "matches": [serializer],
                "needs_clarification": False,
                "enrichment_done": False,
                "verified": True,
            }
            _store_last_matches(request, payload["matches"])
            return payload
        else:
            print(f"[FAST EIN LOOKUP] No match for EIN {tin}")
            openai_result = _search_with_openai(f'EIN (TIN) "{tin}"', previous_context)
            _update_context_session(
                request,
                user_input,
                openai_result.get("results", {}).get("explanation", ""),
            )
            if openai_result.get("via") == "openai":
                _store_last_matches(request, openai_result["results"]["matches"])
            openai_result["verified"] = False
            return openai_result

    matches, any_updated, needs_clarification = _perform_database_search(name, tin)
    if matches:
        msg = (
            "Single charity record found."
            if len(matches) == 1
            else f"Multiple charities found. Showing top {len(matches)}."
        )
        payload = {
            "source": "database",
            "via": "database",
            "message": msg,
            "matches": matches,
            "needs_clarification": needs_clarification,
            "enrichment_done": any_updated,
            "verified": True,
        }
        _store_last_matches(request, payload["matches"])
        _update_context_session(request, user_input, msg)
        return payload

    descriptor = f'name "{name}"' if name else f'EIN "{tin}"'
    openai_result = _search_with_openai(descriptor, previous_context)
    if openai_result.get("via") == "openai":
        _store_last_matches(request, openai_result["results"]["matches"])
    _update_context_session(
        request,
        user_input,
        openai_result.get("results", {}).get("explanation", ""),
    )
    openai_result["verified"] = False
    return openai_result

@csrf_exempt
@api_view(["POST"])
@permission_classes([permissions.AllowAny])
def ai_search_charity(request):
    """
    Direct search endpoint (legacy). Uses DB + OpenAI fallback.
    """
    print("[AI SEARCH]", request.data)
    name = (request.data.get("charity_name") or request.data.get("name") or "").strip()
    tin = (request.data.get("tin") or "").strip()

    if not name and not tin:
        return Response(
            {"error": "charity_name or EIN required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    result = _perform_search(name, tin, request)
    return Response(result, status=status.HTTP_200_OK)


# def _filter_results_basic(matches, message: str):
#     """Simple local filter: match address/location tokens."""
#     filtered = []
#     msg_lower = (message or "").lower()
#     tokens = [t for t in re.split(r"\s+", msg_lower) if t]

#     for m in matches:
#         combined = f"{m.get('address', '')} {m.get('location', '')}".lower()
#         if all(tok in combined for tok in tokens):
#             filtered.append(m)

#     return filtered


# def _clarify_with_openai(message: str, request, last_matches=None):
#     """Ask GPT for a clarification reply."""
#     previous_context = _get_context_from_session(request)
#     context_text = f"\nPrevious context:\n{previous_context}\n" if previous_context else ""
#     match_text = ""
#     if last_matches:
#         top = "\n".join(
#             [f"- {m.get('name')} ({m.get('location', '')})" for m in last_matches[:5]]
#         )
#         match_text = f"\nRecent results:\n{top}\n"

#     prompt = f"""
# You are a helpful assistant for verified US charities.

# User said: "{message}"
# {context_text}
# {match_text}

# If this seems like a refinement (e.g., specifying city, state, or road),
# respond with a brief clarification question.

# Do NOT invent charities or EINs.
# """

#     try:
#         completion = client.chat.completions.create(
#             model="gpt-4o-mini",
#             temperature=0.4,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a factual assistant for US charity data.",
#                 },
#                 {"role": "user", "content": prompt},
#             ],
#             timeout=15,
#         )
#         reply = completion.choices[0].message.content
#         _update_context_session(request, message, reply)
#         return {
#             "via": "openai-clarifier",
#             "reply": reply,
#         }
#     except Exception as e:
#         print(f"[OpenAI Clarifier Error] {e}")
#         return {
#             "via": "openai-clarifier",
#             "reply": "I had trouble processing that. Could you rephrase it?",
#         }
    
def _parse_query_intent_with_llm(message: str):
    """
    Uses GPT to determine what the user wants:
    - Whether it's a search, filter, or chat.
    - If it's a search, what charity name or EIN to look up.
    Returns structured JSON: {intent, charity_name, tin, filter_text}
    """
    if not message.strip():
        return {"intent": "chat", "charity_name": "", "tin": "", "filter_text": ""}

    prompt = f"""
You are an intent classifier for a charity search assistant.

Given this user message:
"{message}"

Classify it as one of:
- "search": when user wants to find or look up a charity by name or EIN.
- "filter": when user wants to narrow, refine, limit, or restrict previous results (e.g. by city, state, street, or type).
- "clarify": when user asks a follow-up question about previous results.
- "other": when message does not fit above.

Also extract any charity name or EIN number if present.
Return ONLY JSON like:
{{"intent": "search|filter|clarify|other", "charity_name": "...", "tin": "...", "filter_text": "..."}}
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a precise intent classification and extraction assistant."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=10,
        )
        parsed = json.loads(completion.choices[0].message.content)
        return parsed
    except Exception as e:
        print(f"[INTENT PARSER ERROR] {e}")
        return {"intent": "chat", "charity_name": "", "tin": "", "filter_text": ""}

def _strip_unverified_contacts(results):
    for r in results or []:
        if not r.get("verified"):
            r.pop("contactEmail", None)
            r.pop("contact_email", None)
            r.pop("contactTelephone", None)
            r.pop("contact_phone", None)
@csrf_exempt
@api_view(["POST"])
@permission_classes([permissions.AllowAny])
def ai_router(request):
    message = (request.data.get("message") or request.data.get("q") or "").strip()
    charity_name = (request.data.get("charity_name") or "").strip()
    tin = (request.data.get("tin") or "").strip()
    charities = request.data.get("charities", [])

    last_matches = _get_last_matches(request)

    if tin:
        print(f"[AI ROUTER] Direct DB lookup for name='{charity_name}', tin='{tin}'")
        matches, any_updated, needs_clarification = _perform_database_search(charity_name, tin)
        if matches:
            payload = {
                "via": "database",
                "source": "database",
                "verified": True,
                "matches": matches,
                "needs_clarification": needs_clarification,
                "message": f"Found {len(matches)} verified charities in the database.",
            }
            _store_last_matches(request, payload["matches"])
            _update_context_session(request, message or charity_name, payload["message"])
            return Response(payload, status=status.HTTP_200_OK)

        openai_result = _search_with_openai(f'name "{charity_name or tin}"')
        openai_result["verified"] = False
        openai_result["source"] = "openai"
        return Response(openai_result, status=status.HTTP_200_OK)

    parsed = _parse_query_intent_with_llm(message)
    print(f"[AI ROUTER] Parsed LLM intent: {parsed}")

    intent = parsed.get("intent")
    charity_name = parsed.get("charity_name", "").strip()
    tin = parsed.get("tin", "").strip()
    filter_text = parsed.get("filter_text", "").strip()
    

    # === 3️⃣ Handle LLM-decided intent ===
    # (a) Search intent → call DB search again
    # if intent == "search" and (charity_name or tin):
    #     lower_msg = message.lower()
    #     if any(k in lower_msg for k in ["filter", "only", "just", "show me", "restrict", "in ", "from "]):
    #         print("[AI ROUTER] Overriding intent to FILTER (keyword heuristic)")
    #         intent = "filter"
    #     print(f"[AI ROUTER] LLM extracted search target '{charity_name}' / '{tin}'")

    #     matches, any_updated, needs_clarification = _perform_database_search(charity_name, tin)
    #     if matches:
    #         payload = {
    #             "via": "database",
    #             "source": "database",
    #             "verified": True,
    #             "matches": matches,
    #             "needs_clarification": needs_clarification,
    #             "message": f"Found {len(matches)} verified charities from database via LLM intent.",
    #         }
    #         _store_last_matches(request, payload["matches"])
    #         _update_context_session(request, message, payload["message"])
    #         return Response(payload, status=status.HTTP_200_OK)

    #     # No DB match → use LLM to find unverified ones
    #     openai_result = _search_with_openai(f'name "{charity_name or tin}"')
    #     openai_result["verified"] = False
    #     openai_result["source"] = "openai"
    #     return Response(openai_result, status=status.HTTP_200_OK)

    if intent == "search":
        print(f"[AI ROUTER] Search intent detected: {parsed}")
        print(message)

        charity_name = parsed.get("charity_name", "").strip()
        tin = parsed.get("tin", "").strip()

        matches, any_updated, needs_clarification = _perform_database_search(charity_name, tin)

        history = request.data.get("history", [])
        if history and isinstance(history, list):
            context_from_frontend = "\n".join(
                [f"{h.get('role', 'user').title()}: {h.get('content', '')}" for h in history[-5:]]
            )
        else:
            context_from_frontend = ""
        print("History : ",history)
        combined_context = (
            (_get_context_from_session(request) or "")
            + "\n"
            + context_from_frontend
        ).strip()

        if matches:
            print(f"[AI ROUTER] Found {len(matches)} verified DB matches for '{charity_name}'")

            try:
                suggestion_prompt = f"""
                The user searched for "{charity_name}".Along with the query of {message}.
                Keep the query in mind as well(e.g. if a location is given give results based on that but don't limit yourself to just that.)
                Suggest other real US charities that might be related or similarly named
                (e.g., same denomination, nearby branches, or similar spelling variants).
                Return only real ones, not invented.

                Return JSON:
                {{
                "matches": [
                    {{
                    "name": "",
                    "location": "",
                    "website": "",
                    "address": "",
                    "tin": "",
                    "confidence": 0.0
                    }}
                ],
                "explanation": "short explanation"
                }}
                history : {history}
                """

                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.25,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a factual assistant listing real US charities. "
                                "Do not invent fake organizations."
                            ),
                        },
                        {"role": "user", "content": suggestion_prompt},
                    ],
                    response_format={"type": "json_object"},
                    timeout=20,
                )

                suggestions_raw = completion.choices[0].message.content
                suggestions = json.loads(suggestions_raw)
                similar_matches = suggestions.get("matches", [])
                explanation = suggestions.get("explanation", "")
                for s in similar_matches:
                    s["verified"] = False
                    s["source"] = "openai"

            except Exception as e:
                print(f"[AI SUGGESTION ERROR] {e}")
                similar_matches = []
                explanation = "No additional suggestions found."

            reply_message = (
                f"Found {len(matches)} verified charities for '{charity_name}'. "
                f"{explanation} "
                "If the one you want isn't listed here, tell me more about it — I can help you find it."
            )

            payload = {
                "via": "database+openai-suggest",
                "verified": True,
                "matches": matches,
                "related": similar_matches,
                "needs_clarification": needs_clarification,
                "message": reply_message,
            }
            _strip_unverified_contacts(similar_matches)
            
            
            _store_last_matches(request, matches + similar_matches)
            _update_context_session(request, message, reply_message)
            return Response(payload, status=status.HTTP_200_OK)

        print(f"[AI ROUTER] No verified match for '{charity_name}' — using OpenAI fallback search.")
        descriptor = ""
        if charity_name != "":
            descriptor = f'Find real US charity named "{charity_name}" (EIN optional).'
        if tin:
            descriptor += f"Heres the EIN - {tin} user searched for.Find real US charity."

        print(descriptor)

        openai_result = _search_with_openai(descriptor, combined_context)
        openai_result["verified"] = False
        openai_result["source"] = "openai"
        _strip_unverified_contacts(openai_result.get("results", {}).get("matches", []))
        print(openai_result)

        reply_message = (
            f"I couldn’t find a verified match for '{charity_name or tin}' in the database, "
            "but here are some real US charities that might be the one you mean. "
            "If none of these look right, tell me more about the charity — maybe its city, type, or purpose — "
            "and I can refine the search for you."
        )

        openai_result["message"] = openai_result["results"]["explanation"] or reply_message
        _store_last_matches(request, openai_result.get("results", {}).get("matches", []))
        _update_context_session(request, message, reply_message)

        return Response(openai_result, status=status.HTTP_200_OK)


    if intent == "filter":
        print("[AI ROUTER] Filter intent detected")

        # Prefer frontend-sent candidates if available
        last_matches = charities or _get_last_matches(request)

        if not last_matches:
            return Response({
                "via": "openai-filter",
                "message": "There are no existing results to filter. Try searching first.",
                "matches": []
            }, status=200)

        filter_text = parsed.get("filter_text") or message
        filtered = []
        reason = ""

        try:
            filter_prompt = f"""
    You are an intelligent charity data filter.
    The user wants to filter the charity list below based on this instruction:
    "{filter_text}"

    You will read the JSON list and return only those entries that match the intent.
    If there aren't any matching ones, you can say so — but do not invent new charities.

    RULES:
    - Never invent new charities or change data.
    - If uncertain, include slightly broader results.
    - Return only JSON in the format:
    {{"filtered": [ ...subset of original charities... ], "reason": "explain briefly"}}

    Here is the data (JSON list):
    {json.dumps(last_matches, indent=2)}
    """

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "You are a precise assistant that filters JSON lists."},
                    {"role": "user", "content": filter_prompt},
                ],
                response_format={"type": "json_object"},
                timeout=25,
            )

            result = completion.choices[0].message.content
            parsed_result = json.loads(result)
            filtered = parsed_result.get("filtered", [])
            reason = parsed_result.get("reason", "Filtered based on given instruction.")

            print(f"[AI FILTER] '{filter_text}' → {len(filtered)} filtered results")

        except Exception as e:
            print(f"[AI FILTER ERROR] {e}")
            return Response({
                "via": "openai-filter",
                "error": str(e),
                "matches": [],
                "message": "AI filter failed."
            }, status=500)

        if not filtered or "none" in reason.lower() or "no matching" in reason.lower():
            print(f"[AI FILTER] No reliable matches for '{filter_text}' → invoking OpenAI fallback search")

            base_name = ""
            if last_matches:
                top_names = list({m.get("name", "") for m in last_matches if m.get("name")})
                base_name = top_names[0] if top_names else ""
            elif _get_context_from_session(request):
                base_name = _get_context_from_session(request).split("\n")[-1].strip()

            descriptor = f'{base_name} {filter_text}'.strip()

            contextual_prompt = f"""
    User previously searched for charities related to "{base_name}".
    Now they specifically want results connected to "{filter_text}" (e.g., branches, locations, or affiliates in that area).

    Find real US-based organizations that match this refined query.

    Return only JSON:
    {{
    "matches": [
        {{
        "name": "",
        "location": "",
        "website": "",
        "address": "",
        "tin": "",
        "confidence": 0.0
        }}
    ],
    "needs_clarification": false,
    "explanation": "short summary"
    }}
    """

            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.25,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a cautious assistant that finds real, US-based charities without hallucination.",
                        },
                        {"role": "user", "content": contextual_prompt},
                    ],
                    response_format={"type": "json_object"},
                    timeout=25,
                )

                openai_raw = completion.choices[0].message.content
                openai_result = json.loads(openai_raw)
                new_matches = openai_result.get("matches", [])

                for m in new_matches:
                    m["verified"] = False
                    m["source"] = "openai"

                if new_matches:
                    reason += (
                        f" The requested branch or location ('{filter_text}') was not found in the verified database, "
                        "so I searched external sources and found possible matches (unverified)."
                    )
                    filtered = new_matches
                else:
                    reason += (
                        f" I could not locate any charities in '{filter_text}' even from external sources. "
                        "Try specifying the full organization name."
                    )

            except Exception as e:
                print(f"[AI FALLBACK FILTER ERROR] {e}")
                reason += " Fallback search failed due to an error."
                filtered = []

        _store_last_matches(request, filtered)
        _update_context_session(request, message, reason)
        
        _strip_unverified_contacts(filtered)

        print(f"[AI FILTER FINAL] '{filter_text}' → {len(filtered)} results total")
        return Response({
            "via": "openai-filter",
            "message": reason,
            "matches": filtered,
            "filter_text": filter_text
        }, status=200)



    # (c) Chat intent → use LLM conversationally
    reply = _chat_with_openai(message, request)
    return Response({"via": "openai-chat", "reply": reply}, status=status.HTTP_200_OK)

@csrf_exempt
@api_view(["POST"])
@permission_classes([permissions.AllowAny])
def ai_filter_charities(request):
    """
    Takes a list of charity dicts (from previous search) + user filter text,
    and uses GPT to return a filtered subset with reasoning.
    """
    try:
        data = request.data
        filter_text = data.get("filter_text", "").strip()
        charities = data.get("charities", [])
        if not filter_text or not charities:
            return Response({"error": "filter_text and charities are required."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Limit to 100 max to save tokens
        charities = charities[:100]

        # Build prompt
        prompt = f"""
You are an intelligent charity data filter.
The user wants to filter the charity list below based on this instruction:
"{filter_text}"

You will read the JSON list and return only those entries that match the intent.

RULES:
- Never invent new charities or change data.
- If uncertain, include slightly broader results.
- Return only JSON in the format:
{{"filtered": [ ...subset of original charities... ], "reason": "explain briefly"}}

Here is the data (JSON list):
{json.dumps(charities, indent=2)}
"""

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a precise assistant that filters JSON lists."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=25,
        )

        result = completion.choices[0].message.content
        parsed = json.loads(result)
        filtered = parsed.get("filtered", [])
        reason = parsed.get("reason", "Filtered based on given instruction.")

        print(f"[AI FILTER] '{filter_text}' → {len(filtered)} results")

        return Response({
            "via": "openai-filter",
            "message": reason,
            "matches": filtered,
            "filter_text": filter_text
        }, status=200)

    except Exception as e:
        print(f"[AI FILTER ERROR] {e}")
        return Response({
            "via": "openai-filter",
            "error": str(e),
            "matches": [],
            "message": "AI filter failed."
        }, status=500)