import os
import requests

from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from rest_framework import status
from django.shortcuts import get_object_or_404
import json, re, requests
from bs4 import BeautifulSoup
from ..models import Charity
from ..serializers import CharitySerializer

from ..models import Charity, Funding_Request, FundingRequestStatus
from ..serializers import CharitySerializer, FundingRequestSerializer
from ..permissions import IsLuciaAdmin
from .pagination import CharityPagination

class CharityPagination(PageNumberPagination):
    page_size = 50

@api_view(['POST','GET'])
@permission_classes([])
def create_charity(request):
    if request.method == 'POST':
        print(request.data)
        serializer = CharitySerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
        # charity = Charity.objects.get(many = True)
        # serialized_charity = CharitySerializer(charity, many=True).data
        # return Response(serialized_charity, status=status.HTTP_200_OK)
        charity = Charity.objects.all()
        paginator = CharityPagination()
        page = paginator.paginate_queryset(charity, request)
        serializer = CharitySerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

@api_view(['POST'])
@permission_classes([])  # public endpoint
def submit_funding_request(request):
    serializer = FundingRequestSerializer(data=request.data)
    if serializer.is_valid():
        # Force all new requests into "pending_vetting" status
        serializer.save(status=FundingRequestStatus.PENDING_VETTING)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([IsLuciaAdmin])
def list_all_funding_requests(request):
    requests = Funding_Request.objects.all()
    serializer = FundingRequestSerializer(requests, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([])  # anyone can view
def get_funding_request(request, id):
    funding_request = get_object_or_404(Funding_Request, id=id)
    serializer = FundingRequestSerializer(funding_request)
    return Response(serializer.data, status=status.HTTP_200_OK)

@api_view(["POST"])
@permission_classes([])  # public lookup
def find_charity(request):
    name = request.data.get("name", "").strip()
    tin = request.data.get("tin", "").strip()
    address = request.data.get("address", "").strip()

    # ---- 1️⃣  Check local Lucia DB ------------------------------------------
    
    exists = Charity.objects.filter(tin__iexact=tin).exists()
    # if not charity and name:
    #     charity = Charity.objects.filter(name__iexact=name).first()
    print(exists)
    return Response({"exists": exists}, status=200)
    # if charity:
    #     data = CharitySerializer(charity).data
    #     return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([]) 
def get_charities(request):
    charities = Charity.objects.all().order_by('id')
    paginator = CharityPagination()
    result_page = paginator.paginate_queryset(charities, request)
    serializer = CharitySerializer(result_page, many=True, context={'request': request})
    return paginator.get_paginated_response(serializer.data)

@api_view(['PATCH'])
@permission_classes([])
def update_charity(request, tin):
    try:
        charity = Charity.objects.get(tin=tin)
    except Charity.DoesNotExist:
        return Response({"error": "Charity not found"}, status=status.HTTP_404_NOT_FOUND)

    serializer = CharitySerializer(charity, data=request.data, partial=True)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



@api_view(["POST"])
@permission_classes([])
def ai_enrich_charity(request):
    """
    Triggered ONLY after user explicitly selects a charity.
    Uses Apify to extract contact info from the website.
    Returns UPDATED charity so frontend can refresh immediately.
    """
    charity_id = request.data.get("charity_id")
    website = request.data.get("website")

    if not website:
        return Response(
            {"error": "website is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    charity = None
    if charity_id:
        charity = get_object_or_404(Charity, id=charity_id)

        # Already enriched → return immediately
        if charity.contact_email or charity.contact_telephone:
            serializer = CharitySerializer(charity)
            return Response(
                {"charity": serializer.data, "source": "cached"},
                status=status.HTTP_200_OK,
            )

    APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
    if not APIFY_TOKEN:
        return Response(
            {"error": "APIFY_API_TOKEN not configured"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    run_url = (
        "https://api.apify.com/v2/acts/"
        "vdrmota~contact-info-scraper/run-sync-get-dataset-items"
    )

    payload = {
        "startUrls": [{"url": website}],
        "maxDepth": 1,
        "maxRequests": 5,
        "proxyConfiguration": {"useApifyProxy": True},
    }

    try:
        res = requests.post(
            run_url,
            params={"token": APIFY_TOKEN, "clean": "true"},
            json=payload,
            timeout=90,
        )
        res.raise_for_status()
        items = res.json() or []

        emails, phones = set(), set()

        for item in items:
            emails.update(e.lower() for e in item.get("emails", []))
            phones.update(item.get("phones", []))
            phones.update(item.get("phonesUncertain", []))

        email = next(iter(emails), None)
        phone = next(iter(phones), None)

        # Update DB only if charity exists
        if charity:
            updated_fields = []
            if email:
                charity.contact_email = email
                updated_fields.append("contact_email")
            if phone:
                charity.contact_telephone = phone
                updated_fields.append("contact_telephone")

            if updated_fields:
                charity.save(update_fields=updated_fields)

            serializer = CharitySerializer(charity)
            return Response(
                {"charity": serializer.data, "source": "apify"},
                status=status.HTTP_200_OK,
            )

        # AI-suggested charity → return enrichment only
        return Response(
            {
                "charity": {
                    "contact_email": email,
                    "contact_telephone": phone,
                },
                "source": "apify",
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

# @api_view(["POST"])
# @permission_classes([])
# def ai_enrich_charity(request):
#     """
#     Triggered ONLY after user explicitly selects a charity.
#     Uses Apify to extract contact info from the website.
#     Returns UPDATED charity so frontend can refresh immediately.
#     """
#     charity_id = request.data.get("charity_id")
#     website = request.data.get("website")

#     if not charity_id or not website:
#         return Response(
#             {"error": "charity_id and website are required"},
#             status=status.HTTP_400_BAD_REQUEST,
#         )

#     charity = get_object_or_404(Charity, id=charity_id)

#     # If already enriched, return serialized charity immediately
#     if charity.contact_email or charity.contact_telephone:
#         serializer = CharitySerializer(charity)
#         return Response(
#             {
#                 "charity": serializer.data,
#                 "source": "cached",
#             },
#             status=status.HTTP_200_OK,
#         )

#     APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
#     if not APIFY_TOKEN:
#         return Response(
#             {"error": "APIFY_API_TOKEN not configured"},
#             status=status.HTTP_500_INTERNAL_SERVER_ERROR,
#         )

#     run_url = (
#         "https://api.apify.com/v2/acts/"
#         "vdrmota~contact-info-scraper/run-sync-get-dataset-items"
#     )

#     payload = {
#         "startUrls": [{"url": website}],
#         "maxDepth": 1,
#         "maxRequests": 5,
#         "proxyConfiguration": {"useApifyProxy": True},
#     }

#     try:
#         res = requests.post(
#             run_url,
#             params={"token": APIFY_TOKEN, "clean": "true"},
#             json=payload,
#             timeout=90,
#         )
#         res.raise_for_status()
#         items = res.json() or []

#         emails = set()
#         phones = set()

#         for item in items:
#             for e in item.get("emails", []):
#                 emails.add(e.lower())
#             for p in item.get("phones", []):
#                 phones.add(p)
#             for p in item.get("phonesUncertain", []):
#                 phones.add(p)

#         email = next(iter(emails), None)
#         phone = next(iter(phones), None)

#         updated_fields = []

#         if email:
#             charity.contact_email = email
#             updated_fields.append("contact_email")

#         if phone:
#             charity.contact_telephone = phone
#             updated_fields.append("contact_telephone")

#         if updated_fields:
#             charity.save(update_fields=updated_fields)

#         # 🔑 THIS IS THE FIX
#         serializer = CharitySerializer(charity)

#         return Response(
#             {
#                 "charity": serializer.data,
#                 "source": "apify",
#             },
#             status=status.HTTP_200_OK,
#         )

#     except Exception as e:
#         return Response(
#             {"error": str(e)},
#             status=status.HTTP_500_INTERNAL_SERVER_ERROR,
#         )
