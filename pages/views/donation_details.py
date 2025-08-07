# views/donation_views.py
from rest_framework.generics import RetrieveAPIView
from rest_framework.response import Response
from rest_framework import status
from ..models import Donation
from ..serializers import DonationReadSerializer

class DonationDetailView(RetrieveAPIView):
    queryset = Donation.objects.all()
    serializer_class = DonationReadSerializer

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        print(serializer.data)  # ✅ This will print the actual serialized data to the console
        return Response(serializer.data, status=status.HTTP_200_OK)


# # @api_view(['GET'])
# def get_donation_by_id(request, donation_id):
#     try:
#         donation = Donation.objects.get(id=donation_id)
#         serializer = DonationReadSerializer(donation)
#         return Response(serializer.data, status=status.HTTP_200_OK)
#     except Donation.DoesNotExist:
#         return Response({"error": "Donation not found."}, status=status.HTTP_404_NOT_FOUND)
