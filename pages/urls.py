from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views 
# from .views.donation_details import get_donation_by_id
# from .views.donation_details import DonationDetailView


router = DefaultRouter()
router.register(r'dafs', views.DAFViewSet, basename='daf')
router.register(r'donations', views.DonationViewSet, basename='donation')
router.register(r'charities', views.CharityViewSet, basename='charity')

urlpatterns = [
    # --- Authentication Endpoints ---
    path('auth/login/', views.login_view, name='login'),
    path('auth/logout/', views.logout_view, name='logout'),
    path('auth/register/', views.register_user_view, name='register'),
    path('password-reset/', views.password_reset_request_view, name='password_reset'),
    path('password-reset/confirm/', views.password_reset_confirm_view, name='password_reset_confirm'),

    # --- Donor-Specific Endpoints ---
    path('dashboard/', views.donor_dashboard_view, name='dashboard'),
    path('dashboard/update-goal/', views.update_goal_view, name='update-goal'),
    path('donations/<uuid:donation_id>/', views.get_donation_by_id, name='get_donation_by_id'),
    path("director-dashboard/", views.director_dashboard_view, name="director-dashboard"),
    path('donations/<uuid:donation_id>/documents/', views.upload_donation_document, name='upload-donation-document'),

    # path('donations/<uuid:pk>/', DonationDetailView.as_view(), name='donation-detail'),
    path('donations/', views.create_donation, name='create_donation'),
    path('donations/<uuid:id>/status/', views.update_donation_status, name='update_donation_status'),
    path('<uuid:id>/votes/', views.cast_vote, name='cast_vote'),
    path("charities/", views.create_charity, name="create-charity"),
    path("funding-requests/", views.submit_funding_request, name="submit-funding-request"),
    path("funding-requests/all/", views.list_all_funding_requests, name="list-all-funding-requests"),
    path("funding-requests/<uuid:id>/", views.get_funding_request, name="get-funding-request"),

    path('admin-dashboard/', views.admin_dashboard, name='admin-dashboard'),
    path('findcharity/', views.find_charity, name="find_charity"),
    path('charities/<str:tin>/', views.update_charity, name='update_charity'),

    # path('chatbot/', views.NLQueryAPIView.as_view(), name='chatbot'),
    path('chatbot/', views.NLQueryPandasAPIView.as_view(), name='chatbot'),
    path("verify-charity/", views.verify_charity, name="verify_charity"),

    path("help-form/", views.help_form_view, name="help-form"),
    path('get_charities/', views.get_charities, name='get_charities'),
    path("ai-enrich/", views.ai_enrich_charity, name="ai_enrich_charity"),

    path('get_donations/', views.get_donations, name='get_charities'),
    path("director/", views.director_view, name="director"),
    # path("lookup/",views.search_and_update_charity, name="lookup"),
    path("ai-search/", views.ai_search_charity, name="director"),
    path("ai/", views.ai_router, name="ai_router"),
    path("ai-filter/", views.ai_filter_charities, name="ai_filter_charities"),

    # --- General API Endpoints ---
    path('api/', include(router.urls)),
]
