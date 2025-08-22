from rest_framework import permissions
from .models import UserRole

class IsLuciaAdmin(permissions.BasePermission):
    """ Allows full access only to users with the Lucia Admin role. """
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role == UserRole.LUCIA_ADMIN

class IsLuciaDirector(permissions.BasePermission):
    """ Allows access to users with the Lucia Director role. """
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role == UserRole.LUCIA_DIRECTOR

class IsDonorAdvisor(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role == UserRole.DONOR_ADVISOR
    
class IsOwnerOfObject(permissions.BasePermission):
    """
    Object-level permission to only allow owners of an object to view/edit it.
    Assumes the model instance has a 'recommending_user' attribute.
    """
    def has_object_permission(self, request, view, obj):
        # Allow access if the user is the one who recommended the donation.
        if hasattr(obj, 'recommending_user'):
            return obj.recommending_user == request.user
        # Allow access if the user is an advisor for the DAF.
        if hasattr(obj, 'advisors'):
            return request.user in obj.advisors.all()
        return False