from fastapi import Depends, HTTPException, status
from utils.utils import get_current_user, get_admin_user


def get_current_user_depends():
    return Depends(get_current_user)


def get_admin_user_depends():
    return Depends(get_admin_user)
