"""WSGI entrypoint for hosting platforms."""

from app import app

# Some platforms look for `application`
application = app
