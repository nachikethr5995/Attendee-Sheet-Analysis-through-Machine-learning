"""Core modules for AIME Backend."""

# Fix for Windows Store Python - ensure user site-packages is in path
# This must run BEFORE any other imports that depend on installed packages
import sys
import site

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)










