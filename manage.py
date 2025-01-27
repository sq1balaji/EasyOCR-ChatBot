#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
# import warnings
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

# # Suppress specific warnings
# warnings.filterwarnings("ignore", category=UserWarning)  # Ignores UserWarnings
# warnings.filterwarnings("ignore", category=FutureWarning)  # Ignores FutureWarnings
# #warnings.filterwarnings("ignore", category=InconsistentVersionWarning)  # Ignores InconsistentVersionWarnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignores DeprecationWarnings
# warnings.filterwarnings("ignore", category=ResourceWarning)  # Ignores ResourceWarnings



def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pdf_analysis_project.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
