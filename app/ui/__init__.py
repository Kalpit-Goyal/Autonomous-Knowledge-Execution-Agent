"""Streamlit client.

Kept out of the package's import path on purpose: this module is run by
``streamlit run app/ui/app.py`` and talks to the FastAPI service over HTTP
rather than importing the graph.
"""
