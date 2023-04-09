---
title: Aiproject
emoji: 📊
colorFrom: pink
colorTo: purple
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
---


# Milestone 2

Here is the link to the HF space:
https://huggingface.co/spaces/jbraha/aiproject

Other notes:
- the docker image was changed to python 3.8.9 to align withe HF deployment, so tensorflow was imported manually
- Git actions got weird: to use a milestone branch while also deploying to HF successfully, I have a git action automatically merging milestone-2 to the main branch and then pushing to the HF space
