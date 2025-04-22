"""
plugin_reviews.py

Simple local storage for plugin ratings and reviews (JSON file).
"""
import json
import os
from datetime import datetime

REVIEWS_PATH = "plugin_reviews.json"


def load_reviews():
    if not os.path.exists(REVIEWS_PATH):
        return {}
    with open(REVIEWS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_reviews(data):
    with open(REVIEWS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def add_review(ptype, name, rating, review, user="anonymous"):
    data = load_reviews()
    key = f"{ptype}:{name}"
    if key not in data:
        data[key] = []
    data[key].append({
        "rating": rating,
        "review": review,
        "user": user,
        "timestamp": datetime.utcnow().isoformat()
    })
    save_reviews(data)

def get_reviews(ptype, name):
    data = load_reviews()
    key = f"{ptype}:{name}"
    return data.get(key, [])

def get_average_rating(ptype, name):
    reviews = get_reviews(ptype, name)
    if not reviews:
        return None
    return sum(r["rating"] for r in reviews) / len(reviews)
