"""
Utility helpers — formatting, timestamps, dummy data generation.
"""

from datetime import datetime
from typing import Any


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_interaction_row(record: dict[str, Any]) -> dict[str, str]:
    return {
        "User ID": str(record["user_id"]),
        "Item ID": str(record["item_id"]),
        "Title": record["title"],
        "Action": record["action"].upper(),
        "Timestamp": record["timestamp"],
    }


def generate_dummy_recommendations(user_id: int) -> list[dict[str, Any]]:
    """
    Simulates a recommendation fetch.
    Replace the body of this function with a real API call in Stage 2.
    """
    base_id = 101
    titles = [
        "Neural Architecture Search",
        "Transformer Attention Mechanisms",
        "Graph Neural Networks",
        "Contrastive Self-Supervised Learning",
        "Diffusion Models for Generation",
        "Reinforcement Learning from Feedback",
        "Sparse Mixture of Experts",
        "Retrieval-Augmented Generation",
        "Multimodal Vision-Language Models",
        "Efficient Fine-Tuning Methods",
    ]
    return [
        {"item_id": base_id + i, "title": title}
        for i, title in enumerate(titles)
    ]
