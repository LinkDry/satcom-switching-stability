"""RAG Knowledge Base for Satellite Communication Domain.

Provides domain-specific context to the LLM intent parser,
enabling better translation of operator commands to objective profiles.
"""

SATCOM_KNOWLEDGE = [
    {
        "topic": "emergency_response",
        "context": (
            "During emergency/disaster scenarios in satellite communications, "
            "the primary objective shifts to maximizing coverage and minimizing outage. "
            "Throughput per user can be reduced to ensure more users maintain connectivity. "
            "Typical weight profile: outage_tolerance=0.1 (very strict), throughput=0.3 (reduced), "
            "fairness=0.7 (high, ensure all beams served), queue=0.8 (clear backlogs fast). "
            "Switching cost should be low (0.2) to allow rapid reconfiguration."
        ),
        "keywords": ["emergency", "disaster", "crisis", "rescue", "SOS", "outage"],
    },
    {
        "topic": "maritime_operations",
        "context": (
            "Maritime satellite traffic is characterized by sparse, uniform demand across beams. "
            "Ships and offshore platforms need reliable but low-bandwidth connections. "
            "Fairness is critical — each vessel needs minimum guaranteed service. "
            "Typical profile: throughput=0.4, fairness=0.8, outage_tolerance=0.2, "
            "switching=0.3 (moderate stability), queue=0.3."
        ),
        "keywords": ["maritime", "ship", "vessel", "ocean", "offshore", "sea"],
    },
    {
        "topic": "urban_peak_traffic",
        "context": (
            "Urban peak hours create concentrated high demand in center beams. "
            "Goal is maximum aggregate throughput while maintaining acceptable outage rates. "
            "Some beam-level unfairness is acceptable if total system throughput is maximized. "
            "Typical profile: throughput=0.9, fairness=0.2, outage_tolerance=0.4, "
            "switching=0.5 (stability matters), queue=0.3."
        ),
        "keywords": ["urban", "city", "peak", "dense", "high demand", "capacity"],
    },
    {
        "topic": "iot_data_collection",
        "context": (
            "IoT data collection involves many low-bandwidth devices sending small packets. "
            "Queue management is critical — data must not be lost. Throughput per device is low "
            "but aggregate reliability matters. Fairness across beams is important. "
            "Typical profile: throughput=0.2, fairness=0.8, outage_tolerance=0.3, "
            "switching=0.2 (stable), queue=0.9 (queue clearance is top priority)."
        ),
        "keywords": ["iot", "sensor", "telemetry", "data collection", "m2m", "device"],
    },
    {
        "topic": "handover_transition",
        "context": (
            "During satellite handover or beam transition periods, minimizing service disruption "
            "is the top priority. Switching cost should be very high to prevent oscillation. "
            "Maintain current allocations as much as possible. "
            "Typical profile: throughput=0.5, fairness=0.4, outage_tolerance=0.3, "
            "switching=0.9 (minimize changes), queue=0.4."
        ),
        "keywords": ["handover", "transition", "switch", "migration", "stable", "maintain"],
    },
    {
        "topic": "energy_saving",
        "context": (
            "Energy-saving mode reduces active beams to conserve satellite power. "
            "Accept lower throughput and some fairness reduction to minimize power usage. "
            "Only serve beams with significant demand, turn off low-demand beams. "
            "Typical profile: throughput=0.3, fairness=0.3, outage_tolerance=0.6 (relaxed), "
            "switching=0.4, queue=0.2."
        ),
        "keywords": ["energy", "power", "save", "green", "efficient", "battery", "solar"],
    },
]


def retrieve_context(query: str, top_k: int = 2) -> str:
    """Simple keyword-based retrieval from knowledge base."""
    query_lower = query.lower()
    scored = []
    for entry in SATCOM_KNOWLEDGE:
        score = sum(1 for kw in entry["keywords"] if kw in query_lower)
        # Boost exact matches
        if any(kw in query_lower for kw in entry["keywords"]):
            score += 1
        scored.append((score, entry))

    scored.sort(key=lambda x: -x[0])
    top = [e for s, e in scored[:top_k] if s > 0]

    if not top:
        return ""

    context_parts = [f"[{e['topic']}]: {e['context']}" for e in top]
    return "\n\n".join(context_parts)
