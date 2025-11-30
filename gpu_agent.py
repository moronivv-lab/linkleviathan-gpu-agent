"""
GPU Infrastructure Matching AI Agent
=====================================
An AI brokerage agent that matches user GPU requirements with available resources.

Features:
- Natural language query parsing (e.g., "150 H100 <$1")
- Mock API data loading and filtering
- ZK-proof stub generation (timestamp + SHA256 hash)
- DePIN rewards calculation
- JSON output format
- Mock Stripe escrow simulation (10-15% commission)

Usage (Console Mode):
    python gpu_agent.py
    Then enter a query like: "150 H100 <$1" or "200 A100 <$0.60"

Usage (Web Mode for URL/Zapier):
    Deploy with gunicorn gpu_agent:app
    Call POST /run with {"query": "your query"}
"""

import os
import requests
import json
import hashlib
import time
import re
from datetime import datetime
from typing import Optional

from flask import Flask, request, jsonify


# ---------------------------
# Data loading & core logic
# ---------------------------

def load_gpu_resources(filepath: str = "mock_apis.json") -> list:
    """
    Load GPU resources from a JSON file.
    """
    try:
        with open(filepath, "r") as file:
            resources = json.load(file)
            print(f"[INFO] Loaded {len(resources)} GPU resources from {filepath}")
            return resources
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {filepath}: {e}")
        raise


def parse_query(query: str) -> dict:
    """
    Parse a natural language query to extract GPU requirements.
    Improved for phrases like "under $0.6" or "max $1".
    """
    result = {
        "hours": None,
        "gpu_type": None,
        "max_price": None,
        "raw_query": query,
    }

    query_lower = query.lower()
    query_clean = query.replace("$", "").replace(",", "")

    # Detect GPU type
    gpu_types = ["H100", "A100", "RTX4090", "RTX3090", "V100", "A6000", "A10", "L40"]
    for gpu in gpu_types:
        if gpu.lower() in query_lower:
            result["gpu_type"] = gpu.upper()
            break

    # Extract hours
    hours_match = re.search(
        r"(\d+)\s*(hours?|hrs?|h|for training)?",
        query_clean,
        re.IGNORECASE,
    )
    if hours_match:
        result["hours"] = int(hours_match.group(1))

    # Extract max price
    price_match = re.search(
        r"(?:<|under|below|max|less than)\s*\$?\s*(\d+\.?\d*)",
        query_clean,
        re.IGNORECASE,
    )
    if price_match:
        result["max_price"] = float(price_match.group(1))

    print(
        f"[INFO] Parsed query: hours={result['hours']}, "
        f"gpu={result['gpu_type']}, max_price=${result['max_price']}"
    )
    return result


def filter_resources(resources: list, requirements: dict) -> list:
    """
    Filter GPU resources based on parsed requirements.
    """
    matches = []
    for resource in resources:
        # Skip non-available
        if "status" in resource and resource["status"] != "available":
            continue

        # GPU type
        if requirements["gpu_type"] and resource.get("gpu", "").upper() != requirements["gpu_type"]:
            continue

        # Price
        if (
            requirements["max_price"] is not None
            and resource.get("price_per_hour", float("inf")) > requirements["max_price"]
        ):
            continue

        # Hours
        if (
            requirements["hours"] is not None
            and resource.get("available_hours", 0) < requirements["hours"]
        ):
            continue

        matches.append(resource)

    # Sort cheapest first
    matches.sort(key=lambda x: x.get("price_per_hour", float("inf")))
    print(f"[INFO] Found {len(matches)} matching resources")
    return matches


def generate_zk_proof_stub(resource: dict) -> dict:
    """
    Generate a lightweight ZK-proof stub for verification.
    """
    timestamp = time.time()
    timestamp_iso = datetime.fromtimestamp(timestamp).isoformat()

    proof_data = (
        f"{resource.get('id', 'unknown')}|"
        f"{resource.get('provider', 'unknown')}|"
        f"{resource.get('gpu', 'unknown')}|"
        f"{resource.get('price_per_hour', 0)}|"
        f"{timestamp}"
    )

    hash_object = hashlib.sha256(proof_data.encode())
    proof_hash = hash_object.hexdigest()
    proof_id = f"ZKP-{resource.get('id', 'unknown')}-{int(timestamp)}"

    return {
        "proof_id": proof_id,
        "timestamp": timestamp_iso,
        "timestamp_unix": timestamp,
        "hash": proof_hash,
        "algorithm": "SHA256",
        "verified": True,
        "proof_data_format": "resource_id|provider|gpu|price|timestamp",
    }


def calculate_depin_rewards(resource: dict, hours_requested: Optional[int] = None) -> dict:
    """
    Calculate DePIN rewards.
    """
    base_points = 10
    hours = hours_requested if hours_requested else resource.get("available_hours", 0)
    # 10 points per 10 hours, cap 100
    hour_bonus = min(hours // 10 * 10, 100)
    price = resource.get("price_per_hour", 1.0)

    if price < 0.50:
        efficiency_bonus = 20
    elif price < 0.80:
        efficiency_bonus = 10
    elif price < 1.00:
        efficiency_bonus = 5
    else:
        efficiency_bonus = 0

    total_points = base_points + hour_bonus + efficiency_bonus
    token_estimate = total_points * 0.01

    return {
        "base_points": base_points,
        "hour_bonus": hour_bonus,
        "efficiency_bonus": efficiency_bonus,
        "total_points": total_points,
        "estimated_tokens": round(token_estimate, 4),
        "reward_tier": "Gold"
        if total_points >= 100
        else "Silver"
        if total_points >= 50
        else "Bronze",
    }


def test_escrow(matches_json: str, commission_rate: float = 0.15) -> None:
    """
    Mock Stripe escrow simulation with 10-15% commission.
    Prints to console/logs only (not used in web response).
    """
    try:
        matches = json.loads(matches_json)
        if not matches:
            print("No deal to escrow.")
            return

        first = matches[0]
        deal_value = first["resource"]["price_per_hour"] * first.get(
            "hours_requested",
            first["resource"].get("available_hours", 100),
        )
        commission = deal_value * commission_rate
        provider_share = deal_value - commission

        print("\nEscrow Simulation:")
        print(f"Deal Value: ${deal_value:.2f}")
        print(
            f"Provider Share: ${provider_share:.2f} "
            f"({(1 - commission_rate) * 100:.0f}%)"
        )
        print(f"Your Commission: ${commission:.2f} ({commission_rate * 100:.0f}%)")
        print("Client Payment Link (Test): https://buy.stripe.com/test")
    except Exception as e:
        print(f"Escrow error: {e}")


def calculate_match_score(resource: dict, requirements: dict) -> float:
    """
    Calculate match score (0-100).
    """
    score = 50.0

    # Price vs max price
    if requirements["max_price"]:
        price_ratio = resource["price_per_hour"] / requirements["max_price"]
        score += max(0, (1 - price_ratio) * 30)

    # Availability vs requested hours
    if requirements["hours"]:
        availability_ratio = resource["available_hours"] / requirements["hours"]
        score += min(20, (availability_ratio - 1) * 10)

    return round(min(100, max(0, score)), 1)


def format_match_result(resource: dict, rank: int, requirements: dict) -> dict:
    """
    Format a single match for output.
    """
    hours = requirements["hours"] or resource["available_hours"]
    # For escrow simulation
    resource["hours_requested"] = hours

    return {
        "rank": rank,
        "resource": {
            k: v
            for k, v in resource.items()
            if k
            in [
                "id",
                "provider",
                "gpu",
                "price_per_hour",
                "available_hours",
                "location",
                "memory_gb",
            ]
        },
        "zk_proof": generate_zk_proof_stub(resource),
        "depin_rewards": calculate_depin_rewards(resource, hours),
        "match_score": calculate_match_score(resource, requirements),
    }


def fetch_vast_resources() -> list:
    """
    Example: fetch live GPU offers from a real provider (e.g. Vast.ai).
    This version is defensive:
    - If there is no API key, it returns [].
    - If the API call fails, it prints a warning and returns [].
    You still keep your mock_apis.json as a fallback.
    """
    api_key = os.getenv("VAST_API_KEY")
    if not api_key:
        print("[WARN] VAST_API_KEY not set – skipping Vast.ai live data.")
        return []

    try:
        # Example endpoint – adjust according to the provider's docs.
        url = "https://console.vast.ai/api/v0/benchmarks/"  # or the search/instances endpoint they document
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # For v1, we can keep params minimal and let your existing filter handle prices/gpu types.
        params = {
            # You can later add query filters here if the API supports them
            # e.g. "gpu_name": "H100", "max_price": 1.0, ...
        }

        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        resources = []

        # You need to adjust this loop based on the actual JSON shape.
        # Example shape: data might be { "offers": [ {...}, {...} ] } or { "data": [ ... ] }
        offers = data.get("offers") or data.get("data") or []

        for item in offers:
            # TODO: inspect one response in logs to map real field names correctly.
            # For now we assume these example keys – you will adjust them once you see real JSON.
            resources.append({
                "id": f"vast-{item.get('id')}",
                "provider": "Vast.ai",
                "gpu": item.get("gpu_name") or item.get("gpu_type", "Unknown"),
                "price_per_hour": float(item.get("price_per_hour", 0)),
                "available_hours": int(item.get("available_hours", 100)),
                "location": item.get("location", "Unknown"),
                "memory_gb": item.get("gpu_mem_gb") or item.get("mem_gb", None),
                "status": "available",
            })

        print(f"[INFO] Loaded {len(resources)} live GPU resources from Vast.ai")
        return resources

    except Exception as e:
        print(f"[WARN] Could not fetch Vast.ai resources: {e}")
        return []


def run_agent(query: str, top_n: int = 3) -> dict:
    """
    Main agent function.
    """
    try:
        # 1) Load resources and compute matches as before
        resources = load_gpu_resources()
        requirements = parse_query(query)
        matches = filter_resources(resources, requirements)

        # Base output structure
        output: dict = {
            "success": True,
            "query": query,
            "parsed_requirements": requirements,
            "total_matches": len(matches),
            "matches": [],
            "generated_at": datetime.now().isoformat(),
            "agent_version": "1.0.1",
        }

        # No matches → return early
        if not matches:
            output["message"] = "No matching GPU resources found."
            return output

        # 2) Build the normal matches list (this is what you already had)
        top_matches = matches[:top_n]
        results = [format_match_result(res, i + 1, requirements)
                   for i, res in enumerate(top_matches)]

        output["showing"] = len(results)
        output["matches"] = results

        # 3) EXTRA: add flat fields for Zapier (best/second/third match)
        def add_flat_match(prefix: str, match: dict):
            res = match["resource"]
            output[f"{prefix}_provider"] = res.get("provider")
            output[f"{prefix}_gpu"] = res.get("gpu")
            output[f"{prefix}_price_per_hour"] = res.get("price_per_hour")
            output[f"{prefix}_location"] = res.get("location")
            output[f"{prefix}_available_hours"] = res.get("available_hours")
            output[f"{prefix}_memory_gb"] = res.get("memory_gb")
            output[f"{prefix}_match_score"] = match.get("match_score")
            output[f"{prefix}_rank"] = match.get("rank")

        if len(results) >= 1:
            add_flat_match("best_match", results[0])
        if len(results) >= 2:
            add_flat_match("second_match", results[1])
        if len(results) >= 3:
            add_flat_match("third_match", results[2])

        return output

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


def print_results(output: dict):
    """
    Pretty print results for console mode.
    """
    if not output.get("success"):
        print(f"[ERROR] {output.get('error')}")
        return

    if output["total_matches"] == 0:
        print(output.get("message"))
        return

    print(
        f"Found {output['total_matches']} matches, "
        f"showing top {output['showing']}:"
    )

    for match in output["matches"]:
        res = match["resource"]
        print(f"\n#{match['rank']} - {res['provider']}")
        print(f" GPU: {res['gpu']} ({res.get('memory_gb', 'N/A')}GB)")
        print(f" Price: ${res['price_per_hour']}/hour")
        print(f" Available: {res['available_hours']} hours")
        print(f" Location: {res.get('location', 'N/A')}")
        print(f" Match Score: {match['match_score']}/100")
        print(
            f" DePIN Rewards: {match['depin_rewards']['total_points']} "
            f"points ({match['depin_rewards']['reward_tier']})"
        )
        print(f" ZK-Proof ID: {match['zk_proof']['proof_id']}")

    print("\nJSON OUTPUT:")
    print(json.dumps(output, indent=2))


def main():
    """
    Interactive console mode.
    """
    print("\nGPU INFRASTRUCTURE MATCHING AI AGENT - v1.0.1")
    print("Example queries: '150 H100 <$1' or 'Need 200 A100 under $0.6'")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Enter GPU query: ").strip()
        if query.lower() in ["quit", "exit"]:
            break
        if not query:
            print("[WARNING] Enter a valid query.")
            continue

        output = run_agent(query)
        print_results(output)

        # Run escrow on JSON (console only)
        test_escrow(json.dumps(output.get("matches", [])))


# ---------------------------------
# Flask Web App for URL/Zapier/etc.
# ---------------------------------

app = Flask(__name__)


@app.route("/", methods=["GET"])
def health():
    """
    Simple health check so going to / in the browser doesn't show 404.
    """
    return jsonify(
        {
            "status": "ok",
            "agent_version": "1.0.1",
        }
    ), 200


@app.route("/run", methods=["POST"])
def webhook_run():
    """
    Main endpoint for Zapier / Framer.
    Expects JSON: { "query": "<client text>" }
    Returns the full agent output as JSON.
    """
    try:
        data = request.get_json() or {}
        query = (data.get("query") or "").strip()

        if not query:
            return jsonify(
                {
                    "success": False,
                    "error": "No query provided",
                }
            ), 400

        # Call your real agent
        output = run_agent(query)

        # Optional: run escrow simulation on the matches (logs only)
        # test_escrow(json.dumps(output.get("matches", [])))

        return jsonify(output), 200

    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
            }
        ), 500


if __name__ == "__main__":
    # Console mode (for local testing in a terminal):
    main()
    # If you ever want to run the HTTP server locally instead, comment main()
    # above and uncomment this:
    # app.run(host="0.0.0.0", port=8080, debug=True)
