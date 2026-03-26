"""
Test script for Workflow B: Deep Analysis (Automated End-to-End Report)

This script tests the full Deep Analysis pipeline by:
1. Generating a session ID
2. Associating a test user_id with the session (this is KEY — without a user_id,
   the Deep Analyzer falls back to plain agent name strings instead of DSPy Signatures)
3. Uploading a dataset
4. Triggering streaming deep analysis and printing real-time progress

Usage:
    python test_deep_analysis.py
"""

import requests
import json
import uuid

# Config
BASE_URL = "http://localhost:7860"
TEST_USER_ID = 1       # We inject this numeric user_id into the session
TEST_CHAT_ID = 1       # Dummy chat_id
FILE_PATH = "Housing.csv"
DATASET_NAME = "Housing_Data"
ANALYSIS_GOAL = (
    "Write a deep analysis report on the distribution of house prices. "
    "Explore how air conditioning, parking, and the number of bedrooms "
    "affect the final sale price. Include charts and statistical insights."
)


def run_deep_analysis():
    # ─── Step 1: Generate session ────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    headers = {"X-Session-ID": session_id}
    print(f"Session ID: {session_id}")

    # ─── Step 2: Initialize the session WITH a user_id ───────────────────────────
    # This is critical. Without user_id, the backend falls back to plain string
    # names for agents, which breaks dspy.Predict(). With a user_id set, it
    # loads proper DSPy Signature objects from the database.
    print("\n--- Setting user context in session ---")
    init_resp = requests.post(
        f"{BASE_URL}/initialize-session",
        json={
            "session_id": session_id,
            "user_id": TEST_USER_ID,
            "user_email": "test@example.com",
            "user_name": "Test User"
        }
    )
    if init_resp.status_code == 200:
        print(f"✅ Session initialized with user_id={TEST_USER_ID}")
    else:
        print(f"⚠️  Session init returned {init_resp.status_code}: {init_resp.text}")
        print("     Continuing anyway (set-message-info will also work)...")

    # Also inject via set-message-info as a belt-and-suspenders approach
    set_msg_resp = requests.post(
        f"{BASE_URL}/set-message-info",
        headers=headers,
        json={
            "user_id": TEST_USER_ID,
            "chat_id": TEST_CHAT_ID,
            "message_id": 1
        }
    )
    if set_msg_resp.status_code == 200:
        print(f"✅ user_id={TEST_USER_ID} injected via set-message-info")
    else:
        print(f"⚠️  set-message-info failed: {set_msg_resp.text}")

    # ─── Step 3: Upload the dataset ──────────────────────────────────────────────
    print(f"\n--- Uploading {FILE_PATH} ---")
    with open(FILE_PATH, "rb") as f:
        upload_resp = requests.post(
            f"{BASE_URL}/upload_dataframe",
            headers=headers,
            files={"file": (FILE_PATH, f, "text/csv")},
            data={
                "name": DATASET_NAME,
                "description": "Housing market dataset with pricing data",
                "columns": "price",     # At least one column name required
                "fill_nulls": "true",
                "convert_types": "true"
            }
        )
    if upload_resp.status_code != 200:
        print(f"❌ Upload failed: {upload_resp.text}")
        return
    print(f"✅ Dataset uploaded successfully")

    # ─── Step 4: Trigger Deep Analysis (streaming) ───────────────────────────────
    print(f"\n--- Triggering Deep Analysis ---")
    print(f"Goal: {ANALYSIS_GOAL[:80]}...")

    stream_headers = {**headers, "Accept": "text/event-stream"}
    with requests.post(
        f"{BASE_URL}/deep_analysis_streaming",
        headers=stream_headers,
        json={"goal": ANALYSIS_GOAL},
        stream=True
    ) as resp:
        if resp.status_code != 200:
            print(f"❌ Deep analysis request failed: {resp.status_code} {resp.text[:200]}")
            return

        for raw_line in resp.iter_lines():
            if raw_line:
                line = raw_line.decode("utf-8")
                # Skip SSE 'data:' prefixes if present
                if line.startswith("data:"):
                    line = line[5:].strip()
                try:
                    update = json.loads(line)
                    step = update.get("step", "?")
                    status = update.get("status", "?")
                    progress = update.get("progress", 0)
                    message = update.get("message", "")

                    print(f"\n[{progress:3d}%] {step.upper()} → {status}")
                    if message:
                        print(f"       {message}")

                    # Show a preview of content when available
                    if update.get("content"):
                        preview = str(update["content"])[:200].replace("\n", " ")
                        print(f"       Preview: {preview}...")

                    # Show error detail
                    if update.get("error"):
                        print(f"       ❌ ERROR: {update['error']}")

                    # Done!
                    if step == "conclusion" and status == "completed":
                        print("\n\n✅ DEEP ANALYSIS COMPLETE!")
                        final = update.get("final_result", {})
                        
                        # --- Save as Markdown Report ---
                        report_md = f"# Deep Analysis Report\n\n**Goal:** {final.get('goal', 'N/A')}\n\n"
                        report_md += "## Synthesized Insights\n"
                        for idx, section in enumerate(final.get('synthesis', []), 1):
                            report_md += f"### Section {idx}\n{section}\n\n"
                        
                        report_md += "## Final Conclusion\n"
                        report_md += final.get('final_conclusion', 'N/A')
                        
                        with open("deep_analysis_report.md", "w", encoding="utf-8") as f:
                            f.write(report_md)
                        
                        # --- Save full JSON for debugging ---
                        with open("deep_analysis_result.json", "w", encoding="utf-8") as f:
                            json.dump(final, f, indent=2)
                        
                        print(f"\n📄 Report saved to: {os.path.abspath('deep_analysis_report.md')}")
                        print(f"📊 Full data saved to: {os.path.abspath('deep_analysis_result.json')}")
                        
                        print(f"\n=== FINAL CONCLUSION ===\n{final.get('final_conclusion', 'N/A')}")

                except json.JSONDecodeError:
                    print(f"[raw] {line[:120]}")


if __name__ == "__main__":
    run_deep_analysis()
