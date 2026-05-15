# ruff: noqa: T201
"""Live integration test: create_thread() against real Snowflake.

Verifies the empty payload fix works end-to-end by calling the
Threads API at /api/v2/cortex/threads on a live Snowflake account.

Usage:
    SNOWFLAKE_PAT=<your_pat> python tests/integration_tests/test_create_thread_live.py
"""

import os
import sys

import pytest
import requests

ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "")
PAT = os.getenv("SNOWFLAKE_PAT", "")

BASE_URL = f"https://{ACCOUNT}.snowflakecomputing.com"
THREADS_URL = f"{BASE_URL}/api/v2/cortex/threads"

HEADERS = {
    "Authorization": f"Bearer {PAT}",
    "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


skip_no_pat = pytest.mark.skipif(not PAT or not ACCOUNT, reason="SNOWFLAKE_ACCOUNT/SNOWFLAKE_PAT not set")


@skip_no_pat
def test_create_thread_empty_body():
    """Test 1: POST /api/v2/cortex/threads with empty JSON body {}.

    This is what create_thread() sends after the fix.
    Before the fix, no body was sent at all, causing a 400.
    """
    print("--- Test 1: POST with empty JSON body {} (the fix) ---")
    resp = requests.post(THREADS_URL, headers=HEADERS, json={}, timeout=30)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:500]}")

    if resp.status_code == 200:
        data = resp.json()
        thread_id = data.get("thread_id")
        print(f"SUCCESS: thread_id = {thread_id}")
        return thread_id
    else:
        print(f"FAILED: Expected 200, got {resp.status_code}")
        sys.exit(1)


@skip_no_pat
def test_create_thread_no_body():
    """Test 2: POST /api/v2/cortex/threads with NO body at all.

    This is what create_thread() sent BEFORE the fix (the bug).
    Expected: 400 Bad Request.
    """
    print("\n--- Test 2: POST with NO body (the bug) ---")
    headers_no_json = {k: v for k, v in HEADERS.items() if k != "Content-Type"}
    resp = requests.post(THREADS_URL, headers=headers_no_json, timeout=30)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:500]}")

    if resp.status_code == 400:
        print("CONFIRMED: No body → 400 (this was the bug)")
    else:
        print(f"Note: Got {resp.status_code} instead of 400")


@skip_no_pat
def test_create_thread_with_metadata():
    """Test 3: POST with metadata (the workaround that always worked)."""
    print("\n--- Test 3: POST with metadata (workaround) ---")
    payload = {"origin_application": "integration_test"}
    resp = requests.post(THREADS_URL, headers=HEADERS, json=payload, timeout=30)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:500]}")

    if resp.status_code == 200:
        data = resp.json()
        thread_id = data.get("thread_id")
        print(f"SUCCESS: thread_id = {thread_id}")
        return thread_id
    else:
        print(f"FAILED: Expected 200, got {resp.status_code}")
        sys.exit(1)


def cleanup_thread(thread_id):
    """Delete a test thread."""
    if not thread_id:
        return
    resp = requests.delete(f"{THREADS_URL}/{thread_id}", headers=HEADERS, timeout=30)
    print(f"Deleted thread {thread_id}: {resp.status_code}")


if __name__ == "__main__":
    if not PAT or not ACCOUNT:
        print("ERROR: Set SNOWFLAKE_ACCOUNT and SNOWFLAKE_PAT environment variables")
        sys.exit(1)

    print(f"Account: {ACCOUNT}")
    print(f"URL: {THREADS_URL}\n")

    tid1 = test_create_thread_empty_body()
    test_create_thread_no_body()
    tid2 = test_create_thread_with_metadata()

    print("\n--- Cleanup ---")
    cleanup_thread(tid1)
    cleanup_thread(tid2)

    print("\nAll tests passed!")
