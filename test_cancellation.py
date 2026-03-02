#!/usr/bin/env python3
"""Cancellation behavior checks for the proxy's Anthropic-compatible endpoint."""

from __future__ import annotations

import asyncio
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:8082")
CLIENT_API_KEY = os.getenv("ANTHROPIC_API_KEY", "test-key")

DEFAULT_HEADERS = {
    "anthropic-version": "2023-06-01",
    "x-api-key": CLIENT_API_KEY,
}


async def test_non_streaming_cancellation() -> None:
    """Test cancellation for non-streaming requests."""
    print("Testing non-streaming request cancellation...")

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            task = asyncio.create_task(
                client.post(
                    f"{BASE_URL}/v1/messages",
                    headers=DEFAULT_HEADERS,
                    json={
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": 1000,
                        "messages": [
                            {
                                "role": "user",
                                "content": "Write a long story about a journey through space.",
                            }
                        ],
                    },
                )
            )

            await asyncio.sleep(0.5)
            task.cancel()

            try:
                response = await task
                print(
                    f"Non-streaming request finished before cancellation (status {response.status_code})"
                )
                if response.status_code == 401:
                    print(
                        "Non-streaming request returned 401; check ANTHROPIC_API_KEY alignment between client and proxy"
                    )
            except asyncio.CancelledError:
                print("Non-streaming request cancelled successfully")
        except Exception as error:
            print(f"Non-streaming test error: {error}")


async def test_streaming_cancellation() -> None:
    """Test cancellation for streaming requests."""
    print("\nTesting streaming request cancellation...")

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            async with client.stream(
                "POST",
                f"{BASE_URL}/v1/messages",
                headers=DEFAULT_HEADERS,
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a long story about a journey through space.",
                        }
                    ],
                    "stream": True,
                },
            ) as response:
                if response.status_code != 200:
                    print(f"Streaming request failed: {response.status_code}")
                if response.status_code == 401:
                    print(
                        "Streaming request returned 401; check ANTHROPIC_API_KEY alignment between client and proxy"
                    )
                    return

                print("Streaming request started successfully")

                chunk_count = 0
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk_count += 1
                    print(f"Received chunk {chunk_count}: {line[:120]}...")
                    if chunk_count >= 3:
                        print("Simulating client disconnect")
                        break

                print("Streaming request cancelled successfully")
        except Exception as error:
            print(f"Streaming test error: {error}")


async def test_server_running() -> bool:
    """Test if the server is running."""
    print("Checking if server is running...")

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("Server is running and healthy")
                return True
            print(f"Server health check failed: {response.status_code}")
            return False
    except Exception as error:
        print(f"Cannot connect to server: {error}")
        print("Make sure to start the server with: uv run clay")
        return False


async def test_openai_path_auth() -> None:
    """Sanity-check OpenAI auth path to avoid false cancellation negatives."""
    print("\nChecking OpenAI auth path...")

    headers = {
        "authorization": f"Bearer {CLIENT_API_KEY}",
        "content-type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
        )

    if response.status_code in {200, 400, 429, 500, 502, 503, 504}:
        print(f"OpenAI auth sanity check passed (status {response.status_code})")
        return

    if response.status_code == 401:
        print("OpenAI auth sanity check failed (401)")
        return

    print(f"OpenAI auth sanity check returned unexpected status {response.status_code}")


async def main() -> None:
    """Main test function."""
    print("Starting HTTP request cancellation tests")
    print("=" * 50)

    if not await test_server_running():
        return

    print("\n" + "=" * 50)
    await test_non_streaming_cancellation()
    await test_streaming_cancellation()
    await test_openai_path_auth()

    print("\n" + "=" * 50)
    print("All cancellation tests completed")


if __name__ == "__main__":
    asyncio.run(main())
