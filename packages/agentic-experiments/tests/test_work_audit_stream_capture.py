import asyncio

import httpx

from agentic_experiments.runners.run_kv_movement_interference import completion


def test_stream_capture_records_content_chunks_without_text():
    payload = (b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
               b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
               b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
               b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n'
               b'data: [DONE]\n\n')

    async def run():
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=payload))
        async with httpx.AsyncClient(transport=transport) as client:
            return await completion(client, base_url="http://test/v1", model="test",
                                    prompt="prompt", request_context={}, max_tokens=2,
                                    capture_chunks=True)

    result = asyncio.run(run())
    assert result["content_chunk_count"] == 2
    assert [item["characters"] for item in result["content_chunks"]] == [5, 6]
    assert result["completion_tokens"] == 2
    assert result["output_characters"] == 11
    assert result["first_token_ns"] <= result["content_chunks"][0]["ts_ns"]
    assert "hello" not in str(result)
