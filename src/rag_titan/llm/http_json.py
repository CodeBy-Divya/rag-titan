"""
Send JSON to a web API and read the JSON reply, using only the standard library.

Cloud chat models (Anthropic, ...) are plain HTTPS APIs, so rag-titan needs no
SDK for them. Error messages show the service's own reason (like "invalid
x-api-key") but never the request headers, so an API key cannot leak into logs.
"""

import json
import urllib.error
import urllib.request


def post_json(url, payload, headers=None, timeout=None, service="The API"):
    """
    POST `payload` as JSON and return the decoded JSON reply.

    Args:
        url: full address, for example "https://api.anthropic.com/v1/messages".
        payload: dict to send.
        headers: extra headers, for example the API key header.
        timeout: seconds to wait, None waits as long as the model needs.
        service: name used in error messages, for example "Anthropic".

    Returns:
        The reply as a dict.
    """
    all_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    all_headers.update(headers or {})
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=all_headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"{service} returned HTTP {error.code}: {error_text(error)}"
        ) from None   # "from None": the original error holds the request and its key
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(
            f"Could not reach {service} at {url} ({error}). Check your internet "
            "connection and the address."
        ) from None

    try:
        return json.loads(text)
    except ValueError:
        raise RuntimeError(f"{service} sent a reply that is not JSON: {text[:200]!r}") from None


def error_text(error):
    """
    Read the reason from a failed reply.

    Understands {"error": {"message": ...}} (Anthropic, OpenAI style) and
    {"error": "..."}; anything else is returned as plain text.

    Args:
        error: the urllib HTTPError.
    """
    try:
        body = error.read().decode("utf-8")
    except OSError:
        return str(error.reason)

    try:
        detail = json.loads(body).get("error", body)
    except (ValueError, AttributeError):
        return body.strip() or str(error.reason)

    if isinstance(detail, dict):
        detail = detail.get("message") or json.dumps(detail)
    return str(detail)
