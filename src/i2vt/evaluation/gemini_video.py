"""Gemini video scoring: both videos are uploaded whole, and the rubric is the system instruction."""

import logging
import time


def create_client(api_key, timeout):
    from google import genai

    return genai.Client(api_key=api_key, http_options={"timeout": int(timeout * 1000)})


def evaluate_video_pair(original, edited, prompt, *, api_key, model, timeout, parse_response):
    """Upload both videos once, wait for processing, score, and release the remote files.

    Gemini samples the frames itself from the uploaded videos.
    """
    uploaded = []
    with create_client(api_key, timeout) as client:
        try:
            for path in (original, edited):
                uploaded.append(client.files.upload(file=str(path)))
            deadline = time.monotonic() + timeout
            for index, video in enumerate(uploaded):
                while getattr(video.state, "name", video.state) == "PROCESSING":
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Gemini video processing timed out: {video.name}")
                    time.sleep(min(2, max(0, deadline - time.monotonic())))
                    video = client.files.get(name=video.name)
                if getattr(video.state, "name", video.state) != "ACTIVE":
                    raise RuntimeError(f"Gemini video processing failed: {video.name} ({video.state})")
                uploaded[index] = video

            last_error = None
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model,
                        config={"system_instruction": prompt},
                        contents=uploaded,
                    )
                    text = response.text or ""
                    scores = parse_response(text)
                    if scores:
                        return scores, text
                    raise ValueError("Gemini returned an invalid three-score response")
                except Exception as exc:
                    last_error = exc
                    if attempt < 2:
                        time.sleep(2)
            raise RuntimeError("Gemini scoring failed after three attempts") from last_error
        finally:
            # Also clean up the first upload if the second upload fails.
            for video in uploaded:
                try:
                    client.files.delete(name=video.name)
                except Exception:
                    logging.getLogger(__name__).warning("Could not delete Gemini file %s", video.name)
