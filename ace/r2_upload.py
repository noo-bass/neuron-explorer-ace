# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Copy a run folder to the R2 bucket, the long-term home for clips and results. Skips files already there at the same size.

    uv run ace/r2_upload.py runs/renderer_sweep_L12 renderer/L12

Reads R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET from the environment or ../../.env (lyras_code/.env).
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import boto3

ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
TYPES = {".ogg": "audio/ogg", ".wav": "audio/wav", ".json": "application/json", ".jsonl": "application/x-ndjson"}


def r2():
    for line in open(ENV):
        k, _, v = line.strip().partition("=")
        if k.startswith("R2_"):
            os.environ.setdefault(k, v.strip("'\""))
    u = urlparse(os.environ["R2_ENDPOINT"])  # the dashboard's URL may end in /<bucket>; the client wants the host only
    client = boto3.client("s3", endpoint_url=f"{u.scheme}://{u.netloc}", aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                          aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")
    return client, os.environ["R2_BUCKET"]


def main(local, prefix):
    s3, bucket = r2()
    there = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        there.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
    files = [os.path.join(d, f) for d, _, fs in os.walk(local) for f in fs]
    todo = [f for f in files if there.get(f"{prefix}/{os.path.relpath(f, local)}") != os.path.getsize(f)]
    print(f"{len(files)} files, {len(files) - len(todo)} already in R2, {len(todo)} to upload", flush=True)
    t0 = time.time()

    def put(f):
        s3.upload_file(f, bucket, f"{prefix}/{os.path.relpath(f, local)}",
                       ExtraArgs={"ContentType": TYPES.get(os.path.splitext(f)[1], "application/octet-stream")})

    with ThreadPoolExecutor(16) as pool:
        for n, _ in enumerate(pool.map(put, todo), 1):
            if n % 250 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} ({100 * n / len(todo):.0f}%), {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
