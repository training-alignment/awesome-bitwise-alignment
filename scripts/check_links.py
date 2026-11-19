#!/usr/bin/env python3
"""Check every http(s) link in a Markdown file. Standard library only.

    python3 scripts/check_links.py README.md

Prints one line per link: ok / redirect / blocked (401/403/429, often bot
protection: check in a browser) / dead (404/410, DNS failure) / error.
Exits 1 if any link is dead.
"""
import concurrent.futures as cf
import re
import ssl
import sys
import urllib.error
import urllib.request

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
LINK = re.compile(r"\]\((https?://[^)\s]+)\)|<(https?://[^>\s]+)>")


def check(url):
    ctx = ssl.create_default_context()
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method, headers={"User-Agent": UA, "Accept": "*/*"})
        try:
            with urllib.request.urlopen(req, timeout=25, context=ctx) as r:
                final = r.geturl()
                return ("ok" if final.rstrip("/") == url.rstrip("/") else "redirect", r.status, final)
        except urllib.error.HTTPError as e:
            if method == "HEAD" and e.code in (400, 403, 404, 405, 429, 500, 501, 503):
                continue  # many servers reject HEAD; retry with GET
            if e.code in (401, 403, 429, 999):
                return ("blocked", e.code, url)
            if e.code in (404, 410):
                return ("dead", e.code, url)
            return ("error", e.code, url)
        except Exception as e:  # DNS, TLS, timeout
            if method == "HEAD":
                continue
            return ("dead" if "Name or service" in str(e) or "nodename" in str(e) else "error", 0, f"{url} ({e})")
    return ("error", 0, url)


def main(path):
    text = open(path, encoding="utf-8").read()
    urls = sorted({a or b for a, b in LINK.findall(text)})
    bad = 0
    with cf.ThreadPoolExecutor(16) as ex:
        for url, (state, code, info) in zip(urls, ex.map(check, urls)):
            bad += state == "dead"
            print(f"{state:8} {code:3} {url}" + (f" -> {info}" if info != url else ""))
    print(f"\n{len(urls)} links, {bad} dead", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "README.md"))
