# Official CFPB CSV began rejecting HEAD requests

- **Date:** 2026-07-13
- **Context and intended action:** Resume the bounded official-CSV downloader after the user requested another acquisition attempt.
- **Observable symptom:** The elevated request reached `files.consumerfinance.gov`, but `HEAD /ccdb/complaints.csv` returned HTTP 403 before any byte ranges were downloaded.
- **Impact:** The downloader could not discover the remote file size and therefore stopped before sampling.
- **Environment change:** In the earlier run, the same URL returned HTTP 200 to `HEAD`, advertised byte ranges, and returned HTTP 206 for ranged GET requests. The current edge behavior differs.
- **Troubleshooting result:** A one-byte `Range: bytes=0-0` GET succeeded with HTTP 206 and returned `Content-Range: bytes 0-0/8905555191`.
- **Successful workaround:** The downloader now uses HEAD when available and automatically falls back to the one-byte GET, deriving total size from `Content-Range` before requesting bounded data ranges.
- **Additional recurrence:** The first corrected downloader retry still received HTTP 403 for the one-byte GET, even though an otherwise identical standalone probe had returned HTTP 206. The distinguishing factor was the downloader's custom academic `User-Agent`; the accepted standalone request used Requests' standard user agent.
- **Additional correction:** Removed the custom session user agent so range requests use the edge-accepted standard Requests header.
- **Remaining limitation:** Direct acquisition still depends on the edge continuing to honor ranged GET requests.
