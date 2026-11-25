"""Make HTTPS work behind virus scanners and company proxies that inspect secure connections.

Such software re-signs every HTTPS connection with its own certificate, which Windows trusts (it is in
the Windows certificate store) but the certificate lists shipped with Python packages do not. `urllib`
already uses the Windows store; `requests` (IIIF download, LLM check) and `openai` / `httpx` (HTR) use
the `certifi` list instead and fail with CERTIFICATE_VERIFY_FAILED.

`setup()` writes one certificate file with the certifi list plus every certificate in the Windows
store, and points both libraries at it (REQUESTS_CA_BUNDLE for requests, SSL_CERT_FILE for httpx).
No extra package is needed, so this also works on a computer where pip itself cannot get through.
On other systems, or when the user has set those variables already, it changes nothing.
"""
from __future__ import annotations

import hashlib
import os
import ssl
import sys
import tempfile
from pathlib import Path


def _windows_pem() -> str:
    pems = []
    for store in ("ROOT", "CA"):
        try:
            for der, enc, trust in ssl.enum_certificates(store):
                if enc == "x509_asn":
                    pems.append(ssl.DER_cert_to_PEM_cert(der))
        except (OSError, AttributeError):
            pass
    return "".join(pems)


def setup() -> str | None:
    """Returns the path of the combined certificate file, or None when nothing was changed."""
    if sys.platform != "win32" or os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE"):
        return None
    try:
        import certifi
        bundle = Path(certifi.where()).read_text(encoding="ascii", errors="ignore")
    except Exception:  # noqa: BLE001 - no certifi: the Windows store alone
        bundle = ""
    pem = bundle + "\n" + _windows_pem()
    path = Path(tempfile.gettempdir()) / f"navapp_ca_{hashlib.sha1(pem.encode()).hexdigest()[:12]}.pem"
    try:
        if not path.exists():
            path.write_text(pem, encoding="ascii")
    except OSError:
        return None
    os.environ["REQUESTS_CA_BUNDLE"] = str(path)
    os.environ["SSL_CERT_FILE"] = str(path)
    return str(path)
