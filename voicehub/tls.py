"""Self-signed certificate generation with openssl."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger("voicehub.tls")


def ensure_cert(cert: Path, key: Path, sans: list[str]) -> None:
    if cert.exists() and key.exists():
        return
    cert.parent.mkdir(parents=True, exist_ok=True)
    alt = ",".join(("IP:" + s) if s.replace(".", "").isdigit() else ("DNS:" + s) for s in sans)
    cmd = ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650",
           "-keyout", str(key), "-out", str(cert), "-subj", "/CN=etl-voicehub",
           "-addext", f"subjectAltName={alt}"]
    subprocess.run(cmd, check=True, capture_output=True)
    key.chmod(0o600)
    log.info("generated self-signed certificate %s (SANs: %s)", cert, alt)
