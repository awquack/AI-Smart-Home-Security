# generate_cert.py – Generate a self-signed SSL certificate for HTTPS
#
# Run once before starting app.py:
#   python generate_cert.py
#
# Requires: pip install cryptography

import os
import datetime
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CERT_FILE = os.path.join(BASE_DIR, "cert.pem")
KEY_FILE  = os.path.join(BASE_DIR, "key.pem")


def generate():
    # Generate private key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Self-signed certificate valid for 3 years
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME,         "AI Security Dashboard"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME,   "AI Smart Home Security"),
        x509.NameAttribute(NameOID.COUNTRY_NAME,        "US"),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=1095))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(__import__("ipaddress").IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    # Write key
    with open(KEY_FILE, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    # Write cert
    with open(CERT_FILE, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"[CERT] Certificate → {CERT_FILE}")
    print(f"[CERT] Private key → {KEY_FILE}")
    print("[CERT] Valid for 3 years.")
    print("[CERT] Note: browser will show a warning — click 'Advanced' → 'Proceed' to accept.")


if __name__ == "__main__":
    generate()
