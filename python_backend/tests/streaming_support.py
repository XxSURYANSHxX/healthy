"""Synthetic test storage, without later search or publication services."""

import hashlib

from services import sanitized_artifacts


def wallet_key(label):
    return (
        "0x"
        + hashlib.sha256(f"bioblock-stream-test-wallet:{label}".encode()).hexdigest()
    )


def wallet_address(label):
    from eth_account import Account

    return Account.from_key(wallet_key(label)).address.lower()


def isolate_storage(tmp_path, monkeypatch):
    monkeypatch.setenv(
        sanitized_artifacts.OUTPUT_DIR_ENV_VAR, str(tmp_path / "artifacts")
    )


def signed_upload_headers_for(label, path, content, profile="strict"):
    from services.upload_auth import signed_upload_headers

    return signed_upload_headers(wallet_key(label), path, content, profile)


def signed(client, label, method, path):
    from services.upload_auth import signed_ingestion_headers

    return client.request(
        method, path, headers=signed_ingestion_headers(wallet_key(label), method, path)
    )
