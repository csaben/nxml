from __future__ import annotations

from typing import BinaryIO

from nxml_control.catalog import Catalog, Upload
from nxml_control.storage import ObjectStorage


class IngestService:
    def __init__(self, catalog: Catalog, storage: ObjectStorage) -> None:
        self.catalog, self.storage = catalog, storage

    def upload(self, upload_id: str, stream: BinaryIO) -> Upload:
        upload = self.catalog.get(upload_id)
        info = self.storage.put_if_absent(upload.object_key, stream)
        if (info.size_bytes, info.sha256) != (upload.expected_size_bytes, upload.expected_sha256):
            raise ValueError("object checksum or size does not match declaration")
        return self.catalog.mark_uploaded(upload_id, size_bytes=info.size_bytes, sha256=info.sha256)

    def inspect(self, upload_id: str) -> Upload:
        upload = self.catalog.get(upload_id)
        info = self.storage.inspect(upload.object_key)
        if info is None:
            return upload
        if (info.size_bytes, info.sha256) != (upload.expected_size_bytes, upload.expected_sha256):
            raise ValueError("stored object checksum or size does not match declaration")
        return self.catalog.mark_uploaded(upload_id, size_bytes=info.size_bytes, sha256=info.sha256)

    def verify_members(self, upload_id: str, manifest: dict) -> None:
        """Verify every immutable tar member against the strict commit manifest."""
        import hashlib
        import tarfile

        upload = self.catalog.get(upload_id)
        if upload.state not in {"uploaded", "committed"}:
            raise ValueError("upload must be verified before member inspection")
        declared = {item["path"]: item for item in manifest.get("members", [])}
        with (
            self.storage.open(upload.object_key) as source,
            tarfile.open(fileobj=source, mode="r:*") as archive,
        ):
            actual = {item.name: item for item in archive.getmembers() if item.isfile()}
            if set(actual) != set(declared):
                raise ValueError("tar members do not exactly match manifest members")
            for path in sorted(declared):
                member = actual[path]
                expected = declared[path]
                if member.size != expected["size_bytes"]:
                    raise ValueError(f"member size mismatch: {path}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"could not read member: {path}")
                digest = hashlib.sha256()
                while chunk := extracted.read(1024 * 1024):
                    digest.update(chunk)
                if digest.hexdigest() != expected["sha256"]:
                    raise ValueError(f"member checksum mismatch: {path}")
