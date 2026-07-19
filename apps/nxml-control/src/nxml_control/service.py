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
