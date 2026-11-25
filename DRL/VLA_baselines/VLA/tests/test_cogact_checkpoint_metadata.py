import io
import pickle
import unittest
from unittest.mock import patch
import zipfile

import torch

from VLA.cogact.audit_checkpoint_metadata import HTTPRangeReader, MetadataUnpickler, summarize


class RangeAndPickleTest(unittest.TestCase):
    def test_tensor_metadata_only_no_tensor_payload_fetch(self):
        payload = io.BytesIO()
        torch.save({"model": {"llm_backbone": {"weight": torch.zeros(1_000_000)}}}, payload)
        data = payload.getvalue()

        class Response:
            status_code = 206
            def __init__(self, start, end):
                self.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
                self.raw = io.BytesIO(data[start:end + 1])
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def raise_for_status(self):
                pass

        def get(url, headers, **kwargs):
            start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            return Response(start, end)

        remote = HTTPRangeReader("https://test.invalid/weights.pt", len(data))
        with patch("VLA.cogact.audit_checkpoint_metadata.requests.get", side_effect=get):
            with zipfile.ZipFile(remote) as archive:
                metadata = archive.read("archive/data.pkl")
        result = MetadataUnpickler(io.BytesIO(metadata)).load()
        summary = summarize(result["model"])
        self.assertEqual(summary["logical_elements_by_storage_dtype"], {"FloatStorage": 1_000_000})
        self.assertLess(remote.bytes_read, 10_000)
        with self.assertRaises(ValueError):
            remote.seek(0)
            remote.read()

    def test_non_metadata_pickle_global_is_rejected(self):
        with self.assertRaises(pickle.UnpicklingError):
            MetadataUnpickler(io.BytesIO(pickle.dumps(eval))).load()


if __name__ == "__main__":
    unittest.main()
