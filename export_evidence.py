"""Package source and evidence without embedding model binaries in the text export."""
import base64
import hashlib
from pathlib import Path
import tarfile
import tempfile

from instruments import ROOT

with tempfile.NamedTemporaryFile(suffix=".tar.gz") as bundle:
    with tarfile.open(bundle.name, "w:gz") as archive:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT)
            if any(part.startswith("soup_tokens_") for part in relative.parts):
                continue
            if path.suffix in (".safetensors", ".bin", ".arrow", ".pt") or path.stat().st_size > 10_000_000:
                continue
            archive.add(path, arcname=relative, recursive=False)
    raw = Path(bundle.name).read_bytes()
packed = base64.b64encode(raw).decode("ascii")
output = ROOT.parent
stride = 65_000
count = (len(packed) + stride - 1) // stride
for index in range(count):
    (output / f"evidence_{index+1:03d}.log").write_text(
        f"TAKEHOME-EVIDENCE {index+1}/{count} {hashlib.sha256(raw).hexdigest()}\n"
        + packed[index*stride:(index+1)*stride] + "\n", encoding="ascii",
    )
print(f"Packed {len(raw)} bytes in {count} parts; SHA-256 {hashlib.sha256(raw).hexdigest()}")
