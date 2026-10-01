"""Download dos arquivos de SRAG publicados no Open DATASUS.

Uso:
    python -m srag_agent.data.download
"""

import logging
from pathlib import Path

import httpx

from srag_agent.config import OPENDATASUS_BASE_URL, get_settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024


def download_file(url: str, destination: Path, *, overwrite: bool = False) -> Path:
    """Baixa `url` para `destination` em streaming, gravando primeiro num arquivo temporário."""
    if destination.exists() and not overwrite:
        logger.info("Já existe, pulando: %s", destination.name)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with httpx.stream("GET", url, timeout=300, follow_redirects=True) as response:
        response.raise_for_status()
        with partial.open("wb") as fh:
            for chunk in response.iter_bytes(CHUNK_SIZE):
                fh.write(chunk)
    partial.replace(destination)
    logger.info("Baixado: %s (%.1f MB)", destination.name, destination.stat().st_size / 1e6)
    return destination


def download_all(*, overwrite: bool = False) -> list[Path]:
    settings = get_settings()
    remote_paths = [*settings.source_files, settings.data_dictionary_file]
    return [
        download_file(
            f"{OPENDATASUS_BASE_URL}/{remote}",
            settings.raw_dir / Path(remote).name,
            overwrite=overwrite,
        )
        for remote in remote_paths
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_all()


if __name__ == "__main__":
    main()
