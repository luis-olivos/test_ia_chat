"""Utility script to generate the persistent Chroma index offline."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv

from main import CHROMA_DIR, PDF_FOLDER, build_vector_store, load_all_documents


def generate_index(pdf_folder: str, chroma_dir: str, recreate: bool = True) -> None:
    """Generate embeddings for all PDFs and persist them into ``chroma_dir``."""

    documents = load_all_documents(pdf_folder)
    if not documents:
        raise RuntimeError(
            f"No PDF documents found in '{pdf_folder}'. Add PDFs before generating the index."
        )

    target_dir = Path(chroma_dir)
    if recreate and target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    build_vector_store(documents, persist_directory=str(target_dir))


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate the persistent Chroma index for the API.")
    parser.add_argument(
        "--pdf-folder",
        default=PDF_FOLDER,
        help="Folder containing the PDF files to index (defaults to the PDF_FOLDER env var).",
    )
    parser.add_argument(
        "--chroma-dir",
        default=CHROMA_DIR,
        help="Directory where the Chroma index will be stored (defaults to the CHROMA_DIR env var).",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help=(
            "Keep an existing Chroma directory instead of recreating it. Useful to append new documents "
            "without rebuilding from scratch."
        ),
    )

    args = parser.parse_args()

    generate_index(args.pdf_folder, args.chroma_dir, recreate=not args.keep_existing)


if __name__ == "__main__":
    main()
