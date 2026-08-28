import os
from pathlib import Path

from pypdf import PdfReader
from docx import Document


# =========================================================
# FILE READER
# =========================================================

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
}


# =========================================================
# READ TEXT FILE
# =========================================================

def read_txt(file_path: str) -> str:
    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:
        return file.read()


# =========================================================
# READ PDF FILE
# =========================================================

def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


# =========================================================
# READ DOCX FILE
# =========================================================

def read_docx(file_path: str) -> str:
    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


# =========================================================
# READ ANY SUPPORTED FILE
# =========================================================

def read_file(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()

    if extension == ".txt":
        return read_txt(file_path)

    if extension == ".pdf":
        return read_pdf(file_path)

    if extension == ".docx":
        return read_docx(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


# =========================================================
# FILE INFORMATION
# =========================================================

def get_file_info(file_path: str) -> dict:
    path = Path(file_path)

    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "supported": path.suffix.lower()
        in SUPPORTED_EXTENSIONS,
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FILE READER TEST")
    print("=" * 60)

    print()

    print(
        "Supported file types:"
    )

    for extension in SUPPORTED_EXTENSIONS:
        print(
            f"  {extension}"
        )

    print()

    print(
        "File Reader module loaded successfully."
    )

    print("=" * 60)