import logging
import mimetypes
import json
from fastapi import HTTPException, status
from pathlib import Path
from config import DOCS_DIR, ERROR_MESSAGES
from ingestion import store_data_in_vector_db, get_loader
from utils.misc import (
    extract_folders_after_data_docs,
    sanitize_filename,
    calculate_sha256_string,
)
from apps.web.models.documents import Documents, DocumentForm

log = logging.getLogger(__name__)


def scan_docs_dir_handler(user, app_state):
    try:
        for path in Path(DOCS_DIR).rglob("./**/*"):
            if path.is_file() and not path.name.startswith("."):
                tags = extract_folders_after_data_docs(path)
                filename = path.name
                file_content_type = mimetypes.guess_type(path)

                f = open(path, "rb")
                collection_name = calculate_sha256_string(f)[:63]
                f.close()

                loader, known_type = get_loader(
                    filename, file_content_type[0], str(path)
                )
                data = loader.load()

                result = store_data_in_vector_db(data, collection_name)

                if result:
                    sanitized_filename = sanitize_filename(filename)
                    doc = Documents.get_doc_by_name(sanitized_filename)

                    if doc is None:
                        Documents.insert_new_doc(
                            user.id,
                            DocumentForm(
                                name=sanitized_filename,
                                title=filename,
                                collection_name=collection_name,
                                filename=filename,
                                content=(
                                    json.dumps(
                                        {"tags": [{"name": name} for name in tags]}
                                    )
                                    if tags
                                    else "{}"
                                ),
                            ),
                        )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )
