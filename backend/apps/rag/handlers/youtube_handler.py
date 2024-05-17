import logging
from langchain_community.document_loaders import YoutubeLoader
from config import YOUTUBE_LOADER_LANGUAGE, YOUTUBE_LOADER_TRANSLATION

log = logging.getLogger(__name__)


def handle_youtube(url):
    return YoutubeLoader.from_youtube_url(
        url,
        add_video_info=True,
        language=YOUTUBE_LOADER_LANGUAGE,
        translation=YOUTUBE_LOADER_TRANSLATION,
    )
