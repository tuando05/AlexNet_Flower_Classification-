import logging
import sys

LOG_FORMAT = '%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s'
DATE_FORMAT = '%H:%M:%S'
DEFAULT_LEVEL = logging.INFO


def setup_logging(level=DEFAULT_LEVEL):
    """
    Cấu hình logging cho entry point (script / CLI / web app).

    Gọi một lần ở đầu mỗi entry point. Các module thư viện chỉ cần
    `logger = logging.getLogger(__name__)` và KHÔNG tự gắn handler.

    Dùng `force=True` để ghi đè cấu hình cũ (ví dụ khi chạy lại trong notebook).
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )
