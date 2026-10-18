import sys


def setup_console():
    """
    Ép stdout/stderr sang UTF-8 trên Windows để in được tiếng Việt.

    Mặc định console Windows dùng cp1252/cp437 nên `print("Hoa Hồng")` sẽ ném
    UnicodeEncodeError. Gọi hàm này ở đầu mỗi entry point (script/CLI/app).
    """
    if sys.platform == 'win32':
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding='utf-8')
            except Exception:
                pass
