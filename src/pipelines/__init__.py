"""
Package chứa các pipeline ML (train / evaluate / inference).

Chủ đích để trống: các entry point được chạy bằng
`python -m src.pipelines.<module>` nên chỉ cần import trực tiếp module con.
Việc KHÔNG import sẵn trong `__init__.py` cũng tránh RuntimeWarning của runpy
("found in sys.modules after import of package ... but prior to execution").
"""
