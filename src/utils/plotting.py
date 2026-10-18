import sys


def configure_matplotlib_backend():
    """
    Chọn backend 'Agg' (không cần màn hình) khi chạy dạng script.

    Trong Jupyter/Colab (`IPython` đã được import) thì GIỮ NGUYÊN backend mặc định
    để biểu đồ hiển thị inline.

    Phải gọi TRƯỚC khi `import matplotlib.pyplot`.
    """
    if 'IPython' in sys.modules:
        return
    import matplotlib
    matplotlib.use('Agg')
