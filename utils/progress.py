try:
    import progressbar as _progressbar
except ImportError:
    _progressbar = None


def progressbar(iterable):
    if _progressbar is None:
        return iterable

    progress = getattr(_progressbar, "progressbar", None)
    if callable(progress):
        return progress(iterable)

    progress_bar = getattr(_progressbar, "ProgressBar", None)
    if progress_bar is not None:
        return progress_bar()(iterable)

    return iterable
