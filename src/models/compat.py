"""Make models pickled with one scikit-learn version usable under another."""
from sklearn.linear_model import LogisticRegression


def fix_sklearn_compat(obj, _seen=None):
    """Recursively add attributes older/newer scikit-learn versions expect on unpickled LogisticRegression objects."""
    _seen = _seen if _seen is not None else set()
    if id(obj) in _seen or obj is None or isinstance(obj, (str, bytes, int, float, bool)): return obj
    _seen.add(id(obj))
    if isinstance(obj, LogisticRegression):
        if not hasattr(obj, "multi_class"): obj.multi_class = "auto"
        if not hasattr(obj, "n_jobs"): obj.n_jobs = None
    if isinstance(obj, dict): items = list(obj.values())
    elif isinstance(obj, (list, tuple, set)): items = list(obj)
    elif hasattr(obj, "__dict__"): items = list(vars(obj).values())
    else: items = []
    for v in items: fix_sklearn_compat(v, _seen)
    return obj
