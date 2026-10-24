import pathlib
from streamlit.testing.v1 import AppTest
APP = str(pathlib.Path(__file__).resolve().parents[1] / "app.py")

def test_all_pages_render():
    at = AppTest.from_file(APP, default_timeout=180); at.run(); assert not at.exception
    nav = lambda: [r for r in at.radio if r.label == "Navigate"][0]
    for o in list(nav().options):
        nav().set_value(o).run(); assert not at.exception, (o, [e.value for e in at.exception])
