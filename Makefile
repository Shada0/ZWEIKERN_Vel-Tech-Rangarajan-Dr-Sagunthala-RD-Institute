.PHONY: setup data train eval test app report real
setup: ; pip install -r requirements.txt
data: ; python -m src.cli generate
train: ; python -m src.cli train
eval: ; python -m src.cli evaluate
test: ; python -m pytest -q
app: ; streamlit run app.py
report: ; python -m src.make_report && python -m src.make_architecture_diagram && python -m src.make_presentation
real: ; python -m src.cli real $(RAW)
