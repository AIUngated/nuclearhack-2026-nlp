PYTHON ?= python

.PHONY: test eda train predict reproduce collect review report app
test:
	$(PYTHON) -m pytest -q
eda:
	$(PYTHON) scripts/run.py eda
train:
	$(PYTHON) scripts/run.py train
predict:
	$(PYTHON) scripts/run.py predict
reproduce:
	$(PYTHON) scripts/run.py reproduce
collect:
	$(PYTHON) scripts/run.py collect
review:
	$(PYTHON) scripts/review_pairs.py
report:
	$(PYTHON) scripts/summarize.py
app:
	$(PYTHON) -m streamlit run app/streamlit_app.py
