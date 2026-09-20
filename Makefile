PY ?= python3
export PYTHONPATH := src

.PHONY: test init probe ingest run site verify report clean
test:
	$(PY) -m unittest discover -s tests -v

init:
	$(PY) -m nhlcomp init

probe:
	$(PY) -m nhlcomp probe

ingest:
	$(PY) -m nhlcomp ingest --settled-pages 8

run:
	$(PY) -m nhlcomp run --settled-pages 8 --cross-check-clubs TOR,BOS

site:
	$(PY) -m nhlcomp build-site

verify:
	$(PY) -m nhlcomp verify

report:
	$(PY) -m nhlcomp report

clean:
	rm -rf data/*.db data/*.db-* data/cache docs
