PAPERS = turbopress_preprint turbopress_openreview

.PHONY: all figures clean
all: figures $(PAPERS:=.pdf)

figures:
	python make_figures.py

%.pdf: %.tex references.bib
	pdflatex -interaction=nonstopmode $*
	bibtex $* || true
	pdflatex -interaction=nonstopmode $*
	pdflatex -interaction=nonstopmode $*

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPERS:=.pdf)
