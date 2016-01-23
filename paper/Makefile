LATEX       = pdflatex
CHECK_RERUN = grep Rerun $*.log

all: abc.pdf

%.pdf: %.tex
	${LATEX} $<
	${LATEX} $<
	${LATEX} $<
