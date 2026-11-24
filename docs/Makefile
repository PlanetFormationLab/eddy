# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build and serve the docs locally at http://127.0.0.1:8800 — the quickest
# way to eyeball the PFL theme without pushing to Read the Docs.
preview: html
	@echo "eddy docs -> http://127.0.0.1:8800  (ctrl-c to stop)"
	@cd "$(BUILDDIR)/html" && python3 -m http.server 8800 --bind 127.0.0.1

# Same, but rebuilds and refreshes the browser on every save. Needs
# `pip install sphinx-autobuild`.
livehtml:
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" --port 8800 $(SPHINXOPTS) $(O)

.PHONY: help Makefile preview livehtml

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)