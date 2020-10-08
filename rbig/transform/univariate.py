from collections import namedtuple


UniParams = namedtuple(
    "UniParams", ["support", "quantiles", "pdf_support", "pdf_empirical", "bounds"]
)
