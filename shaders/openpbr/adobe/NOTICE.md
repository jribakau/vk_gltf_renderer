# Adobe's OpenPBR BSDF

Vendored from <https://github.com/adobe/openpbr-bsdf> at commit `c91aad1` (2026-08-24), unchanged, under
the Apache License 2.0 (`LICENSE`). Its own guide is upstream's `README.md`.

Only the sources are kept: `openpbr*.h`, `impl/` and `interop/`. Configure it from outside
(`../fv_openpbr.h.slang`), through the hooks `openpbr_settings.h` lists, rather than editing these files, so
a new upstream version replaces them whole.
