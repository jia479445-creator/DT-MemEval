# Third-party provenance

DT-MemEval combines the authors' dataset with a selected subset of the locally used MemBase framework. Renaming the distribution does not change the origin of the framework or baseline implementations.

- MemBase: extracted from the adjacent local `MemBase` checkout; Copyright (c) 2026 ZJUNLP, MIT. The complete notice is retained in `LICENSE-MemBase`.
- A-MEM: implementation bundled by MemBase, upstream https://github.com/WujiangXu/A-mem-sys.
- Mem0: implementation bundled by MemBase, upstream https://github.com/mem0ai/mem0. Original file headers are preserved.
- LightMem: bundled source from https://github.com/zjunlp/LightMem, Apache-2.0, with its original `LICENSE` and package metadata under `membase/layers/LightMem/`.
- LangMem and NaiveRAG use external packages installed from the supplied dependency specifications; their licenses remain applicable.

No additional license is asserted for the authors' dataset or modifications by this extraction; the MIT notice applies to the extracted MemBase material, not automatically to every dataset or third-party component.
