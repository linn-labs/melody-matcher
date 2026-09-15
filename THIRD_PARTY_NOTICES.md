# Third-party notices

Reviewed September 13, 2026. The repository contains owned project source and documentation, not a vendored dependency distribution or an installed-environment bill of materials. Third-party packages, fonts, model code, weights, audio, and datasets are not bundled. Their licenses are not replaced by [LICENSE](LICENSE).

## Research and external model dependency

MERT is work by Yizhi Li and collaborators: [MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training](https://arxiv.org/abs/2306.00107). The selected [MERT-v1-330M model card](https://huggingface.co/m-a-p/MERT-v1-330M/blob/main/README.md) specifies CC BY-NC 4.0. Its weights and remote implementation are external requirements of the historical extractor, not included release assets.

The separate [MERT training-code license](https://github.com/yizhilll/MERT/blob/main/LICENSE) is Apache-2.0. The hosted remote model source acknowledges adaptation from Transformers/HuBERT and fairseq/wav2vec2. See the [rights review](docs/DATA_AND_RIGHTS.md) for those upstream licenses and the unresolved exact remote-code/derived-artifact boundary. The local extractor is a wrapper around those external interfaces, not a vendored MERT implementation.

## Browser and Python references

The [HTML shell](harness/frontend/index.html) references React/ReactDOM 18.3.1 and Babel standalone 7.29.0 from a CDN. Their upstream licenses are [React MIT](https://github.com/facebook/react/blob/v18.3.1/LICENSE) and [Babel MIT](https://github.com/babel/babel/blob/main/LICENSE). The page also requests Inter, Instrument Serif, and JetBrains Mono through Google Fonts; no font files are supplied. Inspect the exact resolved distributions and notices before redistributing them.

[requirements.txt](requirements.txt) records external Python package declarations. Additional imports and unresolved dependencies are described in [Components](docs/COMPONENTS.md). No dependency tree was installed or certified. Users who assemble an environment must retain the licenses/notices applicable to that actual distribution.

## Services and assets

Last.fm and Deezer are referenced by historical service clients. Their data/audio are not included and are subject to separate [terms and rights](docs/DATA_AND_RIGHTS.md). Apple Music is referenced by an XML importer and outbound search links; the source-only snapshot includes no Apple logo. Brand names identify interfaces or dependencies and do not indicate affiliation or endorsement.

The retained harness uses simple inline geometric controls and programmatic artwork placeholders. An unattributed brand glyph from the development harness was removed during curation. No third-party image bundle, album art, design archive, or vendor source tree is distributed.
