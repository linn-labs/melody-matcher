# Data and rights

Documentation updated **September 14, 2026**. The external terms and license review below was conducted on **September 13, 2026**; this editorial update does not claim those sources were checked again. This is a bounded review for a source-only research snapshot, not a determination that prior or future data processing is authorized.

The [MIT license](../LICENSE) applies to owned source and documentation from Linn Autoracing Excellence LLC. It does not grant rights to service data, audio, model weights, external code, trademarks, or derived embeddings/checkpoints. No commercial clearance of the complete research pipeline is claimed.

## Material excluded

No real user records, listening events, personal library XML, audio, preview URLs carrying tokens, embeddings, databases, model weights, checkpoints, logs, screenshots, or experiment-result exports are distributed. Operator records, private collaboration documents, vendor/cache trees, and duplicate design archives are also excluded. Personal-library-derived sampling defaults and an unattributed brand glyph were removed from retained source.

Usernames, timestamps, counts, playlists, and song combinations can identify or characterize people. Replacing a name or hashing a username does not by itself anonymize listening behavior. Research reuse needs a lawful basis, relevant permissions, data minimization, retention/deletion rules, and a decision about what can be shared. Use synthetic examples for contributions. The [component guide](COMPONENTS.md) describes schemas, not supplied datasets.

## MERT and external model code

The [MERT-v1-330M model card](https://huggingface.co/m-a-p/MERT-v1-330M/blob/main/README.md) declares **CC BY-NC 4.0** and describes 1024-dimensional representations at 24 kHz. [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) imposes attribution and noncommercial conditions; its permissions are separate from this repository's MIT grant.

The [MERT training-code repository license](https://github.com/yizhilll/MERT/blob/main/LICENSE) is Apache-2.0. This does not relicense the hosted model weights. The hosted [model implementation](https://huggingface.co/m-a-p/MERT-v1-330M/blob/main/modeling_MERT.py) explicitly describes adaptation from Hugging Face Transformers' HuBERT implementation and fairseq's wav2vec2 implementation. Those upstream repositories carry [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) and [MIT](https://github.com/facebookresearch/fairseq/blob/main/LICENSE) licenses respectively.

The retained loader selects mutable remote code without a revision pin. This review does not establish the full license/notice chain for any particular future resolved model-code bundle. No external model code or weights are vendored here. Anyone considering execution must inspect the exact revision and its notices separately. No inference, model download, or training occurred during this review.

Whether particular embeddings, fine-tuned weights, or downstream checkpoints can be used or redistributed depends on their inputs, applicable agreements, and rights. This snapshot supplies none and makes no determination that they are unrestricted or commercially reusable.

## Last.fm

The reviewed [Last.fm API terms](https://www.last.fm/api/tos) request contact before research/academic or commercial use. They impose noncommercial restrictions absent a separate agreement, attribution and privacy duties, prohibit sublicensing the data, and specify a 100 MB reasonable-usage cap unless written consent permits more. Clause 2.6 restricts collection to documented web services and prohibits gathering other data in other ways.

The retained listener-page scraper therefore must not be treated as a terms-approved collection method. Source publication establishes no permission to collect, train on, or redistribute Last.fm records. No project-specific permission or compliance finding is supplied with this release. Recheck the full terms and any applicable agreement before reuse.

## Deezer and music rights

The reviewed [Deezer developer terms](https://developers.deezer.com/termsofuse) limit service use to noncommercial purposes and content use to a private family scope, require necessary rights and permissions, prohibit unauthorized downloading/sharing and DRM circumvention, and reserve other purposes for prior approval. Rights in content remain with Deezer or its right holders.

Publicly accessible search/preview endpoints and a short preview duration do not establish permission for ML extraction, training, redistribution, or commercial use. This review found no project-specific grant covering those activities. The code's download paths are historical research mechanisms, not permission or a supported recipe. No service requests, audio downloads, or account operations were performed during this review.

Apple Music/iTunes names describe an input format and outbound search link. No Apple affiliation, endorsement, catalog access, playback entitlement, or trademark license is claimed. An exported library also contains personal metadata whose sharing requires separate consideration.

## Reuse boundary

The source is available for study and adaptation under its stated license. A new data collection or artifact distribution is a separate rights decision. Excluding data and weights keeps this snapshot from distributing them; it does not retrospectively establish that their creation was permitted. See [Third-party notices](../THIRD_PARTY_NOTICES.md) for external software references and [Provenance](../PROVENANCE.md) for curation boundaries.
