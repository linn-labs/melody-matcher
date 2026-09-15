# Experimental comparison harness

This is source for a historical local experiment, with an XML importer, fuzzy catalog matching, model registry, inference, run comparison, and feedback storage. It is unfinished and unsupported. No database, library, embedding catalog, or trained model is included, and no harness session was tested for this snapshot.

The backend modifies local storage and deserializes artifacts; untrusted pickle/checkpoint files can execute code. The service has no authentication and uses wildcard CORS. The frontend requests external scripts/fonts, sends uploads to its backend, and can trigger inference during comparison. It is not a read-only or offline viewer.

Read [Components](../docs/COMPONENTS.md) for schemas, cache limitations, dependencies, and historical UI labels; [Research](../docs/RESEARCH.md) for the actual meaning of hurdle outputs; and [Data and rights](../docs/DATA_AND_RIGHTS.md) before considering reuse. No deployment or supported setup instructions are provided.
