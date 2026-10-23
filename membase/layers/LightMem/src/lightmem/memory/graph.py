"""The source checkout contains no graph memory implementation."""
class GraphMem:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Graph memory is unavailable in this extracted LightMem version; use graph_mem=false.")
