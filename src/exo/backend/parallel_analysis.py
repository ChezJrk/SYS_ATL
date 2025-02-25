from ..core.LoopIR import LoopIR, LoopIR_Rewrite

from ..rewrite.new_eff import Check_ParallelizeLoop


class ParallelAnalysis(LoopIR_Rewrite):
    def __init__(self):
        self._errors = []

    def run(self, proc):
        assert isinstance(proc, LoopIR.proc)
        self.proc = proc
        proc = super().apply_proc(proc)
        if self._errors:
            errs = "\n".join(self._errors)
            raise TypeError(f"Errors occurred during parallel analysis:\n{errs}")
        return proc

    def err(self, node, msg):
        self._errors.append(f"{node.srcinfo}: {msg}")

    def map_s(self, s):
        if isinstance(s, LoopIR.For) and isinstance(s.loop_mode, LoopIR.Par):
            try:
                Check_ParallelizeLoop(self.proc, s)
            except:
                self.err(
                    s,
                    "parallel loop's body is not parallelizable because of potential data races",
                )
