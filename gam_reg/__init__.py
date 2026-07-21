"""GAM-Reg package."""

__all__ = ["GAMReg"]


def __getattr__(name: str):
    if name == "GAMReg":
        from gam_reg.models.gam_reg import GAMReg

        return GAMReg
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
