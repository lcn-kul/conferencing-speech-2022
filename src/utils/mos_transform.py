from dataclasses import dataclass


@dataclass
class MosTransform:
    in_mos_min: float
    in_mos_max: float
    out_mos_min: float
    out_mos_max: float

    @property
    def in_range(self) -> float:
        return self.in_mos_max - self.in_mos_min

    @property
    def out_range(self) -> float:
        return self.out_mos_max - self.out_mos_min

    def transform(self, mos: float) -> float:
        frac = (mos - self.in_mos_min) / self.in_range
        return self.out_mos_min + frac * self.out_range

    def transform_str(self, mos: str, fmt="%0.6f") -> str:
        out = self.transform(float(mos))
        return fmt % out
