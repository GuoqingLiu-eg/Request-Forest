from __future__ import annotations

from array import array
from typing import List


def flatten_ragged(per_pos_val, per_pos_idx):


    flat_val: List[float] = []
    flat_idx: List[int] = []
    lens: List[int] = []
    if not per_pos_val:
        assert not per_pos_idx, (
            f"ragged idx column has {len(per_pos_idx)} positions but the val "
            "column is empty"
        )
        return flat_val, flat_idx, lens
    assert per_pos_idx is not None and len(per_pos_idx) == len(per_pos_val), (
        f"ragged idx column has {len(per_pos_idx) if per_pos_idx else 0} "
        f"positions, val column has {len(per_pos_val)}"
    )
    for p, pv in enumerate(per_pos_val):
        pi = per_pos_idx[p]
        if pv:


            assert pi is not None and len(pi) == len(pv), (
                f"position {p}: idx len "
                f"{len(pi) if pi is not None else None} != val len {len(pv)}"
            )
            flat_val.extend(pv)
            flat_idx.extend(pi)
            lens.append(len(pv))
        else:
            assert not pi, f"position {p}: idx has {len(pi)} entries but val is empty"
            lens.append(0)
    return flat_val, flat_idx, lens


def flatten_hidden(hs):

    vals: List[float] = []
    lens: List[int] = []
    if not hs:
        return vals, lens
    for row in hs:
        flat = _flatten_floats(row)
        vals.extend(flat)
        lens.append(len(flat))
    return vals, lens


def _flatten_floats(x):

    if isinstance(x, (int, float)):
        return [float(x)]
    out: List[float] = []
    for e in x:
        out.extend(_flatten_floats(e))
    return out


class FlatPairColumns:


    def __init__(self, name, vals, idxs, first_none_to_nan=False):
        self.name = name
        self.vals = vals
        self.idxs = idxs
        self.first_none_to_nan = first_none_to_nan
        self.v = array("f")
        self.i = array("i")
        self.lens = []

    def columns(self):
        return ((f"{self.name}_val", self.vals), (f"{self.name}_idx", self.idxs))

    def accept(self, j):
        vv = (self.vals[j] if self.vals else None) or []
        ii = (self.idxs[j] if self.idxs else None) or []


        assert len(ii) == len(
            vv
        ), f"{self.name}: request {j} has {len(ii)} idx entries but {len(vv)} vals"
        if self.first_none_to_nan and vv and vv[0] is None:
            self.v.append(float("nan"))
            self.v.extend(vv[1:])
        else:
            self.v.extend(vv)
        self.i.extend(ii)
        self.lens.append(len(vv))

    def header_cols(self):
        return [self.lens]

    def data_cols(self):
        return [self.v.tobytes(), self.i.tobytes()]


class RaggedPairColumns:


    def __init__(self, name, vals, idxs):
        self.name = name
        self.vals = vals
        self.idxs = idxs
        self.v = array("f")
        self.i = array("i")
        self.pos = []
        self.req = []

    def columns(self):
        return ((f"{self.name}_val", self.vals), (f"{self.name}_idx", self.idxs))

    def accept(self, j):
        fv, fi, lens = flatten_ragged(
            self.vals[j] if self.vals else None,
            self.idxs[j] if self.idxs else None,
        )
        self.v.extend(fv)
        self.i.extend(fi)
        self.pos.extend(lens)
        self.req.append(len(lens))

    def header_cols(self):
        return [self.req, self.pos]

    def data_cols(self):
        return [self.v.tobytes(), self.i.tobytes()]


class NestedRowColumns:


    def __init__(self, name, rows):
        self.name = name
        self.rows = rows
        self.v = array("f")
        self.pos = []
        self.req = []

    def columns(self):
        return ((self.name, self.rows),)

    def accept(self, j):
        hv, hlens = flatten_hidden(self.rows[j] if self.rows else None)
        self.v.extend(hv)
        self.pos.extend(hlens)
        self.req.append(len(hlens))

    def header_cols(self):
        return [self.req, self.pos]

    def data_cols(self):
        return [self.v.tobytes()]
