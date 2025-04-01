"""Utility functions for the quantum threat tracker algorithms."""
import math
from functools import cached_property
from typing import Optional

from attrs import field, frozen
from qualtran import Bloq, QInt, QMontgomeryUInt, QUInt, Register, Signature
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.basic_gates import CNOT, Toffoli
from qualtran.resource_counting import (
    BloqCountDictT,
    SympySymbolAllocator,
)
from qualtran.resource_counting.generalizers import _ignore_wrapper  # noqa: PLC2701


@frozen
class AddCustom(Bloq):
    """A custom implementation of the `Add` bloq that overrides the call graph."""

    a_dtype: QInt | QUInt | QMontgomeryUInt = field()
    b_dtype: QInt | QUInt | QMontgomeryUInt = field()

    @cached_property
    def signature(self) -> "Signature":
        """Bloq signature."""
        return Signature([Register("a", self.a_dtype), Register("b", self.b_dtype)])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        """Call graph construction.

        Returns
        -------
        BloqCountDictT
            Custom Bloq counts.
        """
        n = self.b_dtype.bitsize
        n_cnot = (n - 2) * 6 + 3
        return {Toffoli(): 2 * (n - 1), CNOT(): n_cnot}


def generalize_and_decomp(bloq: Bloq) -> Optional[Bloq]:
    """Override the default And Bloq of qualtran.

    Returns
    -------
    Optional[Bloq]
        A custom And Bloq if the input is an instance of And, otherwise the result
        of _ignore_wrapper.
    """
    if isinstance(bloq, Add):
        return AddCustom(a_dtype=bloq.a_dtype, b_dtype=bloq.b_dtype)

    return _ignore_wrapper(generalize_and_decomp, bloq)


def fips_strength_level(n: int) -> float:
    """Calculate the security strength level based on FIPS 140-2 guidelines.

    This function calculates the security strength in bits based on the asymptotic
    complexity of the sieving step in the general number field sieve (GNFS).

    The formula is derived from FIPS 140-2 IG CMVP, page 110 and implemented based on
    https://github.com/Strilanc/efficient-quantum-factoring-2019/

    Parameters
    ----------
    n : int
        The bit length of the cryptographic key.

    Returns
    -------
    float
        The equivalent security strength in bits.
    """
    ln = math.log
    return (1.923 * (n * ln(2))**(1 / 3) * ln(n * ln(2))**(2 / 3) - 4.69) / ln(2)


def fips_strength_level_rounded(n: int) -> int:
    """Round the security strength level to the nearest multiple of 8.

    Parameters
    ----------
    n : int
        The bit length of the cryptographic key.

    Returns
    -------
    int
        The security strength level rounded to the nearest multiple of 8.
    """
    return 8 * round(fips_strength_level(n) / 8)
