"""Integer arithmetic coder (Witten, Neal & Cleary 1987 style).

Works on a 256-symbol alphabet.  The model hands us a cumulative frequency
table of exactly PROB_TOTAL, with every symbol given at least frequency 1 so
that no byte is ever uncodable.

Precision: 32-bit interval, 16-bit frequencies.  After renormalisation the
interval is always >= QUARTER = 2^30, so range // PROB_TOTAL >= 2^14 > 0 and
every symbol keeps a non-empty sub-interval.
"""
from __future__ import annotations

CODE_BITS = 32
TOP = (1 << CODE_BITS) - 1
HALF = 1 << (CODE_BITS - 1)
QUARTER = 1 << (CODE_BITS - 2)
THREE_Q = 3 * QUARTER

PROB_BITS = 16
PROB_TOTAL = 1 << PROB_BITS


class _BitWriter:
    __slots__ = ("buf", "cur", "n")

    def __init__(self):
        self.buf = bytearray()
        self.cur = 0
        self.n = 0

    def write(self, bit: int):
        self.cur = (self.cur << 1) | bit
        self.n += 1
        if self.n == 8:
            self.buf.append(self.cur)
            self.cur = 0
            self.n = 0

    def finish(self) -> bytes:
        if self.n:
            self.buf.append(self.cur << (8 - self.n))
            self.cur = 0
            self.n = 0
        return bytes(self.buf)


class _BitReader:
    __slots__ = ("data", "pos", "cur", "n")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.cur = 0
        self.n = 0

    def read(self) -> int:
        if self.n == 0:
            if self.pos < len(self.data):
                self.cur = self.data[self.pos]
                self.pos += 1
            else:
                self.cur = 0  # past the end: feed zeros, standard trick
            self.n = 8
        self.n -= 1
        return (self.cur >> self.n) & 1


class ArithmeticEncoder:
    def __init__(self):
        self.low = 0
        self.high = TOP
        self.pending = 0
        self.out = _BitWriter()

    def _emit(self, bit: int):
        self.out.write(bit)
        while self.pending:
            self.out.write(bit ^ 1)
            self.pending -= 1

    def encode(self, cum_lo: int, cum_hi: int):
        """Encode a symbol occupying [cum_lo, cum_hi) of PROB_TOTAL."""
        rng = self.high - self.low + 1
        self.high = self.low + ((rng * cum_hi) >> PROB_BITS) - 1
        self.low = self.low + ((rng * cum_lo) >> PROB_BITS)
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_Q:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        # Two bits are enough to disambiguate the final interval.
        self.pending += 1
        if self.low < QUARTER:
            self._emit(0)
        else:
            self._emit(1)
        return self.out.finish()


class ArithmeticDecoder:
    def __init__(self, data: bytes):
        self.inp = _BitReader(data)
        self.low = 0
        self.high = TOP
        self.value = 0
        for _ in range(CODE_BITS):
            self.value = (self.value << 1) | self.inp.read()

    def target(self) -> int:
        """Cumulative-frequency position of the next symbol (0 <= t < PROB_TOTAL)."""
        rng = self.high - self.low + 1
        return (((self.value - self.low + 1) << PROB_BITS) - 1) // rng

    def consume(self, cum_lo: int, cum_hi: int):
        """Advance past the symbol whose interval the model resolved from target()."""
        rng = self.high - self.low + 1
        self.high = self.low + ((rng * cum_hi) >> PROB_BITS) - 1
        self.low = self.low + ((rng * cum_lo) >> PROB_BITS)
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < THREE_Q:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self.inp.read()
