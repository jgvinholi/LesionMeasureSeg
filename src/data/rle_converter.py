import numpy as np
from typing import List

class InputStream:
    def __init__(self, data: str):
        self.data = data
        self.i = 0

    def read(self, size: int) -> int:
        out = self.data[self.i : self.i + size]
        self.i += size
        return int(out, 2)

def access_bit(data: List[int], num: int) -> int:
    base = num // 8
    shift = 7 - (num % 8)
    return (data[base] & (1 << shift)) >> shift

def bytes2bit(data: List[int]) -> str:
    return ''.join(str(access_bit(data, i)) for i in range(len(data) * 8))

def rle_to_mask(rle: List[int], height: int, width: int) -> np.ndarray:
    """
    Converts Label Studio RLE to a (H×W) uint8 mask.
    """
    bitstream = InputStream(bytes2bit(rle))
    num_pixels = bitstream.read(32)
    word_size  = bitstream.read(5) + 1
    rle_sizes  = [bitstream.read(4) + 1 for _ in range(4)]

    out = np.zeros(num_pixels, dtype=np.uint8)
    i = 0
    while i < num_pixels:
        x = bitstream.read(1)
        size_code = bitstream.read(2)
        j = i + 1 + bitstream.read(rle_sizes[size_code])
        if x:
            val = bitstream.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = bitstream.read(word_size)
                out[i] = val
                i += 1

    # reshape into RGBA and take the alpha channel
    return out.reshape((height, width, 4))[:, :, 3]
