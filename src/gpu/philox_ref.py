import math

import numpy as np

M0, M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
W0, W1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)
MASK32 = np.uint64(0xFFFFFFFF)
TWO53 = 9007199254740992.0
TWO26 = 67108864.0


def philox4x32_10(c0, c1, c2, c3, k0, k1):
    c0, c1, c2, c3 = (np.asarray(c, dtype=np.uint32) for c in (c0, c1, c2, c3))
    k0, k1 = np.uint32(k0), np.uint32(k1)
    with np.errstate(over="ignore"):
        for _ in range(10):
            p0 = M0 * c0.astype(np.uint64)
            p1 = M1 * c2.astype(np.uint64)
            hi0, lo0 = (p0 >> np.uint64(32)).astype(np.uint32), (p0 & MASK32).astype(np.uint32)
            hi1, lo1 = (p1 >> np.uint64(32)).astype(np.uint32), (p1 & MASK32).astype(np.uint32)
            c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
            k0, k1 = k0 + W0, k1 + W1
    return c0, c1, c2, c3


def seed_to_key(seed):
    return seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF


def normals_for_pairs(pairs, seed):
    pairs = np.asarray(pairs, dtype=np.uint64)
    k0, k1 = seed_to_key(seed)
    zero = np.zeros(pairs.shape, dtype=np.uint32)
    r0, r1, r2, r3 = philox4x32_10((pairs & MASK32).astype(np.uint32),
                                   (pairs >> np.uint64(32)).astype(np.uint32),
                                   zero, zero, k0, k1)
    u1 = ((r0 >> 5).astype(np.float64) * TWO26 + (r1 >> 6).astype(np.float64) + 1.0) / TWO53  # (0, 1]
    u2 = ((r2 >> 5).astype(np.float64) * TWO26 + (r3 >> 6).astype(np.float64)) / TWO53        # [0, 1)
    r = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2
    return r * np.cos(angle), r * np.sin(angle)


def price(n, seed=30, s0=100.0, k=100.0, rate=0.05, sig=0.20, t=1.0, chunk=1_000_000):
    drift, vol, disc = (rate - 0.5 * sig**2) * t, sig * math.sqrt(t), math.exp(-rate * t)
    sums = np.zeros(4)
    n_pairs = (n + 1) // 2
    for start in range(0, n_pairs, chunk):
        p = np.arange(start, min(start + chunk, n_pairs), dtype=np.uint64)
        z0, z1 = normals_for_pairs(p, seed)
        z = np.stack([z0, z1], axis=1).ravel()[: min(2 * len(p), n - 2 * start)]
        st = s0 * np.exp(drift + vol * z)
        call, put = disc * np.maximum(st - k, 0.0), disc * np.maximum(k - st, 0.0)
        sums += [call.sum(), (call * call).sum(), put.sum(), (put * put).sum()]
    return tuple(sums)


if __name__ == "__main__":
    # Known-answer tests from the Random123 reference distribution (kat_vectors).
    kats = [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
        ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
         (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
    ]
    for ctr, key, expected in kats:
        got = tuple(int(x[0]) for x in philox4x32_10(*[[c] for c in ctr], *key))
        status = "OK" if got == expected else f"FAIL (got {[hex(g) for g in got]})"
        print(f"Philox KAT {[hex(c) for c in ctr]}: {status}")