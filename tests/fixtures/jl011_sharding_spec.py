# jaxlint-expect: JL011
"""JL011: legacy ShardingSpec attribute reference (no jax import needed)."""


class _Fake:
    ShardingSpec = None


_t = _Fake.ShardingSpec
