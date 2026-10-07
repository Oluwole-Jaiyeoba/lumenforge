from types import SimpleNamespace

from agentic_backends.sglang.trace.patch import _storage_prefetch_plan


def _fixture(*, matched=0, free=512, prefix=256, rate_limited=False):
    root = SimpleNamespace(key=[], parent=None)
    node = SimpleNamespace(key=list(range(matched)), parent=root) if matched else root
    tree = SimpleNamespace(root_node=root, page_size=64, prefetch_threshold=64)
    pool = SimpleNamespace(available_size=lambda: free)
    controller = SimpleNamespace(mem_pool_host=pool, prefetch_rate_limited=lambda: rate_limited)
    match = SimpleNamespace(last_host_node=node)
    tokens = list(range(prefix))
    return tree, controller, match, tokens


def test_full_storage_prefix_prefetch_uses_root():
    tree, controller, match, tokens = _fixture()
    node, suffix, status = _storage_prefetch_plan(tree, controller, match, tokens, 0)
    assert node is tree.root_node
    assert suffix == tokens
    assert status["host_required_tokens"] == 256


def test_partial_storage_prefix_uses_matched_host_node_and_suffix():
    tree, controller, match, tokens = _fixture(matched=128)
    node, suffix, status = _storage_prefetch_plan(tree, controller, match, tokens, 128)
    assert node is match.last_host_node
    assert suffix == tokens[128:]
    assert status["host_required_tokens"] == 128


def test_prefetch_refuses_native_host_eviction_when_capacity_is_short():
    tree, controller, match, tokens = _fixture(matched=128, free=64)
    node, suffix, status = _storage_prefetch_plan(tree, controller, match, tokens, 128)
    assert node is None and suffix == []
    assert status["status"] == "host_capacity_insufficient"


def test_prefetch_refuses_mismatched_tree_anchor():
    tree, controller, match, tokens = _fixture(matched=64)
    node, suffix, status = _storage_prefetch_plan(tree, controller, match, tokens, 128)
    assert node is None and suffix == []
    assert status["status"] == "storage_prefix_anchor_mismatch"


def test_prefetch_rate_limit_is_a_named_skip():
    tree, controller, match, tokens = _fixture(rate_limited=True)
    node, suffix, status = _storage_prefetch_plan(tree, controller, match, tokens, 0)
    assert node is None and suffix == []
    assert status["status"] == "storage_prefetch_rate_limited"
