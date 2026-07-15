from nanovllm.engine.block_manager import BlockManager


def make_block_manager(num_blocks: int = 8) -> BlockManager:
    return BlockManager(
        num_blocks=num_blocks,
        block_size=256,
        kv_capacity_threshold=1.0,
        chunk_size=512,
    )


def test_allocate_grows_block_table_one_chunk_at_a_time(make_sequence):
    manager = make_block_manager()
    seq = make_sequence(800)

    manager.allocate(seq)

    assert len(seq.block_table) == 2
    assert len(manager.used_block_ids) == 2

    seq.num_chunked_tokens = 512
    manager.allocate(seq)

    assert len(seq.block_table) == 4
    assert len(manager.used_block_ids) == 4


def test_can_allocate_rejects_when_one_chunk_does_not_fit(make_sequence):
    manager = make_block_manager(num_blocks=1)
    seq = make_sequence(800)

    assert manager.can_allocate(seq) is False
    assert seq.block_table == []
    assert manager.used_block_ids == set()
    assert list(manager.free_block_ids) == [0]


def test_deallocate_releases_every_owned_block(make_sequence):
    manager = make_block_manager()
    seq = make_sequence(800)

    manager.allocate(seq)
    seq.num_chunked_tokens = 512
    manager.allocate(seq)
    manager.deallocate(seq)

    assert seq.block_table == []
    assert manager.used_block_ids == set()
    assert set(manager.free_block_ids) == set(range(8))
    assert all(block.ref_count == 0 for block in manager.blocks)
