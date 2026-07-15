from nanovllm.engine.sequence import SequenceStatus


def test_first_schedule_selects_one_chunk_without_committing_it(
    make_scheduler,
    make_sequence,
    assert_block_invariants,
):
    scheduler = make_scheduler()
    seq = make_sequence(800)
    scheduler.add(seq)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert scheduled == [seq]
    assert seq.num_scheduled_tokens == 512
    assert seq.num_chunked_tokens == 0
    assert len(seq.block_table) == 2
    assert seq.status is SequenceStatus.WAITING
    assert_block_invariants(scheduler)


def test_partial_prefill_completes_in_two_steps_then_enters_decode(
    make_scheduler,
    make_sequence,
    assert_block_invariants,
):
    scheduler = make_scheduler()
    seq = make_sequence(800)
    scheduler.add(seq)

    first_batch, first_is_prefill = scheduler.schedule()
    assert first_is_prefill is True
    assert seq.num_scheduled_tokens == 512

    scheduler.postprocess(first_batch, [None])

    assert seq.num_chunked_tokens == 512
    assert seq.num_completion_tokens == 0
    assert len(seq.block_table) == 2
    assert seq.status is SequenceStatus.WAITING
    assert_block_invariants(scheduler)

    second_batch, second_is_prefill = scheduler.schedule()

    assert second_is_prefill is True
    assert second_batch == [seq]
    assert seq.num_scheduled_tokens == 288
    assert seq.num_chunked_tokens == 512
    assert len(seq.block_table) == 4
    assert_block_invariants(scheduler)

    scheduler.postprocess(second_batch, [42])

    assert seq.num_chunked_tokens == 800
    assert seq.completion_token_ids == [42]
    assert seq.status is SequenceStatus.RUNNING
    assert_block_invariants(scheduler)

    decode_batch, is_prefill = scheduler.schedule()

    assert is_prefill is False
    assert decode_batch == [seq]
