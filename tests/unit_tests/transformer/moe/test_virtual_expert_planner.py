# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-local coverage for the virtual-expert load balancer's host side: runtime parameters
and pointer tables, the backward hooks, and the non-GTP config rules.

Every test here runs in one process on one GPU, so a plain ``pytest
tests/unit_tests/transformer/moe/test_virtual_expert_planner.py`` runs the whole file.

The planner kernel exchanges histograms between ranks itself, so its placement and route
mapping are covered by the four-rank kernel tier in ``test_virtual_expert_triton.py``;
end-to-end gradient parity lives in ``test_virtual_expert_hybridep.py``.
"""

import os
import weakref
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.experts import _VirtualExpertFC2WgradStore
from megatron.core.transformer.moe.virtual_expert_load_balancer import (
    VirtualExpertLoadBalancer,
    VirtualExpertPlan,
    WeightDirection,
    _VirtualExpertHook,
    _VirtualExperts,
)
from tests.unit_tests.transformer.test_transformer_config import _virtual_expert_hybridep_config

# The GB200 CI bucket launches marked files with four ranks, which these tests need.
pytestmark = pytest.mark.launch_on_gb200

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module", autouse=True)
def _select_rank_device():
    """Select the rank's GPU before these tests populate TE's device-implicit caches."""
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _load_balancer(virtual_experts=None):
    """A balancer with only the state the process-local paths read; no transport, no arenas."""
    load_balancer = VirtualExpertLoadBalancer.__new__(VirtualExpertLoadBalancer)
    load_balancer.virtual_experts = virtual_experts
    load_balancer._plan = None
    return load_balancer


@pytest.mark.parametrize(
    ("ep_size", "num_experts", "num_local_experts", "topk"),
    [
        (1, 2, 2, 2),
        (65, 130, 2, 2),
        (2, 0, 0, 1),
        (2, 8194, 4097, 2),
        (4, 7, 1, 2),
        (2, 4, 1, 2),
        (2, 4, 2, 0),
        (2, 4, 2, 5),
        (2, 64, 32, 33),
    ],
)
def test_virtual_expert_init_rejects_layout_before_cuda(
    monkeypatch, ep_size, num_experts, num_local_experts, topk
):
    """Validate actual process-group size and expert ownership before touching CUDA."""
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: ep_size)
    monkeypatch.setattr(
        torch.cuda, "current_device", lambda: pytest.fail("invalid layout reached CUDA")
    )
    with pytest.raises(ValueError, match="requires 2..64 EP ranks"):
        VirtualExpertLoadBalancer().initialize_virtual_expert_load_balancer(
            group=object(),
            num_local_experts=num_local_experts,
            router_topk=topk,
            num_experts=num_experts,
            config=None,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"moe_flex_dispatcher_num_sms": 0}, "num_sms>0"),
        ({"moe_flex_dispatcher_num_sms": -1}, "num_sms>0"),
        ({"moe_hybridep_num_sms": 0}, "num_sms>0"),
        ({"moe_deepep_num_sms": -1}, "num_sms>0"),
        ({"moe_layer_recompute": True}, "no MoE layer recompute"),
        (
            {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            },
            "no MoE layer recompute",
        ),
        ({"moe_router_load_balancing_type": "sinkhorn"}, "no sinkhorn"),
        (
            {
                "moe_router_load_balancing_type": ["aux_loss", "sinkhorn"],
                "moe_aux_loss_coeff": [0.01, 0.0],
            },
            "no sinkhorn",
        ),
    ],
)
def test_virtual_expert_init_checks_normalized_settings(monkeypatch, overrides, message):
    """Initialization sees canonical settings even when the caller uses deprecated aliases."""
    config = _virtual_expert_hybridep_config(**overrides)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    with pytest.raises(ValueError, match=message):
        VirtualExpertLoadBalancer().initialize_virtual_expert_load_balancer(
            group=object(), num_local_experts=1, router_topk=2, num_experts=2, config=config
        )


@requires_cuda
def test_virtual_expert_compact_api_requirement_preserves_plain_hybridep(monkeypatch):
    """Old HybridEP remains usable without virtual experts; virtual experts fail before launch."""
    from megatron.core.transformer.moe import token_dispatcher

    monkeypatch.setattr(token_dispatcher, "hybrid_ep_dense_topk_routing", lambda *_: False)
    monkeypatch.setattr(token_dispatcher, "hybrid_ep_dispatch", object())
    for virtual in (False, True):
        config = _virtual_expert_hybridep_config(moe_virtual_expert_load_balance=virtual)
        if virtual:
            with pytest.raises(ValueError, match="compact top-k routing API"):
                token_dispatcher._HybridEPManager(object(), 1, 2, config)
        else:
            manager = token_dispatcher._HybridEPManager(object(), 1, 2, config)
            assert not manager._dense_topk_routing


@requires_cuda
@pytest.mark.parametrize(
    ("ep_size", "num_experts", "topk", "routing"),
    [(2, 2, 1, "none"), (64, 8192, 32, "seq_aux_loss"), (64, 512, 10, "quantile_balancing")],
)
def test_virtual_expert_init_accepts_supported_limits(
    monkeypatch, ep_size, num_experts, topk, routing
):
    """Large positive HybridEP SM budgets are valid; transport caps its own budget later."""
    config = _virtual_expert_hybridep_config(
        moe_flex_dispatcher_num_sms=64,
        moe_router_load_balancing_type=routing,
        moe_router_fusion=routing != "quantile_balancing",
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: ep_size)
    manager = VirtualExpertLoadBalancer()
    manager.initialize_virtual_expert_load_balancer(
        group=object(),
        num_local_experts=num_experts // ep_size,
        router_topk=topk,
        num_experts=num_experts,
        config=config,
    )
    assert (manager.ep_size, manager.num_owned_experts, manager.router_topk) == (
        ep_size,
        num_experts // ep_size,
        topk,
    )


def test_virtual_expert_rank_capacity_includes_per_expert_padding():
    """Size the static dropless capacity for HybridEP's per-segment alignment."""
    load_balancer = _load_balancer()
    load_balancer.router_topk, load_balancer.num_owned_experts = 22, 32
    load_balancer.config = SimpleNamespace(moe_expert_rank_capacity_factor=1.0)
    load_balancer._alignment = 256
    assert load_balancer._compute_rank_capacity(8192) == 196608
    load_balancer.config.moe_expert_rank_capacity_factor = 2.0
    assert load_balancer._compute_rank_capacity(8192) == 360448
    load_balancer.config.moe_expert_rank_capacity_factor = 1.0
    load_balancer._alignment = 0
    assert load_balancer._compute_rank_capacity(8192) == 180224


def test_virtual_expert_hooks_do_not_keep_their_owner_alive():
    """A method captured by an autograd context must not close a cycle through its owner, and
    is called with its arguments when the gradient passes."""

    class Owner:
        def __init__(self):
            self.calls = []

        def method(self, value):
            self.calls.append(value)

    owner = Owner()
    hidden = torch.ones((), requires_grad=True)
    _VirtualExpertHook.apply(hidden, owner.method, (7,)).backward()
    assert owner.calls == [7]
    kept = _VirtualExpertHook.apply(hidden, owner.method, (8,))
    alive = weakref.ref(owner)
    del owner
    assert alive() is None and kept.requires_grad


def test_virtual_expert_backward_hooks_span_the_transport_window():
    """Order the four transport hooks across one layer's backward, as the dispatcher places them,
    and deliver what the finish returns as the source parameters' gradients."""
    events = []
    plan = VirtualExpertPlan(None, None)

    class FakeLoadBalancer:
        source_parameters = (torch.nn.Parameter(torch.ones(())), torch.nn.Parameter(torch.ones(())))

        def __init__(self):
            self.source_grads = tuple(
                torch.full_like(parameter, index + 1)
                for index, parameter in enumerate(self.source_parameters)
            )

        def _start_backward(self, current):
            assert current is plan
            events.append("start_weight_push")

        def _prepare_expert_backward(self):
            events.append("wait_weight_push")

        def _start_pending_grad_reduces(self):
            events.append("start_grad_reduce")

        def _finish_grad_reduce(self):
            events.append("finish_grad_reduce")
            return self.source_grads

    class BackwardMarker(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, label):
            ctx.label = label
            return value

        @staticmethod
        def backward(ctx, grad):
            events.append(ctx.label)
            return grad, None

    load_balancer = FakeLoadBalancer()
    hidden = torch.ones((), requires_grad=True)
    hidden = _VirtualExpertHook.apply(
        hidden, load_balancer._finish_grad_reduce, (), *load_balancer.source_parameters
    )
    hidden = BackwardMarker.apply(hidden, "router_and_shared_expert_backward")
    hidden = _VirtualExpertHook.apply(hidden, load_balancer._start_pending_grad_reduces, ())
    hidden = BackwardMarker.apply(hidden, "dispatch_backward")
    hidden = BackwardMarker.apply(hidden, "expert_backward")
    hidden = _VirtualExpertHook.apply(hidden, load_balancer._prepare_expert_backward, ())
    hidden = BackwardMarker.apply(hidden, "combine_backward")
    hidden = BackwardMarker.apply(hidden, "latent_up_fc_layer_backward")
    hidden = _VirtualExpertHook.apply(hidden, load_balancer._start_backward, (plan,))
    hidden.backward()

    assert events == [
        "start_weight_push",
        "latent_up_fc_layer_backward",
        "combine_backward",
        "wait_weight_push",
        "expert_backward",
        "dispatch_backward",
        "start_grad_reduce",
        "router_and_shared_expert_backward",
        "finish_grad_reduce",
    ]
    for index, parameter in enumerate(load_balancer.source_parameters):
        torch.testing.assert_close(
            parameter.grad, torch.full_like(parameter, index + 1), rtol=0, atol=0
        )


def test_virtual_expert_fc2_reduction_starts_from_the_wgrad_store_and_fc1_after_dispatch():
    """FC2 reduces behind its own wgrad GEMM; only FC1 waits for dispatch backward."""
    started = []
    load_balancer = _load_balancer()
    load_balancer._plan = VirtualExpertPlan(None, None)

    def record(fc_layer, owner=weakref.proxy(load_balancer)):
        started.append(fc_layer)
        owner._plan.started.add(fc_layer)

    load_balancer._start_grad_reduce = record

    # TE's delayed-wgrad protocol hands the GEMM to the store instead of
    # launching it; the store runs it and starts FC2's reduction right behind.
    gemm_calls = []
    store = _VirtualExpertFC2WgradStore(load_balancer)
    assert store.delay_wgrad_compute() and store.context is None
    store.put(["x", "dy", "out"], lambda *tensors: gemm_calls.append(tensors))
    assert gemm_calls == [("x", "dy", "out")]
    assert started == [1]

    load_balancer._start_pending_grad_reduces()
    assert started == [1, 0]

    # A reduction cannot start before the backward push was waited for (the expert backward's
    # preparation), nor outside a backward pass.
    load_balancer._plan = VirtualExpertPlan(None, None, push_in_flight=True)
    with pytest.raises(RuntimeError, match="prepared backward"):
        VirtualExpertLoadBalancer._start_grad_reduce(load_balancer, 1)

    load_balancer._plan = None
    with pytest.raises(RuntimeError, match="prepared backward"):
        VirtualExpertLoadBalancer._start_grad_reduce(load_balancer, 1)

    alive = weakref.ref(load_balancer)
    del load_balancer
    assert alive() is None, "TE's wgrad store must not retain the load balancer"


MEMBER_SHAPE = (128, 128)


def _mxfp8(tensor):
    """Quantize a BF16 tensor into an MXFP8 tensor holding both GEMM orientations."""
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    return MXFP8Quantizer(DType.kFloat8E4M3)(tensor)


def _fake_virtual_experts(mxfp8, device, num_local_experts, template, staging):
    """The slots object over plain tensors: no symmetric memory, no group."""
    numel = MEMBER_SHAPE[0] * MEMBER_SHAPE[1]

    # Isolate shared class state between tests; exercise the real per-layer constructor.
    class LocalVirtualExperts(_VirtualExperts):
        pass

    virtual_experts = LocalVirtualExperts
    virtual_experts.config = SimpleNamespace(
        mxfp8=mxfp8,
        member_shapes=(MEMBER_SHAPE,),
        grad_dtype=staging.dtype if staging is not None else torch.float32,
        device=device,
    )
    virtual_experts.num_local_experts = num_local_experts
    virtual_experts._weight_sections = [
        num_local_experts * numel,
        num_local_experts * (numel // 32 if mxfp8 else 0),
    ]
    virtual_experts._grad_sections = [num_local_experts * numel]
    virtual_experts.weight_arena = torch.zeros(
        sum(virtual_experts._weight_sections),
        dtype=torch.uint8 if mxfp8 else torch.bfloat16,
        device=device,
    )
    virtual_experts.grad_arena = torch.zeros(
        sum(virtual_experts._grad_sections), dtype=virtual_experts.config.grad_dtype, device=device
    )
    virtual_experts.slot_weights = (virtual_experts._slot_parameters(0, template),)
    virtual_experts.native_staging = (staging,)
    return virtual_experts


def _build_fc_layer(parameters, template, staging, *, mxfp8=False):
    """Build a FC layer over shared slots and unsharded native gradient staging."""
    virtual_experts = _fake_virtual_experts(
        mxfp8, parameters[0].device, len(parameters), template, staging
    )
    return virtual_experts(None, virtual_experts.config, (parameters,))


def _weight_push(monkeypatch, fc_layer):
    """Exercise the real push binding and stream handoff, replacing only the transport kernel."""
    balancer = _load_balancer(fc_layer)
    balancer.device = fc_layer.config.device
    balancer._plan = VirtualExpertPlan(None, None)
    balancer.prefetch_done = torch.cuda.Event()
    stream = torch.cuda.Stream()
    balancer._weight_stream = lambda current: stream
    monkeypatch.setattr(
        "megatron.core.transformer.moe.virtual_expert_load_balancer."
        "launch_virtual_expert_weight_prefetch",
        lambda *args, **kwargs: None,
    )

    def push(direction):
        balancer._start_weight_push(direction)
        balancer._wait_weight_push()
        return fc_layer._tables[0][direction][0]

    return balancer, push


def _data_ptrs(weights, weight_format, direction):
    """Return the per-expert data pointers the push reads for one direction."""
    if weight_format == "bf16":
        return [weight.data_ptr() for weight in weights]
    field = "_rowwise_data" if direction == WeightDirection.FORWARD else "_columnwise_data"
    return [getattr(weight, field).data_ptr() for weight in weights]


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_plain_fc_layer_binds_its_tables_once(weight_format):
    """Plain parameters and their staging never move: the first weight table of a direction is
    written once, every later call checks that the pointers it was built from still hold."""
    device = torch.device("cuda", torch.cuda.current_device())
    mxfp8 = weight_format == "mxfp8"
    weights = [torch.randn(MEMBER_SHAPE, dtype=torch.bfloat16, device=device) for _ in range(3)]
    parameters = tuple(torch.nn.Parameter(_mxfp8(w) if mxfp8 else w) for w in weights)
    staging = torch.zeros(
        (3, *MEMBER_SHAPE), dtype=torch.bfloat16 if mxfp8 else torch.float32, device=device
    )
    fc_layer = _build_fc_layer(parameters, parameters[0], staging, mxfp8=mxfp8)
    torch.cuda.synchronize(device)
    assert fc_layer.native_grads[0] is staging
    assert not fc_layer._tables[0]
    assert fc_layer.grad_tables()[0].tolist() == [grad.data_ptr() for grad in staging]
    assert [parameter.main_grad.data_ptr() for parameter in fc_layer.runtime_weights[0][:3]] == [
        grad.data_ptr() for grad in staging
    ]
    # The first call of each direction writes its table; every later call returns the same
    # device tensor after validating.
    tables = {
        direction: fc_layer.get_weight_table(0, direction, parameters)
        for direction in WeightDirection
    }
    runtime_ptrs = {d: fc_layer._ptrs(d, fc_layer.runtime_weights[0]) for d in WeightDirection}
    grad_ptrs = [p.main_grad.data_ptr() for p in fc_layer.runtime_weights[0]]
    for _ in range(3):
        for direction in WeightDirection:
            table = fc_layer.get_weight_table(0, direction, parameters)
            assert table is tables[direction]
            assert table[0].tolist() == _data_ptrs(parameters, weight_format, direction)
            assert fc_layer._ptrs(direction, fc_layer.runtime_weights[0]) == runtime_ptrs[direction]
            assert [p.main_grad.data_ptr() for p in fc_layer.runtime_weights[0]] == grad_ptrs

    # The storage is static by contract: a moved parameter is an error, not a rebind.
    if mxfp8:
        parameters[0]._rowwise_data = parameters[0]._rowwise_data.clone()
    else:
        parameters[0].data = parameters[0].data.clone()
    with pytest.raises(RuntimeError, match="moved"):
        fc_layer.get_weight_table(0, WeightDirection.FORWARD, parameters)


@requires_cuda
def test_virtual_expert_plain_fc_layer_hands_wgrads_to_the_source_parameters():
    """Accumulate the FP32 staging into ``main_grad`` and return a BF16 dummy for autograd; a
    parameter without a main_grad gets a copy of the staging, never an alias."""
    device = torch.device("cuda", torch.cuda.current_device())
    parameters = tuple(
        torch.nn.Parameter(torch.ones(MEMBER_SHAPE, dtype=torch.bfloat16, device=device))
        for _ in range(2)
    )
    parameters[0].main_grad = torch.zeros(MEMBER_SHAPE, dtype=torch.float32, device=device)
    parameters[0].grad_added_to_main_grad = False
    staging = torch.full((2, *MEMBER_SHAPE), 1.0001, dtype=torch.float32, device=device)
    fc_layer = _build_fc_layer(parameters, parameters[0], staging)

    grads = fc_layer.hand_off_wgrads(0)
    torch.testing.assert_close(parameters[0].main_grad, staging[0], rtol=0, atol=0)
    assert parameters[0].grad_added_to_main_grad
    assert grads[0].dtype == torch.bfloat16 and tuple(grads[0].shape) == MEMBER_SHAPE
    torch.testing.assert_close(grads[1], staging[1], rtol=0, atol=0)
    assert grads[1].data_ptr() != staging[1].data_ptr()


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
@pytest.mark.parametrize("bind_table_first", [False, True], ids=["first_table", "cached_table"])
def test_virtual_expert_rejects_moved_runtime_storage(monkeypatch, weight_format, bind_table_first):
    """Every table lookup checks the runtime binding, including before a table exists."""
    device = torch.device("cuda", torch.cuda.current_device())
    weights = [torch.randn(MEMBER_SHAPE, dtype=torch.bfloat16, device=device) for _ in range(2)]
    parameters = tuple(
        torch.nn.Parameter(_mxfp8(w) if weight_format == "mxfp8" else w) for w in weights
    )
    staging = torch.zeros((2, *MEMBER_SHAPE), dtype=torch.float32, device=device)
    fc_layer = _build_fc_layer(parameters, parameters[0], staging, mxfp8=weight_format == "mxfp8")
    _, push = _weight_push(monkeypatch, fc_layer)
    push(WeightDirection.FORWARD)
    if not bind_table_first:
        fc_layer._tables[0].clear()
    parameter = fc_layer.runtime_weights[0][0]
    if weight_format == "bf16":
        parameter.data = parameter.data.clone()
    else:
        parameter._rowwise_data = parameter._rowwise_data.clone()
    moved_pointer = _data_ptrs([parameter], weight_format, WeightDirection.FORWARD)
    with pytest.raises(RuntimeError, match="runtime storage moved"):
        fc_layer.get_weight_table(0, WeightDirection.FORWARD, parameters)
    assert _data_ptrs([parameter], weight_format, WeightDirection.FORWARD) == moved_pointer


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_owners_share_slots_but_keep_native_bindings_separate(weight_format):
    """Two layers share transport memory, preserve their own weights/tables, and tear down once."""
    device = torch.device("cuda", torch.cuda.current_device())
    mxfp8 = weight_format == "mxfp8"
    weights = [
        torch.full(MEMBER_SHAPE, float(i + 1), dtype=torch.bfloat16, device=device)
        for i in range(2)
    ]
    parameters = [torch.nn.Parameter(_mxfp8(w) if mxfp8 else w) for w in weights]
    staging = torch.zeros((1, *MEMBER_SHAPE), dtype=torch.float32, device=device)
    first = _build_fc_layer((parameters[0],), parameters[0], staging, mxfp8=mxfp8)
    cls, config = type(first), first.config
    second = cls(None, config, ((parameters[1],),))
    assert first.weight_arena is second.weight_arena and first.grad_arena is second.grad_arena
    assert first.runtime_weights[0][0] is not second.runtime_weights[0][0]
    assert first.runtime_weights[0][1] is second.runtime_weights[0][1]
    assert first.native_grads[0] is second.native_grads[0] is staging
    for direction in WeightDirection:
        tables = [owner.weight_tables(direction)[0] for owner in (first, second)]
        assert tables[0] is not tables[1]
        for owner, parameter, table in zip((first, second), parameters, tables):
            expected = _data_ptrs([parameter], weight_format, direction)
            assert table[0].tolist() == expected
            assert _data_ptrs(owner.runtime_weights[0][:1], weight_format, direction) == expected
    bad = SimpleNamespace(**vars(config))
    bad.member_shapes = ((256, 128),)
    with pytest.raises(ValueError, match="share one layout"):
        cls(None, bad, ((parameters[1],),))

    arenas = [weakref.ref(cls.weight_arena), weakref.ref(cls.grad_arena)]
    cls.destroy()
    cls.destroy()  # Idempotent, even while layers retain runtime parameters.
    assert all(ref() is None for ref in arenas), "slot views retained their arena base"
    assert cls.config is None and cls.weight_arena is None and cls.grad_arena is None
    assert cls.slot_weights == cls.native_staging == ()
    for owner in (first, second):
        slot = owner.runtime_weights[0][1]
        assert slot.main_grad is None
        assert (
            all(getattr(slot, name).numel() == 0 for name in ("_rowwise_data", "_columnwise_data"))
            if mxfp8
            else slot.numel() == 0
        )
    for parameter, expected in zip(parameters, weights):
        torch.testing.assert_close(
            parameter.dequantize() if mxfp8 else parameter, expected, rtol=0, atol=0
        )
    alive = weakref.ref(first)
    del first
    assert alive() is None, "shared storage must not retain per-layer owners"


@requires_cuda
def test_virtual_expert_grad_table_is_created_before_reduction_stream_wait(monkeypatch):
    """Binding main_grad does not upload; first reduce creates on the producer stream, then reuses."""
    device = torch.device("cuda", torch.cuda.current_device())
    parameters = tuple(
        torch.nn.Parameter(torch.randn(MEMBER_SHAPE, dtype=torch.bfloat16, device=device))
        for _ in range(2)
    )
    staging = torch.zeros((2, *MEMBER_SHAPE), dtype=torch.float32, device=device)
    owner = _build_fc_layer(parameters, parameters[0], staging)
    balancer = _load_balancer(owner)
    balancer.device = device
    balancer.grad_stream = torch.cuda.Stream()
    balancer.grad_reduce_done = torch.cuda.Event()
    producer = torch.cuda.Stream()
    grads = tuple(staging)
    assert not owner._tables[0]
    seen = []
    get_table = owner.get_weight_table

    def table_on_producer(*args):
        assert torch.cuda.current_stream() == producer
        return get_table(*args)

    def reduction_on_consumer(*args, native_grads, **kwargs):
        assert torch.cuda.current_stream() == balancer.grad_stream
        seen.append(native_grads[0].clone())

    monkeypatch.setattr(owner, "get_weight_table", table_on_producer)
    monkeypatch.setattr(
        "megatron.core.transformer.moe.virtual_expert_load_balancer.launch_virtual_expert_grad_reduce",
        reduction_on_consumer,
    )
    for _ in range(2):
        balancer._plan = VirtualExpertPlan(None, None)
        with torch.cuda.stream(producer):
            balancer._start_grad_reduce(0)
        balancer.grad_reduce_done.synchronize()
        assert seen[-1].tolist() == [g.data_ptr() for g in grads]
        monkeypatch.setattr(
            torch, "tensor", lambda *args, **kwargs: pytest.fail("repeated table upload")
        )
        monkeypatch.setattr(
            owner, "get_weight_table", lambda *args: pytest.fail("rechecking bound table")
        )
    assert len(seen) == 2


@requires_cuda
def test_virtual_expert_shared_allocation_failure_resets_class_state():
    """A failed first allocation must not leave a partly initialized owner for a later model."""

    class FailingVirtualExperts(_VirtualExperts):
        @classmethod
        def _allocate_shared(cls, group, config, templates):
            cls.config = config
            cls.weight_arena = torch.empty(1, device=config.device)
            raise RuntimeError("allocation failed")

    config = SimpleNamespace(device=torch.device("cuda", torch.cuda.current_device()))
    with pytest.raises(RuntimeError, match="allocation failed"):
        FailingVirtualExperts(None, config, ((None,),))
    assert FailingVirtualExperts.config is None and FailingVirtualExperts.weight_arena is None
