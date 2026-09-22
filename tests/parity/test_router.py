"""router parity: narde.router must match laya.router (route decisions, aliases, LRU)."""
import pf  # noqa: F401  (bootstrap sys.path)
import laya.router as L
import narde.router as N


def test_normalise_name_parity():
    valid = ["en", "english", "laya", "default", "multi", "ml", "laya-multilingual",
             "multilingual", "typed", "typed_decisions", "laya-typed-decisions",
             "decisions", "typed-decisions"]
    for name in valid:
        assert L.normalise_name(name) == N.normalise_name(name), name

    # Unknown names must raise the same ValueError message on both sides.
    for bad in ("nope", "french", "  Mixed-Case "):
        try:
            L.normalise_name(bad)
            l_raised = False
        except ValueError as e:
            l_raised, l_msg = True, str(e)
        try:
            N.normalise_name(bad)
            n_raised = False
        except ValueError as e:
            n_raised, n_msg = True, str(e)
        assert l_raised is n_raised and l_raised is True, bad
        if l_raised:
            assert l_msg == n_msg, (l_msg, n_msg)


def test_match_workflow_parity():
    cases = [
        {"action": 1, "needs_review": 1, "outcome": 1, "risk": 1, "urgency": 1},
        {"action": 1, "category": 1, "churn_risk": 1, "needs_human": 1, "urgency": 1},
        {"discrepancy_severity": 1, "disposition": 1, "duplicate": 1, "matches_order": 1, "urgency": 1},
        {"credential_compromise": 1, "disposition": 1, "severity": 1, "true_positive": 1, "urgency": 1},
        {"action": 1, "urgency": 1},                      # subset -> None
        {"action": 1, "needs_review": 1, "outcome": 1, "risk": 1, "urgency": 1, "extra": 1},  # superset -> None
        {},
        None,
    ]
    for c in cases:
        assert L.match_typed_decisions_workflow(c) == N.match_typed_decisions_workflow(c), c


def _compare_decisions(ld, nd):
    assert ld.model == nd.model
    assert ld.reason == nd.reason, f"{ld.reason!r} != {nd.reason!r}"
    assert ld["repo"] == nd["repo"]
    assert ld.get("workflow") == nd.get("workflow")
    ld_det, nd_det = ld.get("detection"), nd.get("detection")
    assert ld_det == nd_det, f"detection differs:\n{ld_det}\nvs\n{nd_det}"


STATES = [
    "Please refund the duplicate charge on my card.",
    "Mein Konto wurde zweimal belastet, bitte erstatten Sie.",
    "Mon compte a été débité deux fois.",
    "मरा खतात दोनदा रक्कम काढलेली आहे.",
    "제 계좌에서 두 번 인출되었습니다.",
    "我的账户被扣了两次款。",
    "تم خصم المبلغ مرتين.",
    "",
    "!!! ???",
    {"message": "I was charged twice, please refund."},
    ["hello there friend", "this is a test message from the user today"],
]


def test_route_autodetect_parity():
    lr, nr = L.Router(), N.Router()
    for state in STATES:
        _compare_decisions(lr.route(state), nr.route(state))


def test_route_explicit_parity():
    lr, nr = L.Router(), N.Router()
    questions = {"urgency": {"type": "noul", "instructions": "urgent?"}}
    kwargs_sets = [
        dict(model="en"),
        dict(model="multilingual"),
        dict(model="typed"),
        dict(task="typed_decisions"),
        dict(task="english"),
        dict(lang="fr"),
        dict(lang="en-US"),
        dict(lang="de"),
        dict(model="laya", task="multi", lang="ja"),  # model wins
        dict(task="typed", lang="ja"),                # task wins
    ]
    for kw in kwargs_sets:
        _compare_decisions(lr.route("irrelevant state text", questions, **kw),
                           nr.route("irrelevant state text", questions, **kw))


def test_route_workflow_detection_parity():
    wf_q = {"action": {"type": "noul", "instructions": "x"},
            "needs_review": {"type": "noul", "instructions": "x"},
            "outcome": {"type": "noul", "instructions": "x"},
            "risk": {"type": "score", "instructions": "x", "criteria": ["a", "b"]},
            "urgency": {"type": "score", "instructions": "x", "criteria": ["a", "b"]}}
    for auto in (False, True):
        lr, nr = L.Router(auto_task_detection=auto), N.Router(auto_task_detection=auto)
        _compare_decisions(lr.route("text", wf_q), nr.route("text", wf_q))
        # lang explicit beats auto workflow
        _compare_decisions(lr.route("text", wf_q, lang="de"), nr.route("text", wf_q, lang="de"))


def test_lru_lifecycle_parity():
    import torch

    class FakeAgent:
        def __init__(self, repo, device=None, token=None, subfolder=None):
            self.repo = repo
            self.subfolder = subfolder
            self.device = torch.device("cpu")

    import laya.agent as laya_agent
    import narde.agent as narde_agent
    orig_l, orig_n = laya_agent.Agent, narde_agent.Agent
    laya_agent.Agent = FakeAgent
    narde_agent.Agent = FakeAgent
    try:
        for standalone in (False, True):
            lr = L.Router(max_loaded=2, standalone_repos=standalone)
            nr = N.Router(max_loaded=2, standalone_repos=standalone)
            for name in ("english", "multilingual", "typed-decisions", "english"):
                la, na = lr.load(name), nr.load(name)
                assert type(la) is type(na)
                assert (la.repo, la.subfolder) == (na.repo, na.subfolder), (name, la, na)
                assert lr.loaded == nr.loaded, (name, lr.loaded, nr.loaded)
            assert len(lr.loaded) == 2 and len(nr.loaded) == 2
            assert lr.loaded == nr.loaded
            # attach + unload
            lr.attach("english", FakeAgent("x"))
            nr.attach("english", FakeAgent("x"))
            assert lr.loaded == nr.loaded
            lr.unload("english")
            nr.unload("english")
            assert lr.loaded == nr.loaded
            lr.unload()
            nr.unload()
            assert lr.loaded == nr.loaded == []
    finally:
        laya_agent.Agent, narde_agent.Agent = orig_l, orig_n


def test_route_decision_repr():
    d_l = L.RouteDecision(model="english", repo="x", reason="r", detection=None, workflow=None)
    d_n = N.RouteDecision(model="english", repo="x", reason="r", detection=None, workflow=None)
    assert repr(d_l) == repr(d_n)
    assert d_l.model == d_n.model == "english"
    assert d_l.reason == d_n.reason
