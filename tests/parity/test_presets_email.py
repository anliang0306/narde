"""presets + email parity."""
import pf  # noqa: F401  (bootstrap sys.path)
import laya.email as L_email
import laya.presets as L_presets
import narde.email as N_email
import narde.presets as N_presets


def test_presets_parity():
    assert L_presets.triage_questions() == N_presets.triage_questions()
    assert L_presets.email_questions() == N_presets.email_questions()
    assert L_presets.guard_questions() == N_presets.guard_questions()
    assert L_presets.moderation_questions() == N_presets.moderation_questions()
    assert L_presets.router_questions() == N_presets.router_questions()
    custom = {"a": "one", "b": "two"}
    assert L_presets.email_questions(custom) == N_presets.email_questions(custom)


EMAILS = [
    ("simple", "Hello,\n\nI would like to cancel my subscription effective end of month.\n\nThanks,\nBob"),
    ("crlf", "Subject line here\r\n\r\nBody text line one.\r\nSecond line.\r\n\r\nRegards,\r\nAlice"),
    ("quoted reply", "Original message\n\n> On Jan 1, 2025, Bob wrote:\n> old text\n> more old text\n\nMy reply: please help."),
    ("forwarded header", "Hey,\n\n-----Original Message-----\nFrom: boss@x.com\nTo: me@x.com\n\nPlease process the attached invoice.\n\n--\nSent from my iPhone"),
    ("signature dashdash", "Hi team,\n\nThe charge was duplicated on my card, please refund.\n\n-- \nBob Builder\nbob@example.com"),
    ("disclaimer mixed", "Hi,\n\nPlease refund the double charge. This message is confidential and intended solely for the named recipient. If you have received this email in error, please notify us."),
    ("all disclaimer", "This email is confidential and intended solely for the named recipient. If you have received this email in error please delete it."),
    ("long", ("word " * 2000).strip()),
    ("none", None),
    ("empty", ""),
]


def test_clean_email_parity():
    for label, body in EMAILS:
        for limit in (3000, 200, 50):
            assert L_email.clean_email_body(body, max_chars=limit) == N_email.clean_email_body(body, max_chars=limit), label


def test_email_state_parity():
    for label, body in EMAILS:
        for clean in (True, False):
            for sender in (None, "alice@example.com"):
                la = L_email.email_state("Test subject", body, sender=sender, clean=clean, priority="high")
                na = N_email.email_state("Test subject", body, sender=sender, clean=clean, priority="high")
                assert la == na, (label, clean, sender)
    # None sender omitted
    assert "from" not in N_email.email_state("s", "b", sender=None)


def test_email_questions_presets_parity():
    assert L_email.email_questions() == N_email.email_questions()
    assert L_email.email_questions({"x": "y"}) == N_email.email_questions({"x": "y"})
