"""lang parity: narde.lang must match laya.lang exactly.

`detect_script` / `guess_latin_language` take raw strings and raise TypeError on
dict/list/None in BOTH packages -- parity means mirroring the raise, not the crash.
`analyse` and `is_english` accept str/dict/list/None and must return equal values.
"""
import pf  # noqa: F401  (bootstrap sys.path)
import laya.lang as L
import narde.lang as N

CASES = [
    # (label, state) — state may be str / dict / list / None
    ("english text", "Please refund my duplicate charge from last Tuesday, the amount was wrong."),
    ("german text", "Mein Konto wurde zweimal belastet. Bitte erstatten Sie die Zahlung zuruck."),
    ("french text", "Mon compte a été débité deux fois, merci de faire un avoir."),
    ("french deaccented", "Mon compte a ete debite deux fois, merci de faire un avoir."),
    ("spanish text", "Mi cuenta fue debitada dos veces, por favor reembolsen el cargo."),
    ("portuguese", "A minha conta foi debitada duas vezes, por favor reembolsa o valor."),
    ("italian", "Il mio conto è stato addebitato due volte, per favore rimborsami."),
    ("romanian", "Contul meu a fost debitat de doua ori, te rog sa ma restituie banii."),
    ("polish", "Moje konto zostalo obciazone dwa razy, prosze o zwrot platnosci."),
    ("czech", "Můj účet byl stržen dvakrát, prosím o vrácení platby."),
    ("hungarian", "A számlámat kétszer vették le, kérlek térítsd vissza a fizetést."),
    ("turkish", "Hesabim iki kez tahsilat yapildi, lutfen iade yapin."),
    ("hindi devanagari", "मरा खतातान दोनद रकम कडलेली आहे, कृपया परतफड करा."),
    ("korean", "제 계좌서 두 번 인출되었습니다. 환금해 주세요."),
    ("chinese", "我的账户被扣了两次款，请退款。"),
    ("japanese kana", "口座から2回引き落としがありました。返金してください。"),
    ("arabic", "تم خصم المبلغ من حسابي مرتين، أرجو استرداد المبلغ."),
    ("thai", "บญชีของฉนถูกหักเงินสองครั้่ง กรุณาคืนเงิน"),
    ("mixed en+de", "Hello bitte stornieren Sie die Zahlung von Monday. Please confirm."),
    ("empty", ""),
    ("digits only", "12345 67890"),
    ("short en", "hi"),
]

# States where the raw-string helpers crash identically in both packages.
RAW_CRASH_CASES = [
    ("state dict", {"message": "Charge twice on my card, refund please", "from": "x@y.z"}),
    ("state list", ["user: charge twice", "agent: we'll look", "user: urgent!!"]),
    ("none", None),
]


def _call(fn, state):
    """Return ('ok', value) or ('err', 'ExcType: message') mirroring both packages."""
    try:
        return ("ok", fn(state))
    except Exception as e:  # noqa: BLE001 - mirror any exception type/message
        return ("err", f"{type(e).__name__}: {e}")


def test_detect_script_parity():
    for label, state in CASES:
        assert L.detect_script(state) == N.detect_script(state), f"script differs on: {label}"
    for label, state in RAW_CRASH_CASES:
        lr = _call(L.detect_script, state)
        nr = _call(N.detect_script, state)
        assert lr == nr, f"detect_script crash differs on {label!r}: {lr} vs {nr}"


def test_analyse_parity():
    for label, state in CASES + RAW_CRASH_CASES:
        la = L.analyse(state)
        na = N.analyse(state)
        assert la == na, f"analyse differs on {label!r}:\n laya={la}\n narde={na}"


def test_is_english_parity():
    for label, state in CASES + RAW_CRASH_CASES:
        assert L.is_english(state) == N.is_english(state), f"is_english differs: {label}"


def test_guess_latin_language_parity():
    for label, state in CASES:
        assert L.guess_latin_language(state) == N.guess_latin_language(state), f"guess differs: {label}"
    for label, state in RAW_CRASH_CASES:
        lr = _call(L.guess_latin_language, state)
        nr = _call(N.guess_latin_language, state)
        assert lr == nr, f"guess_latin_language crash differs on {label!r}: {lr} vs {nr}"
