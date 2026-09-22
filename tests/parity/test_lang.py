"""lang parity: narde.lang must match laya.lang exactly."""
import pf  # noqa: F401  (bootstrap sys.path)
import laya.lang as L
import narde.lang as N

CASES = [
    # (label, state) — state may be str / dict / list / None
    ("english text", "Please refund my duplicate charge from last Tuesday, the amount was wrong."),
    ("german text", "Mein Konto wurde zweimal belastet. Bitte erstatten Sie die Zahlung zurück."),
    ("french text", "Mon compte a été débité deux fois, merci de faire un avoir."),
    ("french deaccented", "Mon compte a ete debite deux fois, merci de faire un avoir."),
    ("spanish text", "Mi cuenta fue debitada dos veces, por favor reembolsen el cargo."),
    ("portuguese", "A minha conta foi debitada duas vezes, por favor reembolsa o valor."),
    ("italian", "Il mio conto è stato addebitato due volte, per favore rimborsami."),
    ("romanian", "Contul meu a fost debitat de doua ori, te rog sa ma restituie banii."),
    ("polish", "Moje konto zostało obciążone dwa razy, proszę o zwrot płatności."),
    ("czech", "Můj účet byl stržen dvakrát, prosím o vrácení platby."),
    ("hungarian", "A számlámat kétszer vették le, kérlek térítsd vissza a fizetést."),
    ("turkish", "Hesabım iki kez tahsilat yapıldı, lütfen iade yapın."),
    ("hindi devanagari", "मरा खतात दोनदा रक्कम काढलेली आहे, कृपया परतफेड करा."),
    ("korean", "제 계좌에서 두 번 인출되었습니다. 환불해 주세요."),
    ("chinese", "我的账户被扣了两次款，请退款。"),
    ("japanese kana", "口座から2回引き落としがありました。返金してください。"),
    ("arabic", "تم خصم المبلغ من حسابي مرتين، أرجو استرداد المبلغ."),
    ("thai", "บัญชีของฉันถูกหักเงินสองครั้ง กรุณาคืนเงิน"),
    ("mixed en+de", "Hello bitte stornieren Sie die Zahlung von Monday. Please confirm."),
    ("empty", ""),
    ("digits only", "12345 67890"),
    ("short en", "hi"),
    ("state dict", {"message": "Charge twice on my card, refund please", "from": "x@y.z"}),
    ("state list", ["user: charge twice", "agent: we'll look", "user: urgent!!"]),
    ("none", None),
]


def test_detect_script_parity():
    for label, state in CASES:
        assert L.detect_script(state) == N.detect_script(state), f"script differs on: {label}"


def test_analyse_parity():
    for label, state in CASES:
        la = L.analyse(state)
        na = N.analyse(state)
        assert la == na, f"analyse differs on {label!r}:\n laya={la}\n narde={na}"


def test_is_english_parity():
    for label, state in CASES:
        assert L.is_english(state) == N.is_english(state), f"is_english differs: {label}"


def test_guess_latin_language_parity():
    for label, state in CASES:
        assert L.guess_latin_language(state) == N.guess_latin_language(state), f"guess differs: {label}"
