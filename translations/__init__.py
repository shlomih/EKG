"""
EKG Intelligence — i18n helpers.

Usage in app.py:
    from translations import t, language_selector
    language_selector()          # renders the language picker
    st.title(t("app_title"))
    st.header(t("patient_profile_header"))
"""
import importlib
import streamlit as st

# Registry: display name → language code.
# To add a new language: add an entry here and create translations/<code>.py
LANGUAGES = {
    "English":  "en",
    "Español":  "es",
    "Français": "fr",
    # Uncomment when translation files are ready:
    # "עברית":    "he",
    # "العربية":  "ar",
}

RTL_LANGS = {"he", "ar"}

# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_strings(lang: str) -> dict:
    try:
        mod = importlib.import_module(f"translations.{lang}")
        return mod.STRINGS
    except (ImportError, AttributeError):
        pass
    # Fallback to English
    try:
        from translations import en
        return en.STRINGS
    except (ImportError, AttributeError):
        return {}


def _get_lang() -> str:
    return st.session_state.get("lang", "en")


# ── Public API ────────────────────────────────────────────────────────────────

def t(key: str, **kwargs) -> str:
    """Return the translated string for *key* in the current language.

    Falls back to English if the key is missing in the active language.
    Falls back to the key string itself if the key is missing everywhere.

    kwargs are passed to str.format() — use them for dynamic values:
        t("saved_msg", name="John Doe", pid=42)
    """
    lang = _get_lang()
    strings = _load_strings(lang)
    text = strings.get(key)

    if text is None and lang != "en":
        # Try English fallback
        from translations import en  # noqa: PLC0415
        text = en.STRINGS.get(key, key)

    if text is None:
        text = key

    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text


def inject_rtl_css() -> None:
    """Inject right-to-left CSS when the active language requires it."""
    if _get_lang() in RTL_LANGS:
        st.markdown(
            """
            <style>
            .stApp { direction: rtl; text-align: right; }
            .stTextInput > label, .stSelectbox > label,
            .stSlider > label, .stNumberInput > label,
            .stToggle > label { direction: rtl; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def language_selector() -> None:
    """Render a compact language drop-down and update session state.

    Call this once near the top of app.py, before any translated text is
    rendered, so the page re-renders in the new language immediately.
    """
    if "lang" not in st.session_state:
        st.session_state.lang = "en"

    lang_names = list(LANGUAGES.keys())
    lang_codes = list(LANGUAGES.values())
    current_code = st.session_state.lang
    current_idx = lang_codes.index(current_code) if current_code in lang_codes else 0

    selected_name = st.selectbox(
        "🌐",
        lang_names,
        index=current_idx,
        key="_lang_selector",
        label_visibility="collapsed",
    )
    st.session_state.lang = LANGUAGES[selected_name]
    inject_rtl_css()
