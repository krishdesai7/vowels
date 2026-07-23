from vowels.schema import COLORS, Dialect, Wells

# A length mark is not a second vowel quality, and a trailing rhotic is
# r-colouring rather than a glide. Strip both and a genuine diphthong is what
# still has two vowel symbols left.
LENGTH_MARK = "ː"


def _nucleus(ipa: str) -> str:
    return ipa.replace(LENGTH_MARK, "").removesuffix("r")


def test_wells_value_is_its_own_name() -> None:
    # Mixed-case names must survive verbatim: the label parser does not
    # normalise case, so haPPY/coMMA/leTTER would be corrupted by auto().
    for w in Wells:
        assert w.value == w.name
    assert str(Wells.haPPY) == "haPPY"
    assert str(Wells.PRICE) == "PRICE"


def test_every_set_has_a_colour() -> None:
    assert {w: COLORS[w] for w in Wells}.keys() == set(Wells)
    assert all(Wells[w.name].color.startswith("#") for w in Wells)
    # Colours must be distinguishable, so no set may reuse another's.
    assert len(set(COLORS.values())) == len(COLORS)


def test_every_dialect_covers_every_set() -> None:
    for dialect in Dialect:
        assert set(dialect.profile.expected) == set(Wells)


def test_declared_diphthongs_match_the_transcriptions() -> None:
    # Guards the prior against drifting away from the IPA table it describes.
    for dialect in Dialect:
        profile = dialect.profile
        derived = {w for w, ipa in profile.expected.items() if len(_nucleus(ipa)) > 1}
        assert derived == set(profile.diphthongs), dialect


def test_declared_r_colored_match_the_transcriptions() -> None:
    for dialect in Dialect:
        profile = dialect.profile
        derived = {w for w, ipa in profile.expected.items() if ipa.endswith("r")}
        assert derived == set(profile.r_colored), dialect


def test_rhoticity_agrees_with_the_transcriptions() -> None:
    for dialect in Dialect:
        profile = dialect.profile
        has_rhotics = any(ipa.endswith("r") for ipa in profile.expected.values())
        assert profile.rhotic is has_rhotics, dialect


def test_rp_keeps_centring_diphthongs_that_ga_treats_as_rhotic() -> None:
    rp, ga = Dialect.RP.profile, Dialect.GA.profile
    centring = {Wells.NEAR, Wells.SQUARE, Wells.CURE}
    assert centring <= rp.diphthongs
    assert centring.isdisjoint(ga.diphthongs)
    assert centring <= ga.r_colored
    # The gliding core is dialect-independent.
    assert {Wells.FACE, Wells.GOAT, Wells.PRICE, Wells.CHOICE, Wells.MOUTH} <= (
        rp.diphthongs & ga.diphthongs
    )
