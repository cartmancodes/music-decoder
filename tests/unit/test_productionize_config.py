import dataclasses

import pytest

from music_decoder.productionize.config import (
    COMPOSER_SECTIONS,
    CONGAS,
    HISAAB_BAR_ROOTS,
    HISAAB_GRID,
    HISAAB_SECTIONS,
    HISAAB_V3_REFERENCE,
    PERCUSSION_OPTIONS,
    SHAKER,
    TAMBOURINE,
    Grid,
    PercussionLayer,
    ProductionConfig,
    with_percussion,
)


def test_hisaab_grid_matches_the_build_spec():
    g = HISAAB_GRID
    assert g.bar_s == pytest.approx(2.995, abs=1e-3)
    assert g.section_s == pytest.approx(11.980, abs=1e-3)
    assert g.step_s == pytest.approx(0.1872, abs=1e-4)
    assert g.tempo_bpm == pytest.approx(80.135, abs=1e-3)


def test_time_of_a_hit_is_section_plus_bar_plus_step():
    g = HISAAB_GRID
    assert g.time(2, 3, 10) == pytest.approx(2 * g.section_s + 3 * g.bar_s + 10 * g.step_s)
    assert [round(g.time(i), 2) for i in (1, 2, 6, 7, 8)] == [11.98, 23.96, 71.88, 83.86, 95.84]


def test_final_length_is_sections_plus_tail_plus_lead_in():
    cfg = ProductionConfig()
    length = HISAAB_GRID.time(len(HISAAB_SECTIONS)) + cfg.mix.tail_s + cfg.master.lead_in_s
    assert length == pytest.approx(102.96, abs=0.005)


def test_hisaab_form():
    assert [s.name for s in HISAAB_SECTIONS] == [
        "Intro", "Hook 1", "Verse 1", "Verse 2", "Verse 3", "Verse 4", "Hook 2", "Outro"
    ]  # fmt: skip
    assert [s.drums for s in HISAAB_SECTIONS] == [
        "none", "light", "full", "full", "full", "full", "full", "outro"
    ]  # fmt: skip
    assert [s.bass for s in HISAAB_SECTIONS] == [False, True, True, True, True, True, True, False]
    assert [s.strings for s in HISAAB_SECTIONS] == [False] * 6 + [True, True]
    assert [s.reverb_boost for s in HISAAB_SECTIONS] == [True] + [False] * 6 + [True]
    assert HISAAB_BAR_ROOTS == (30, 35, 37, 30)  # F#1 B1 C#2 F#1


def test_composer_form_keeps_verses_as_held_chords():
    verses = [s for s in COMPOSER_SECTIONS if s.role == "verse"]
    assert len(verses) == 4
    assert all(s.piano_texture == "held_chords" for s in verses)
    assert [s.drums for s in COMPOSER_SECTIONS] == [s.drums for s in HISAAB_SECTIONS]


def test_v3_levels_and_master_targets():
    cfg = ProductionConfig()
    assert (cfg.mix.drums_dbfs, cfg.mix.bass_dbfs, cfg.mix.piano_dbfs) == (-28.4, -15.0, -22.8)
    assert cfg.mix.strings_dbfs == -31.0
    assert (cfg.bass.second_harmonic, cfg.bass.drive) == (0.20, 2.0)
    assert (cfg.master.target_lufs, cfg.master.ceiling_dbtp) == (-10.5, -1.8)
    assert (cfg.master.clip_ratio, cfg.master.lead_in_s) == (0.85, 0.12)
    assert cfg.percussion.layers == ()


def test_reference_has_one_level_per_section_and_ten_bands():
    assert len(HISAAB_V3_REFERENCE.section_rms_dbfs) == len(HISAAB_SECTIONS)
    assert len(HISAAB_V3_REFERENCE.octave_band_db) == len(ProductionConfig().qc.octave_centres_hz)


def test_percussion_patterns():
    assert len(SHAKER.pattern) == 16
    by_step = {h.step: h for h in SHAKER.pattern}
    assert [by_step[s].gain_db for s in (0, 2, 4, 1)] == [0.0, -2.5, 0.0, -6.0]
    assert all(by_step[s].voice == s % 2 for s in range(16))  # down on even, up on odd
    assert [h.step for h in TAMBOURINE.pattern] == [4, 12]
    assert [(h.step, h.gain_db) for h in CONGAS.pattern] == [
        (2, -5.0),
        (6, 0.0),
        (8, -6.0),
        (14, -1.0),
    ]
    assert sum(v.round_robins for o in PERCUSSION_OPTIONS.values() for v in o.voices) == 48


def test_with_percussion_switches_layers_on():
    cfg = with_percussion(ProductionConfig(), (PercussionLayer("shaker"),))
    assert cfg.percussion.layers == (PercussionLayer("shaker", (2, 3, 4, 5, 6)),)
    with pytest.raises(ValueError, match="cowbell"):
        with_percussion(ProductionConfig(), (PercussionLayer("cowbell"),))


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ProductionConfig().mix.bass_dbfs = 0.0  # type: ignore[misc]
    assert Grid(beat_s=0.5).samples(1.0) == 44_100
