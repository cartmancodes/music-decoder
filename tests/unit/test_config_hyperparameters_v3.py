from pathlib import Path

from music_decoder.config.hyperparameters import load_hyperparameters

_MIN = """
id: t
basic_pitch: {onset_threshold: 0.5, frame_threshold: 0.3, minimum_note_length_ms: 58,
              minimum_frequency_hz: 60, maximum_frequency_hz: 2000}
post_processing: {median_filter_window: 5, min_note_duration_s: 0.05,
                  same_pitch_merge_gap_s: 0.05, rhythmic_snap_confidence_threshold: 0.7}
key_detection: {hpss_margin: 1.0, windowed_segment_length_s: 8, windowed_hop_s: 2}
beat_tracking: {start_bpm: 120, tightness: 100}
chord_detection: {qualities: [maj], min_segment_duration_s: 0.4}
tab_assignment: {weights: {w_move: 1.0}, max_fret: 22}
"""


def test_old_yaml_gets_v2_defaults(tmp_path: Path) -> None:
    p = tmp_path / "hp.yaml"
    p.write_text(_MIN)
    hp = load_hyperparameters(p)
    assert hp.key_detection.backend == "profile"
    assert hp.beat_tracking.backend == "librosa"
    assert hp.basic_pitch.melodia_trick is True
    assert hp.post_processing.enabled is False


def test_new_keys_are_read(tmp_path: Path) -> None:
    p = tmp_path / "hp.yaml"
    p.write_text(
        _MIN.replace("hpss_margin: 1.0", "hpss_margin: 1.0, backend: cnn")
        .replace("tightness: 100", "tightness: 100, backend: madmom")
        .replace("maximum_frequency_hz: 2000", "maximum_frequency_hz: 2000, melodia_trick: false")
        .replace("median_filter_window: 5", "median_filter_window: 5, enabled: true")
    )
    hp = load_hyperparameters(p)
    assert hp.key_detection.backend == "cnn"
    assert hp.beat_tracking.backend == "madmom"
    assert hp.basic_pitch.melodia_trick is False
    assert hp.post_processing.enabled is True


def test_env_var_overrides_default_path(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "hp.yaml"
    p.write_text(_MIN.replace("id: t", "id: from-env"))
    monkeypatch.setenv("MUSIC_DECODER_HYPERPARAMETERS", str(p))
    assert load_hyperparameters().id == "from-env"


def test_analyze_metadata_id_follows_loaded_hyperparameters(
    tmp_path: Path, monkeypatch  # type: ignore[no-untyped-def]
) -> None:
    from music_decoder.api import _hp_id

    p = tmp_path / "hp.yaml"
    p.write_text(_MIN.replace("id: t", "id: stamped"))
    monkeypatch.setenv("MUSIC_DECODER_HYPERPARAMETERS", str(p))
    monkeypatch.chdir(tmp_path)  # must not depend on the working directory
    assert _hp_id() == "stamped"
