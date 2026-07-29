"""Pruebas unitarias rapidas de S00 v2 -- datos sinteticos en memoria.

No procesan el corpus real de 26 archivos (eso vive en
test_s00_integration.py, marcado @pytest.mark.integration).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import s00_raw_ingestion as ing


def make_config() -> dict:
    return {
        "pipeline_version": "s00_v2_test",
        "source": {
            "path": "00_source",
            "filename_regex": r"^(\d{2})_mnq_(03|06|09|12)_(\d{2})\.Last\.txt$",
            "expected_count": 2,
            "delimiter": ";",
            "columns": ["datetime", "open", "high", "low", "close", "volume"],
            "timestamp_format": "%Y%m%d %H%M%S",
            "contract_month_map": {"03": "H", "06": "M", "09": "U", "12": "Z"},
            "instrument": "MNQ",
        },
        "artifacts": {
            "raw_dir": "01_raw",
            "parquet_name": "mnq_raw_v2.parquet",
            "summary_name": "mnq_raw_v2_summary.json",
            "manifest_name": "mnq_raw_v2_manifest.json",
            "gaps_name": "mnq_raw_v2_gaps.parquet",
            "source_manifest_csv": "manifests/s00_source_manifest.csv",
        },
        "timezone": {
            "timezone_stored": None,
            "timezone_assumption": "UTC",
            "timezone_evidence": "inferred_not_confirmed",
            "timestamp_semantics": "unknown_not_confirmed",
        },
        "bar": {
            "bar_interval": "1_minute",
            "price_type": "Last",
            "price_type_evidence": "inferred_from_filename",
        },
        "gaps": {"extraordinary_threshold_hours": 70},
        "force_rebuild": False,
    }


def write_file(dir_: Path, name: str, lines: list[str]) -> Path:
    p = dir_ / name
    p.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
    return p


def make_two_valid_files(source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    write_file(source_dir, "00_mnq_03_20.Last.txt", [
        "20200101 030100;100.0;100.5;99.5;100.25;10",
        "20200101 030200;100.25;100.75;100.0;100.5;12",
    ])
    write_file(source_dir, "01_mnq_06_20.Last.txt", [
        "20200401 030100;110.0;110.5;109.5;110.25;20",
        "20200401 030200;110.25;110.75;110.0;110.5;22",
    ])


# ---------------------------------------------------------------------------
# Validacion de nombre de archivo
# ---------------------------------------------------------------------------

def test_filename_regex_accepts_valid_names(tmp_path):
    config = make_config()
    source_dir = tmp_path / "00_source"
    make_two_valid_files(source_dir)

    infos = ing.validate_source_filenames(source_dir, config)

    assert [i.filename for i in infos] == ["00_mnq_03_20.Last.txt", "01_mnq_06_20.Last.txt"]
    assert [i.order for i in infos] == [0, 1]


def test_filename_regex_rejects_unexpected_file(tmp_path):
    config = make_config()
    source_dir = tmp_path / "00_source"
    make_two_valid_files(source_dir)
    (source_dir / "notes.txt").write_text("no calza el patron", encoding="ascii")

    with pytest.raises(ing.IngestionError, match="inesperados"):
        ing.validate_source_filenames(source_dir, config)


def test_order_sequence_must_be_consecutive_from_zero(tmp_path):
    config = make_config()
    config["source"]["expected_count"] = 2
    source_dir = tmp_path / "00_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    write_file(source_dir, "00_mnq_03_20.Last.txt", ["20200101 030100;1;1;1;1;1"])
    write_file(source_dir, "02_mnq_06_20.Last.txt", ["20200401 030100;1;1;1;1;1"])  # hueco: falta 01

    with pytest.raises(ing.IngestionError, match="no consecutiva"):
        ing.validate_source_filenames(source_dir, config)


# ---------------------------------------------------------------------------
# Extraccion unificada de instrument/contract/contract_full
# ---------------------------------------------------------------------------

def test_contract_extraction_unified(tmp_path):
    config = make_config()
    source_dir = tmp_path / "00_source"
    make_two_valid_files(source_dir)

    infos = ing.validate_source_filenames(source_dir, config)

    first = infos[0]
    assert first.instrument == "MNQ"
    assert first.contract == "H20"
    assert first.contract_full == "MNQH20"

    second = infos[1]
    assert second.contract == "M20"
    assert second.contract_full == "MNQM20"


# ---------------------------------------------------------------------------
# Esquema / parseo -- fail-fast, sin descarte silencioso
# ---------------------------------------------------------------------------

def test_schema_rejection_wrong_field_count(tmp_path):
    config = make_config()
    source_dir = tmp_path / "00_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    write_file(source_dir, "00_mnq_03_20.Last.txt", [
        "20200101 030100;100.0;100.5;99.5;100.25",  # falta volume
    ])
    infos = ing.validate_source_filenames(
        source_dir, {**config, "source": {**config["source"], "expected_count": 1}}
    )

    with pytest.raises(ing.IngestionError, match="esquema inv"):
        ing.parse_source_file(infos[0], config)


def test_timestamp_parse_invalid(tmp_path):
    config = make_config()
    source_dir = tmp_path / "00_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    write_file(source_dir, "00_mnq_03_20.Last.txt", [
        "2020-01-01 03:01:00;100.0;100.5;99.5;100.25;10",
    ])
    infos = ing.validate_source_filenames(
        source_dir, {**config, "source": {**config["source"], "expected_count": 1}}
    )

    with pytest.raises(ing.IngestionError, match="timestamp no parseable"):
        ing.parse_source_file(infos[0], config)


def test_ohlc_invariant_violation(tmp_path):
    config = make_config()
    source_dir = tmp_path / "00_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    # high < open: viola el invariante
    write_file(source_dir, "00_mnq_03_20.Last.txt", [
        "20200101 030100;100.0;90.0;80.0;95.0;10",
    ])
    infos = ing.validate_source_filenames(
        source_dir, {**config, "source": {**config["source"], "expected_count": 1}}
    )

    with pytest.raises(ing.IngestionError, match="invariante OHLC"):
        ing.parse_source_file(infos[0], config)


def test_negative_volume_raises(tmp_path):
    config = make_config()
    source_dir = tmp_path / "00_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    write_file(source_dir, "00_mnq_03_20.Last.txt", [
        "20200101 030100;100.0;100.5;99.5;100.25;-5",
    ])
    infos = ing.validate_source_filenames(
        source_dir, {**config, "source": {**config["source"], "expected_count": 1}}
    )

    with pytest.raises(ing.IngestionError, match="volumen negativo"):
        ing.parse_source_file(infos[0], config)


def test_empty_file_raises(tmp_path):
    config = make_config()
    source_dir = tmp_path / "00_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    write_file(source_dir, "00_mnq_03_20.Last.txt", [])
    infos = ing.validate_source_filenames(
        source_dir, {**config, "source": {**config["source"], "expected_count": 1}}
    )

    with pytest.raises(ing.IngestionError, match="vac"):
        ing.parse_source_file(infos[0], config)


# ---------------------------------------------------------------------------
# Duplicados y orden tras concatenar
# ---------------------------------------------------------------------------

def test_duplicate_timestamp_and_contract_raises(tmp_path):
    # Dos filas identicas (mismo timestamp, mismo contrato, mismo OHLCV):
    # pandas trata timestamps iguales consecutivos como monotonic_increasing,
    # asi que la deteccion real ocurre en concatenate_and_validate (duplicado
    # exacto de fila), no en el parseo por archivo.
    config = make_config()
    source_dir = tmp_path / "00_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    write_file(source_dir, "00_mnq_03_20.Last.txt", [
        "20200101 030100;100.0;100.5;99.5;100.25;10",
        "20200101 030100;100.0;100.5;99.5;100.25;10",
    ])
    infos = ing.validate_source_filenames(
        source_dir, {**config, "source": {**config["source"], "expected_count": 1}}
    )
    df = ing.parse_source_file(infos[0], config)

    with pytest.raises(ing.IngestionError, match="duplicad"):
        ing.concatenate_and_validate([df])


def test_duplicate_timestamp_different_ohlcv_caught_as_timestamp_contract_dup(tmp_path):
    # Mismo timestamp y contrato pero OHLCV distinto: no es duplicado exacto
    # de fila, pero SI es un conflicto de (timestamp, contract) -- debe
    # detenerse igual, no promediarse ni quedarse con la primera silenciosamente.
    config = make_config()
    source_dir = tmp_path / "00_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    write_file(source_dir, "00_mnq_03_20.Last.txt", [
        "20200101 030100;100.0;100.5;99.5;100.25;10",
        "20200101 030100;101.0;101.5;100.5;101.25;11",
    ])
    infos = ing.validate_source_filenames(
        source_dir, {**config, "source": {**config["source"], "expected_count": 1}}
    )
    df = ing.parse_source_file(infos[0], config)

    with pytest.raises(ing.IngestionError, match="timestamp, contract"):
        ing.concatenate_and_validate([df])


def test_same_timestamp_different_contract_is_not_confused_with_duplicate(tmp_path):
    # Mismo instante, contratos distintos (transicion de roll): valido, no es
    # un duplicado por (timestamp, contract).
    config = make_config()
    source_dir = tmp_path / "00_source"
    make_two_valid_files(source_dir)
    write_file(source_dir, "00_mnq_03_20.Last.txt", [
        "20200401 030100;100.0;100.5;99.5;100.25;10",
    ])
    infos = ing.validate_source_filenames(source_dir, config)
    dfs = [ing.parse_source_file(i, config) for i in infos]

    df = ing.concatenate_and_validate(dfs)
    assert df.index.is_monotonic_increasing


def test_chronological_order_after_concat_and_sort(tmp_path):
    config = make_config()
    source_dir = tmp_path / "00_source"
    make_two_valid_files(source_dir)
    infos = ing.validate_source_filenames(source_dir, config)
    dfs = [ing.parse_source_file(i, config) for i in infos]

    df = ing.concatenate_and_validate(dfs)

    assert list(df.index) == sorted(df.index)
    assert df["contract"].iloc[0] == "H20"
    assert df["contract"].iloc[-1] == "M20"


# ---------------------------------------------------------------------------
# Clasificacion estructural de gaps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seconds,expected_bucket", [
    (5 * 60, ing.BUCKET_2_9_MIN),
    (30 * 60, ing.BUCKET_10_70_MIN),
    (10 * 3600, ing.BUCKET_70MIN_100H),
    (150 * 3600, ing.BUCKET_GT_100H),
])
def test_structural_bucket_classification(seconds, expected_bucket):
    assert ing._structural_bucket(seconds) == expected_bucket


def test_gaps_never_confirmed_evidence_level(tmp_path):
    config = make_config()
    source_dir = tmp_path / "00_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    write_file(source_dir, "00_mnq_03_20.Last.txt", [
        "20200101 030100;100.0;100.5;99.5;100.25;10",
        "20200101 040100;100.0;100.5;99.5;100.25;10",  # gap de 1h -> 10-70min bucket
    ])
    infos = ing.validate_source_filenames(
        source_dir, {**config, "source": {**config["source"], "expected_count": 1}}
    )
    df = ing.concatenate_and_validate([ing.parse_source_file(infos[0], config)])

    gaps_df = ing.compute_gaps(df)

    assert len(gaps_df) == 1
    assert gaps_df.iloc[0]["gap_type_structural"] == "intra_file"
    assert "confirmed" not in gaps_df.iloc[0]["evidence_level"]
    assert "PROVISIONAL" in gaps_df.iloc[0]["provisional_interpretation_utc_hypothesis"]


def test_inter_contract_gap_detected_at_transition(tmp_path):
    config = make_config()
    source_dir = tmp_path / "00_source"
    make_two_valid_files(source_dir)
    infos = ing.validate_source_filenames(source_dir, config)
    dfs = [ing.parse_source_file(i, config) for i in infos]
    df = ing.concatenate_and_validate(dfs)

    gaps_df = ing.compute_gaps(df)

    inter = gaps_df[gaps_df["gap_type_structural"] == "inter_contract"]
    assert len(inter) == 1
    assert inter.iloc[0]["source_file_left"] == "00_mnq_03_20.Last.txt"
    assert inter.iloc[0]["source_file_right"] == "01_mnq_06_20.Last.txt"


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

def test_normalized_config_hash_stable_across_formatting():
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert ing.sha256_bytes(ing.normalized_config_bytes(a)) == ing.sha256_bytes(ing.normalized_config_bytes(b))


def test_staleness_fields_match_true_when_identical():
    manifest = {"staleness": {
        "source_files_sha256": {"a": "h1"}, "module_sha256": "m1",
        "config_sha256_normalized": "c1", "schema_expected": ["x"],
        "pipeline_version": "v1", "force_rebuild": False,
    }}
    assert ing.staleness_fields_match(manifest, manifest) is True


def test_staleness_fields_match_false_when_module_hash_changes():
    old = {"staleness": {
        "source_files_sha256": {"a": "h1"}, "module_sha256": "m1",
        "config_sha256_normalized": "c1", "schema_expected": ["x"],
        "pipeline_version": "v1", "force_rebuild": False,
    }}
    new = {"staleness": {**old["staleness"], "module_sha256": "m2"}}
    assert ing.staleness_fields_match(old, new) is False


def test_staleness_fields_match_false_when_force_rebuild_true():
    old = {"staleness": {
        "source_files_sha256": {"a": "h1"}, "module_sha256": "m1",
        "config_sha256_normalized": "c1", "schema_expected": ["x"],
        "pipeline_version": "v1", "force_rebuild": False,
    }}
    new = {"staleness": {**old["staleness"], "force_rebuild": True}}
    assert ing.staleness_fields_match(old, new) is False


def test_staleness_fields_match_false_when_no_previous_manifest():
    new = {"staleness": {
        "source_files_sha256": {}, "module_sha256": "m1",
        "config_sha256_normalized": "c1", "schema_expected": ["x"],
        "pipeline_version": "v1", "force_rebuild": False,
    }}
    assert ing.staleness_fields_match(None, new) is False


def test_git_provenance_does_not_affect_staleness():
    # git_commit/git_dirty no forman parte de staleness.staleness_fields_match
    old = {"staleness": {
        "source_files_sha256": {"a": "h1"}, "module_sha256": "m1",
        "config_sha256_normalized": "c1", "schema_expected": ["x"],
        "pipeline_version": "v1", "force_rebuild": False,
    }, "provenance_metadata_only": {"git_commit": "aaa", "git_dirty": False}}
    new = {"staleness": old["staleness"],
           "provenance_metadata_only": {"git_commit": "bbb", "git_dirty": True}}
    assert ing.staleness_fields_match(old, new) is True
