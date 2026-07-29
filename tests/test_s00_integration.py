"""Pruebas de integracion de S00 v2 sobre el corpus real de 26 archivos.

Corren contra data/00_source/ como entrada de SOLO LECTURA. Toda escritura
de artefactos usa tmp_path -- nunca tocan data/01_raw/ productivo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import s00_raw_ingestion as ing

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"
PRODUCTIVE_RAW_DIR = PROJECT_ROOT / "data" / "01_raw"

EXPECTED_N_ROWS = 2_172_640
EXPECTED_FIRST_TS = "2019-12-23 03:01:00"
EXPECTED_LAST_TS = "2026-04-17 20:18:00"


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("s00_integration_output")
    return ing.run_s00_ingestion(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
        force_rebuild=True,
    )


def test_never_writes_to_productive_raw_dir(result):
    # data/01_raw/ debe seguir vacio: nada de esta prueba escribe ahi.
    if PRODUCTIVE_RAW_DIR.exists():
        assert list(PRODUCTIVE_RAW_DIR.iterdir()) == []


def test_artifacts_written_only_under_tmp_path(result):
    assert result.parquet_path.exists()
    assert result.manifest_path.exists()
    assert result.summary_path.exists()
    assert result.gaps_path.exists()
    assert PRODUCTIVE_RAW_DIR not in result.parquet_path.parents


def test_row_count_matches_audit(result):
    assert len(result.df) == EXPECTED_N_ROWS


def test_temporal_range_matches_audit(result):
    assert str(result.df.index.min()) == EXPECTED_FIRST_TS
    assert str(result.df.index.max()) == EXPECTED_LAST_TS


def test_schema_matches_historical_contract(result):
    # S01 espera exactamente estas columnas (formato de contrato corto).
    assert list(result.df.columns) == ["open", "high", "low", "close", "volume", "contract"]


def test_manifest_has_26_source_records(result):
    assert len(result.source_records) == 26
    filenames = [r["filename"] for r in result.source_records]
    assert filenames[0] == "00_mnq_03_20.Last.txt"
    assert filenames[-1] == "25_mnq_06_26.Last.txt"


def test_atomic_write_verified_and_rereadable(result):
    reread = pd.read_parquet(result.parquet_path)
    assert reread.shape == result.df.shape
    assert list(reread.columns) == list(result.df.columns)
    assert result.parquet_sha256 and len(result.parquet_sha256) == 64


def test_gaps_parquet_contains_the_two_known_extraordinary_cases(result):
    gaps_df = pd.read_parquet(result.gaps_path)
    extraordinary = gaps_df[gaps_df["duration_seconds"] / 3600.0 >= 70]

    # Gap interno M23: ~260.25h dentro de 13_mnq_06_23.Last.txt
    m23 = extraordinary[
        (extraordinary["source_file_left"] == "13_mnq_06_23.Last.txt")
        & (extraordinary["gap_type_structural"] == "intra_file")
    ]
    assert len(m23) == 1
    assert 259 <= m23.iloc[0]["duration_seconds"] / 3600.0 <= 261

    # Gap de transicion H25 -> M25: ~15d19h12min
    h25_m25 = extraordinary[
        (extraordinary["source_file_left"] == "20_mnq_03_25.Last.txt")
        & (extraordinary["source_file_right"] == "21_mnq_06_25.Last.txt")
    ]
    assert len(h25_m25) == 1
    assert 379 <= h25_m25.iloc[0]["duration_seconds"] / 3600.0 <= 380


def test_manifest_and_summary_hold_only_aggregations_not_full_gap_list(result):
    assert "extraordinary_cases" in result.manifest["gaps"]
    assert "by_structural_bucket" in result.manifest["gaps"]
    assert len(result.manifest["gaps"]["extraordinary_cases"]) < 20
    assert "gaps_summary" in result.summary
    assert len(str(result.summary)) < 20_000  # no es plausible que contenga miles de filas


def test_second_run_reuses_when_nothing_changed(tmp_path):
    output_dir = tmp_path / "s00_reuse_check"
    first = ing.run_s00_ingestion(
        project_root=PROJECT_ROOT, config_path=CONFIG_PATH,
        output_dir=output_dir, force_rebuild=False,
    )
    second = ing.run_s00_ingestion(
        project_root=PROJECT_ROOT, config_path=CONFIG_PATH,
        output_dir=output_dir, force_rebuild=False,
    )
    assert first.reused_existing is False
    assert second.reused_existing is True
    assert second.parquet_sha256 == first.parquet_sha256
