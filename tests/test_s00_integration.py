"""Pruebas de integracion de S00 v2 sobre el corpus real de 27 archivos.

Corren contra data/00_source/ como entrada de SOLO LECTURA. Toda escritura
de artefactos usa tmp_path -- nunca tocan data/01_raw/ productivo.

Los archivos fuente se actualizaron (ver manifests/s00_source_manifest.csv):
data/00_source ahora cubre hasta 2026-07-31 y ya no contiene los gaps
extraordinarios documentados historicamente (S00-05 gap H25->M25, S00-06
gap interno M23) -- ver 02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md SS4.1-bis.
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

EXPECTED_N_ROWS = 2_329_783
EXPECTED_FIRST_TS = "2019-12-23 03:01:00"
EXPECTED_LAST_TS = "2026-07-31 20:10:00"


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
    # Esta prueba corre siempre contra tmp_path (ver fixture `result`); no
    # debe crear NINGUN archivo nuevo en data/01_raw/ productivo. Ya no se
    # asume que el directorio productivo este vacio (S00 v2 esta aprobado y
    # sus artefactos reales viven ahi) -- se verifica que el conjunto de
    # nombres presentes sea exactamente el esperado, sin archivos extra.
    if not PRODUCTIVE_RAW_DIR.exists():
        return
    expected_names = {
        "mnq_raw_v2.parquet",
        "mnq_raw_v2_summary.json",
        "mnq_raw_v2_manifest.json",
        "mnq_raw_v2_gaps.parquet",
    }
    actual_names = {p.name for p in PRODUCTIVE_RAW_DIR.iterdir()}
    assert actual_names <= expected_names, (
        f"archivos inesperados en {PRODUCTIVE_RAW_DIR}: {actual_names - expected_names}"
    )


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


def test_manifest_has_27_source_records(result):
    assert len(result.source_records) == 27
    filenames = [r["filename"] for r in result.source_records]
    assert filenames[0] == "00_mnq_03_20.Last.txt"
    assert filenames[-1] == "26_mnq_09_26.Last.txt"


def test_atomic_write_verified_and_rereadable(result):
    reread = pd.read_parquet(result.parquet_path)
    assert reread.shape == result.df.shape
    assert list(reread.columns) == list(result.df.columns)
    assert result.parquet_sha256 and len(result.parquet_sha256) == 64


def test_no_unresolved_multi_day_gaps_remain(result):
    # Los archivos fuente se actualizaron y ya no reproducen los gaps
    # extraordinarios historicos S00-05 (H25->M25, ~15d19h) ni S00-06 (M23
    # interno, ~260h): el bucket ">100h" debe estar vacio. Los gaps
    # "70min-100h" restantes son fines de semana largos / feriados
    # ordinarios (ver structural_bucket), no anomalias sin explicar.
    gaps_df = pd.read_parquet(result.gaps_path)
    over_100h = gaps_df[gaps_df["structural_bucket"] == ing.BUCKET_GT_100H]
    assert len(over_100h) == 0

    m23 = gaps_df[
        (gaps_df["source_file_left"] == "13_mnq_06_23.Last.txt")
        & (gaps_df["gap_type_structural"] == "intra_file")
        & (gaps_df["duration_seconds"] / 3600.0 >= 100)
    ]
    assert len(m23) == 0

    h25_m25_overlap = gaps_df[
        (gaps_df["source_file_left"] == "20_mnq_03_25.Last.txt")
        & (gaps_df["source_file_right"] == "21_mnq_06_25.Last.txt")
    ]
    assert len(h25_m25_overlap) == 0  # los contratos ahora se solapan, no hay gap que registrar


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
