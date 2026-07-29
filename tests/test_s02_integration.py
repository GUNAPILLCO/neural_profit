"""Pruebas de integracion de S02 v2 sobre data/02_intraday/mnq_intraday_v2.parquet
real (artefacto aprobado de S01 v2).

Solo lectura sobre las entradas; toda escritura de artefactos usa tmp_path,
nunca data/02_intraday/ productivo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import s02_intraday_analysis as s02

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "s02_analysis_config.yaml"
PRODUCTIVE_INTRADAY_DIR = PROJECT_ROOT / "data" / "02_intraday"


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("s02_integration_output")
    return s02.run_s02_analysis(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
        force_rebuild=True,
    )


def test_never_writes_to_productive_intraday_dir_besides_preexisting(result):
    assert PRODUCTIVE_INTRADAY_DIR not in result.output_dir.parents
    assert result.output_dir != PRODUCTIVE_INTRADAY_DIR


def test_all_artifacts_written(result):
    for filename in result.manifest["output_files"].values():
        assert Path(filename["path"]).exists()
    assert result.manifest_path.exists()


def test_validation_passes_on_real_v2_dataset(result):
    assert result.validation["ok"] is True
    assert result.validation["failed_critical_checks"] == []


def test_full_day_eligible_reproduces_historical_v1_shape(result):
    """Verificacion cruzada exacta con S01 v1 / S02 historico: el subconjunto
    full_day_eligible debe reproducir 1.482 dias x 691 barras = 1.024.062."""
    n_rows = result.manifest["population_row_counts"][s02.POPULATION_FULL_DAY]
    assert n_rows == 1_024_062


def test_population_row_counts_sum_consistently(result):
    counts = result.manifest["population_row_counts"]
    assert counts["all"] == 1_087_777
    # full_day_eligible + partial_regime_eligible + descriptive_only +
    # not_model_eligible (implicito, no listado) deben ser <= all.
    quantitative_and_descriptive = (
        counts[s02.POPULATION_FULL_DAY] + counts[s02.POPULATION_PARTIAL_REGIME] + counts[s02.POPULATION_DESCRIPTIVE]
    )
    assert quantitative_and_descriptive < counts["all"]


def test_body_signed_and_abs_pts_present_and_unambiguous(result):
    df = result.df_ohlcv_by_population[s02.POPULATION_FULL_DAY]
    assert "body_signed_pts" in df.columns
    assert "body_abs_pts" in df.columns
    assert "body_pts" not in df.columns
    assert (df["body_abs_pts"] >= 0).all()


def test_no_window_leakage_across_date_boundaries(result):
    df = result.df_ohlcv_by_population[s02.POPULATION_FULL_DAY]
    # Ultima barra de cada dia (minute_of_day == 960) nunca puede tener
    # ventana futura valida (cruzaria al dia siguiente).
    last_bar_mask = df["minute_of_day"] == 960
    for h in (30, 60, 90):
        assert not df.loc[last_bar_mask, f"future_window_valid_{h}m"].any()
    # Primera barra de cada dia (minute_of_day == 270) nunca puede tener
    # ventana historica valida.
    first_bar_mask = df["minute_of_day"] == 270
    for h in (30, 60, 90):
        assert not df.loc[first_bar_mask, f"hist_window_valid_{h}m"].any()


def test_window_validity_summary_matches_documented_historical_counts(result):
    """Sobre full_day_eligible (identico al dataset historico v1), la validez
    de ventana no deberia diferir sustancialmente de los conteos documentados
    en S02_intraday_data_analysis_CONTEXT.md (30m: 981.084 hist validas;
    60m: 936.624; 90m: 892.164), salvo por el bloqueo adicional de contrato
    unico (nuevo en v2), que solo puede reducir, nunca aumentar, la validez."""
    wv = result.tables["s02_window_validity_summary"]
    sub = wv.loc[(wv["population"] == s02.POPULATION_FULL_DAY) & (wv["window_type"] == "historical")]
    expected_max = {30: 981_084, 60: 936_624, 90: 892_164}
    for h, expected in expected_max.items():
        valid_rows = int(sub.loc[sub["horizon_minutes"] == h, "valid_rows"].iloc[0])
        assert valid_rows <= expected
        # La perdida adicional por bloqueo de contrato debe ser marginal
        # (los rollovers de contrato ocurren casi siempre fuera de sesion).
        assert valid_rows >= expected - 5000


def test_rollover_window_audit_only_flags_intra_segment_transitions_with_impact(result):
    audit = result.tables["s02_rollover_window_audit"]
    intra_segment = audit.loc[audit["intra_segment_transition"]]
    inter_segment = audit.loc[~audit["intra_segment_transition"]]
    # Transiciones fuera de sesion (la inmensa mayoria) no deben aportar
    # ningun bar invalidado por contrato -- ya las invalida el propio
    # cambio de segmento.
    assert (inter_segment["n_bars_hist_invalidated_by_contract"] == 0).all()
    assert (inter_segment["n_bars_future_invalidated_by_contract"] == 0).all()
    # Reporta cuantas transiciones intra-segmento existen realmente (puede
    # ser 0; se deja como informacion, no como aserto rigido, dado que
    # depende de datos reales de rollover del proveedor).
    print(f"transiciones intra-segmento con impacto de ventana: {len(intra_segment)} de {len(audit)} totales")


def test_manifest_schema_and_staleness_reproducible(result):
    manifest = result.manifest
    assert manifest["pipeline_version"] == "s02_v2"
    assert "staleness" in manifest
    assert manifest["staleness"]["intraday_parquet_sha256"]
    assert manifest["staleness"]["trading_day_audit_sha256"]
    # El fixture usa force_rebuild=True (por diseno, nunca reutiliza en
    # pruebas); staleness_fields_match siempre devuelve False si
    # force_rebuild=True, por eso se compara aqui con una copia que lo
    # desactiva, para verificar unicamente la logica de hashes.
    non_forced = {**manifest, "staleness": {**manifest["staleness"], "force_rebuild": False}}
    assert s02.staleness_fields_match(non_forced, non_forced)
    assert not s02.staleness_fields_match(manifest, manifest)


def test_no_ohlcv_metric_uses_ambiguous_body_pts_name(result):
    for table_name, table in result.tables.items():
        assert "body_pts" not in table.columns, f"{table_name} usa el nombre ambiguo body_pts"


def test_acf_and_dependence_tables_reference_only_full_day_eligible(result):
    acf = result.tables["s02_acf_summary"]
    dep = result.tables["s02_dependence_tests_summary"]
    assert set(acf["population"].unique()) <= {s02.POPULATION_FULL_DAY}
    assert set(dep["population"].unique()) <= {s02.POPULATION_FULL_DAY}
    assert not acf.empty
    assert not dep.empty
