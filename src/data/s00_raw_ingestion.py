"""S00 v2 — ingestión cruda de los históricos trimestrales MNQ.

Lee los archivos fuente en `data/00_source/`, valida su nombre, esquema,
parseo, invariantes OHLCV y ausencia de duplicados, concatena y ordena
cronológicamente, cataloga los gaps encontrados (sin clasificarlos por
causa) y persiste el resultado de forma atómica junto con un manifiesto
autoritativo y un summary.

No aplica ninguna transformación temporal (sin tz_localize, tz_convert,
filtrado de sesión ni calendario) — eso pertenece a S01.

No tiene autoridad para descartar, corregir ni rellenar filas: cualquier
fila que falle una validación detiene la generación del artefacto.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

MODULE_PATH = Path(__file__).resolve()

REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]

# Buckets estructurales por duración (§14 del plan). "~1min_step" es el paso
# normal entre barras consecutivas y nunca se materializa como fila en el
# catálogo de gaps -- solo se registran diffs != 60s.
BUCKET_2_9_MIN = "2-9min"
BUCKET_10_70_MIN = "10-70min"
BUCKET_70MIN_100H = "70min-100h"
BUCKET_GT_100H = ">100h"

GAP_EVIDENCE_STRUCTURAL_ONLY = "structural_only"
GAP_EVIDENCE_PROVISIONAL_PATTERN = "provisional_pattern_match"
GAP_EVIDENCE_UNCONFIRMED = "unconfirmed"


class IngestionError(Exception):
    """Error que debe detener la generación del artefacto (no se descarta la fila)."""


@dataclass
class RowFailure:
    file: str
    line_number: int
    raw_line: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line_number": self.line_number,
            "raw_line": self.raw_line,
            "reason": self.reason,
        }


@dataclass
class SourceFileInfo:
    order: int
    filename: str
    path: Path
    instrument: str
    contract: str          # formato corto vigente, ej. "H20"
    contract_full: str      # ej. "MNQH20", solo informativo/manifiesto
    month_code: str
    year_short: str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config:
        raise IngestionError(f"Config vacía o ilegible: {config_path}")
    return config


def normalized_config_bytes(config: dict[str, Any]) -> bytes:
    """Serialización determinística para hashear la config (ignora formato/espacios)."""
    return json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Hashing / procedencia
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_module_hash() -> str:
    return sha256_file(MODULE_PATH)


def get_git_provenance(repo_root: Path) -> dict[str, Any]:
    """Best-effort: commit actual y flag de working tree sucio. Metadata de
    procedencia únicamente -- no participa en la decisión de staleness."""
    commit = None
    dirty = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            dirty = len(result.stdout.strip()) > 0
    except (OSError, subprocess.SubprocessError):
        pass
    return {"git_commit": commit, "git_dirty": dirty}


# ---------------------------------------------------------------------------
# Validación de nombres de archivo fuente
# ---------------------------------------------------------------------------

def validate_source_filenames(source_dir: Path, config: dict[str, Any]) -> list[SourceFileInfo]:
    src_cfg = config["source"]
    pattern = re.compile(src_cfg["filename_regex"])
    month_map = src_cfg["contract_month_map"]
    instrument = src_cfg.get("instrument", "MNQ")
    expected_count = src_cfg["expected_count"]

    entries = sorted(p for p in source_dir.iterdir() if p.is_file())

    unexpected = [p.name for p in entries if not pattern.match(p.name)]
    if unexpected:
        raise IngestionError(
            f"Archivos inesperados en {source_dir} que no calzan filename_regex: {unexpected}"
        )

    infos: list[SourceFileInfo] = []
    for p in entries:
        m = pattern.match(p.name)
        order_str, month, year_short = m.group(1), m.group(2), m.group(3)
        order = int(order_str)
        if month not in month_map:
            raise IngestionError(f"Mes sin mapeo H/M/U/Z en {p.name}: {month}")
        contract_code = month_map[month]
        contract = f"{contract_code}{year_short}"
        contract_full = f"{instrument}{contract_code}{year_short}"
        infos.append(SourceFileInfo(
            order=order, filename=p.name, path=p,
            instrument=instrument, contract=contract, contract_full=contract_full,
            month_code=contract_code, year_short=year_short,
        ))

    if len(infos) != expected_count:
        raise IngestionError(
            f"Se esperaban {expected_count} archivos fuente, se encontraron {len(infos)}"
        )

    infos.sort(key=lambda i: i.order)
    orders = [i.order for i in infos]
    expected_orders = list(range(expected_count))
    if orders != expected_orders:
        raise IngestionError(
            f"Secuencia de <orden> no consecutiva desde 00: encontrada {orders}, esperada {expected_orders}"
        )

    return infos


# ---------------------------------------------------------------------------
# Parseo y validación fila a fila (fail-fast, sin descartes silenciosos)
# ---------------------------------------------------------------------------

def _validate_ohlcv_row(o: float, h: float, low: float, c: float, v: int) -> str | None:
    """Devuelve la razón de fallo, o None si la fila es válida."""
    for name, val in (("open", o), ("high", h), ("low", low), ("close", c)):
        if val != val or val in (float("inf"), float("-inf")):
            return f"{name} es NaN/Inf"
    if o <= 0 or h <= 0 or low <= 0 or c <= 0:
        return "precio no positivo"
    if v < 0:
        return "volumen negativo"
    if not float(v).is_integer():
        return "volumen no entero"
    if not (h >= o and h >= c and low <= o and low <= c and h >= low):
        return "invariante OHLC violado (high>=open,close,low; low<=open,close)"
    return None


def parse_source_file(info: SourceFileInfo, config: dict[str, Any]) -> pd.DataFrame:
    """Lee un archivo fuente línea a línea. Ante la primera fila inválida,
    detiene el pipeline (IngestionError) con evidencia de archivo/línea."""
    src_cfg = config["source"]
    delimiter = src_cfg["delimiter"]
    ts_format = src_cfg["timestamp_format"]
    n_expected_fields = len(src_cfg["columns"])

    rows: list[tuple] = []
    with open(info.path, "r", encoding="ascii", errors="strict") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split(delimiter)
            if len(parts) != n_expected_fields:
                raise IngestionError(str(RowFailure(
                    file=info.filename, line_number=line_no, raw_line=line,
                    reason=f"esquema inválido: {len(parts)} campos, se esperaban {n_expected_fields}",
                ).to_dict()))

            ts_raw, o_raw, h_raw, l_raw, c_raw, v_raw = parts
            try:
                ts = datetime.strptime(ts_raw, ts_format)
            except ValueError as e:
                raise IngestionError(str(RowFailure(
                    file=info.filename, line_number=line_no, raw_line=line,
                    reason=f"timestamp no parseable con '{ts_format}': {e}",
                ).to_dict())) from e

            try:
                o, h, low, c = float(o_raw), float(h_raw), float(l_raw), float(c_raw)
                v = int(v_raw)
            except ValueError as e:
                raise IngestionError(str(RowFailure(
                    file=info.filename, line_number=line_no, raw_line=line,
                    reason=f"campo numérico no parseable: {e}",
                ).to_dict())) from e

            failure_reason = _validate_ohlcv_row(o, h, low, c, v)
            if failure_reason:
                raise IngestionError(str(RowFailure(
                    file=info.filename, line_number=line_no, raw_line=line,
                    reason=failure_reason,
                ).to_dict()))

            rows.append((ts, o, h, low, c, v))

    if not rows:
        raise IngestionError(f"Archivo vacío (sin filas de datos): {info.filename}")

    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["contract"] = info.contract
    df["source_file"] = info.filename
    df.set_index("datetime", inplace=True)
    if not df.index.is_monotonic_increasing:
        raise IngestionError(
            f"Timestamps no monotónicos crecientes dentro de {info.filename}"
        )
    return df


# ---------------------------------------------------------------------------
# Consolidación y duplicados (segunda pasada, requiere el corpus completo)
# ---------------------------------------------------------------------------

def concatenate_and_validate(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(dfs)
    df.sort_index(inplace=True)

    # Se incluye el indice (timestamp) en la comparacion: dos barras con
    # OHLCV identico pero timestamps distintos NO son una fila duplicada.
    exact_dupe_mask = df.reset_index().duplicated(keep=False)
    if exact_dupe_mask.any():
        raise IngestionError(
            f"Filas exactamente duplicadas detectadas: {int(exact_dupe_mask.sum())}"
        )

    dup_ts_contract = df.reset_index().duplicated(subset=["datetime", "contract"], keep=False)
    if dup_ts_contract.any():
        raise IngestionError(
            f"Duplicados por (timestamp, contract) detectados: {int(dup_ts_contract.sum())}"
        )

    global_dup_ts = df.index.duplicated(keep=False)
    n_global_dup = int(global_dup_ts.sum())
    # Un mismo timestamp puede aparecer en dos contratos distintos en el
    # instante de un roll -- eso ya se validó arriba y no es un error por sí
    # solo. Solo abortamos si además coincide con el mismo contrato (caso ya
    # cubierto), así que aquí solo se deja constancia informativa.
    if not df.index.is_monotonic_increasing:
        raise IngestionError("El índice global no quedó monotónico creciente tras ordenar")

    return df


# ---------------------------------------------------------------------------
# Catálogo de gaps (estructural, sin clasificación causal definitiva)
# ---------------------------------------------------------------------------

def _structural_bucket(duration_seconds: float) -> str:
    minutes = duration_seconds / 60.0
    hours = duration_seconds / 3600.0
    if minutes < 10:
        return BUCKET_2_9_MIN
    if minutes <= 70:
        return BUCKET_10_70_MIN
    if hours <= 100:
        return BUCKET_70MIN_100H
    return BUCKET_GT_100H


def _provisional_utc_hypothesis(gap_type: str, duration_seconds: float) -> str:
    minutes = duration_seconds / 60.0
    hours = duration_seconds / 3600.0
    if gap_type == "intra_file" and 10 <= minutes <= 70:
        return (
            "PROVISIONAL bajo hipótesis UTC: duración compatible con un corte "
            "diario de mantenimiento tipo CME Globex. No confirmado."
        )
    if 40 <= hours <= 80:
        return (
            "PROVISIONAL bajo hipótesis UTC: duración compatible con un cierre "
            "de fin de semana / feriado de mercado. No confirmado."
        )
    if hours > 80:
        return (
            "PROVISIONAL: duración muy superior a un cierre de fin de semana "
            "típico; sin patrón estructural reconocido en S00. Requiere "
            "auditoría adicional (no resuelto por S00)."
        )
    return "PROVISIONAL: sin patrón estructural reconocido en S00."


def compute_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Recorre el DataFrame global ordenado y registra todo salto != 60s
    entre filas consecutivas. No decide causa; ver §14 del plan."""
    idx = df.index
    source_files = df["source_file"].to_numpy()
    contracts = df["contract"].to_numpy()

    records = []
    for i in range(1, len(df)):
        prev_ts, next_ts = idx[i - 1], idx[i]
        delta = (next_ts - prev_ts).total_seconds()
        if delta == 60.0:
            continue
        if delta <= 0:
            raise IngestionError(
                f"Orden temporal inválido detectado entre filas: {prev_ts} -> {next_ts}"
            )

        same_file = source_files[i - 1] == source_files[i]
        gap_type = "intra_file" if same_file else "inter_contract"

        records.append({
            "gap_type_structural": gap_type,
            "source_file_left": source_files[i - 1],
            "source_file_right": source_files[i],
            "contract_left": contracts[i - 1],
            "contract_right": contracts[i],
            "previous_timestamp": prev_ts,
            "next_timestamp": next_ts,
            "duration_seconds": delta,
            "structural_bucket": _structural_bucket(delta),
        })

    gaps_df = pd.DataFrame.from_records(records)
    if gaps_df.empty:
        gaps_df = pd.DataFrame(columns=[
            "gap_type_structural", "source_file_left", "source_file_right",
            "contract_left", "contract_right", "previous_timestamp", "next_timestamp",
            "duration_seconds", "structural_bucket", "recurrence",
            "provisional_interpretation_utc_hypothesis", "evidence_level",
        ])
        return gaps_df

    # recurrence: cuántos gaps comparten bucket estructural -- evidencia
    # puramente numérica, no interpretación causal.
    bucket_counts = gaps_df["structural_bucket"].value_counts()
    gaps_df["recurrence"] = gaps_df["structural_bucket"].map(bucket_counts)

    gaps_df["provisional_interpretation_utc_hypothesis"] = gaps_df.apply(
        lambda r: _provisional_utc_hypothesis(r["gap_type_structural"], r["duration_seconds"]),
        axis=1,
    )

    def evidence_level(row) -> str:
        bucket = row["structural_bucket"]
        recurrence = row["recurrence"]
        hours = row["duration_seconds"] / 3600.0
        if bucket == BUCKET_GT_100H:
            return GAP_EVIDENCE_UNCONFIRMED
        if bucket == BUCKET_70MIN_100H and 40 <= hours <= 80 and recurrence >= 5:
            return GAP_EVIDENCE_PROVISIONAL_PATTERN
        if bucket == BUCKET_70MIN_100H:
            return GAP_EVIDENCE_UNCONFIRMED
        return GAP_EVIDENCE_STRUCTURAL_ONLY

    gaps_df["evidence_level"] = gaps_df.apply(evidence_level, axis=1)
    return gaps_df


EXTRAORDINARY_BUCKETS = {BUCKET_70MIN_100H, BUCKET_GT_100H}


def extraordinary_gaps(gaps_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    threshold_h = config.get("gaps", {}).get("extraordinary_threshold_hours", 70)
    if gaps_df.empty:
        return gaps_df
    mask = (gaps_df["duration_seconds"] / 3600.0) >= threshold_h
    return gaps_df[mask].copy()


# ---------------------------------------------------------------------------
# Manifiesto, staleness y summary
# ---------------------------------------------------------------------------

def build_source_manifest_records(
    infos: list[SourceFileInfo],
    per_file_df: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    records = []
    for info in infos:
        stat = info.path.stat()
        df = per_file_df[info.filename]
        records.append({
            "path": str(info.path),
            "filename": info.filename,
            "size_bytes": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "sha256": sha256_file(info.path),
            "instrument": info.instrument,
            "contract": info.contract,
            "contract_full": info.contract_full,
            "n_rows": int(len(df)),
            "first_timestamp": df.index.min().isoformat(),
            "last_timestamp": df.index.max().isoformat(),
            "errors": [],
        })
    return records


def build_manifest(
    *,
    config: dict[str, Any],
    config_path: Path,
    source_records: list[dict[str, Any]],
    gaps_df: pd.DataFrame,
    extraordinary_df: pd.DataFrame,
    df: pd.DataFrame,
    repo_root: Path,
) -> dict[str, Any]:
    provenance = get_git_provenance(repo_root)

    gaps_agg: dict[str, Any] = {"total_gaps": int(len(gaps_df))}
    if not gaps_df.empty:
        gaps_agg["by_structural_bucket"] = gaps_df["structural_bucket"].value_counts().to_dict()
        gaps_agg["by_evidence_level"] = gaps_df["evidence_level"].value_counts().to_dict()
    else:
        gaps_agg["by_structural_bucket"] = {}
        gaps_agg["by_evidence_level"] = {}
    gaps_agg["extraordinary_cases"] = json.loads(
        extraordinary_df.assign(
            previous_timestamp=lambda d: d["previous_timestamp"].astype(str),
            next_timestamp=lambda d: d["next_timestamp"].astype(str),
        ).to_json(orient="records")
    ) if not extraordinary_df.empty else []

    manifest = {
        "pipeline_version": config["pipeline_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "staleness": {
            "source_files_sha256": {r["filename"]: r["sha256"] for r in source_records},
            "module_sha256": get_module_hash(),
            "config_sha256_normalized": sha256_bytes(normalized_config_bytes(config)),
            "schema_expected": config["source"]["columns"] + ["contract"],
            "pipeline_version": config["pipeline_version"],
            "force_rebuild": bool(config.get("force_rebuild", False)),
        },
        "provenance_metadata_only": {
            # Trazabilidad; NO participan en la decisión de staleness.
            "git_commit": provenance["git_commit"],
            "git_dirty": provenance["git_dirty"],
        },
        "timezone": config["timezone"],
        "bar": config["bar"],
        "sources": source_records,
        "dataset": {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "first_timestamp": df.index.min().isoformat(),
            "last_timestamp": df.index.max().isoformat(),
            "index_tz": "tz-naive",
        },
        "gaps": gaps_agg,
    }
    return manifest


def staleness_fields_match(old: dict[str, Any] | None, new_manifest: dict[str, Any]) -> bool:
    """True si el artefacto existente sigue siendo válido (no stale)."""
    if old is None:
        return False
    old_st = old.get("staleness", {})
    new_st = new_manifest["staleness"]
    if new_st["force_rebuild"]:
        return False
    keys = [
        "source_files_sha256", "module_sha256", "config_sha256_normalized",
        "schema_expected", "pipeline_version",
    ]
    return all(old_st.get(k) == new_st.get(k) for k in keys)


def build_summary(manifest: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Summary honesto: no localiza el índice a UTC para el reporte ni sugiere
    tz-aware. Solo agregaciones de gaps, nunca el detalle fila a fila."""
    return {
        "name": "mnq_raw_v2",
        "pipeline_version": manifest["pipeline_version"],
        "shape": [int(len(df)), int(df.shape[1])],
        "columns": list(df.columns),
        "index_type": "DatetimeIndex",
        "index_tz": "tz-naive (sin zona horaria en el índice persistido)",
        "timezone_assumption": manifest["timezone"]["timezone_assumption"],
        "timezone_evidence": manifest["timezone"]["timezone_evidence"],
        "timestamp_semantics": manifest["timezone"]["timestamp_semantics"],
        "bar_interval": manifest["bar"]["bar_interval"],
        "price_type": manifest["bar"]["price_type"],
        "price_type_evidence": manifest["bar"]["price_type_evidence"],
        "datetime_min": df.index.min().isoformat(),
        "datetime_max": df.index.max().isoformat(),
        "n_sources": len(manifest["sources"]),
        "gaps_summary": manifest["gaps"],
        "note": (
            "Este summary NO afirma que el índice sea UTC confirmado. "
            "timezone_assumption es una suposición heredada, no verificada "
            "documentalmente. El catálogo completo de gaps vive en "
            "mnq_raw_v2_gaps.parquet, no en este archivo."
        ),
    }


# ---------------------------------------------------------------------------
# Escritura atómica + relectura
# ---------------------------------------------------------------------------

def _dataframes_logically_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a.shape != b.shape:
        return False
    if list(a.columns) != list(b.columns):
        return False
    if not a.index.equals(b.index):
        return False
    for col in a.columns:
        if not a[col].equals(b[col]):
            return False
    return True


def atomic_write_parquet(df: pd.DataFrame, final_path: Path) -> dict[str, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    df.to_parquet(tmp_path, index=True)
    reread = pd.read_parquet(tmp_path)

    equal = _dataframes_logically_equal(df, reread)
    if not equal:
        tmp_path.unlink(missing_ok=True)
        raise IngestionError(
            f"La relectura de {tmp_path} no es equivalente al DataFrame en memoria; "
            "escritura abortada, no se movió al nombre final."
        )

    tmp_path.replace(final_path)
    return {"sha256": sha256_file(final_path), "verified_equivalent": True}


def write_json_atomic(obj: dict[str, Any], final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    tmp_path.replace(final_path)


def write_source_manifest_csv(source_records: list[dict[str, Any]], path: Path) -> None:
    """Vista tabular DERIVADA del manifest.json -- no autoritativa, el
    pipeline nunca la lee para decidir staleness."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [{k: v for k, v in r.items() if k != "errors"} for r in source_records]
    pd.DataFrame(flat).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

@dataclass
class S00Result:
    reused_existing: bool
    df: pd.DataFrame
    gaps_df: pd.DataFrame
    manifest: dict[str, Any]
    summary: dict[str, Any]
    parquet_path: Path
    parquet_sha256: str
    manifest_path: Path
    summary_path: Path
    gaps_path: Path
    source_manifest_csv_path: Path
    source_records: list[dict[str, Any]]
    extraordinary_gaps: list[dict[str, Any]]


def run_s00_ingestion(
    project_root: Path,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    force_rebuild: bool | None = None,
) -> S00Result:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path) if config_path else project_root / "config" / "data_config.yaml"
    config = load_config(config_path)
    if force_rebuild is not None:
        config["force_rebuild"] = force_rebuild

    source_dir = project_root / config["source"]["path"]
    artifacts_dir = Path(output_dir) if output_dir else project_root / config["artifacts"]["raw_dir"]

    parquet_path = artifacts_dir / config["artifacts"]["parquet_name"]
    summary_path = artifacts_dir / config["artifacts"]["summary_name"]
    manifest_path = artifacts_dir / config["artifacts"]["manifest_name"]
    gaps_path = artifacts_dir / config["artifacts"]["gaps_name"]
    source_manifest_csv_path = (
        Path(output_dir) / "s00_source_manifest.csv" if output_dir
        else project_root / config["artifacts"]["source_manifest_csv"]
    )

    infos = validate_source_filenames(source_dir, config)

    per_file_df: dict[str, pd.DataFrame] = {}
    parsed_dfs: list[pd.DataFrame] = []
    for info in infos:
        file_df = parse_source_file(info, config)
        per_file_df[info.filename] = file_df
        parsed_dfs.append(file_df)

    source_records = build_source_manifest_records(infos, per_file_df)

    df_full = concatenate_and_validate(parsed_dfs)
    gaps_df = compute_gaps(df_full)
    extraordinary_df = extraordinary_gaps(gaps_df, config)

    # El dataset persistido conserva exactamente el schema histórico que
    # espera S01: open, high, low, close, volume, contract. `source_file`
    # es un campo de trabajo interno para el cálculo de gaps, no se persiste.
    df = df_full.drop(columns=["source_file"])

    manifest = build_manifest(
        config=config, config_path=config_path, source_records=source_records,
        gaps_df=gaps_df, extraordinary_df=extraordinary_df, df=df,
        repo_root=project_root,
    )

    existing_manifest = None
    if manifest_path.exists() and parquet_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_manifest = None

    if staleness_fields_match(existing_manifest, manifest):
        reused_df = pd.read_parquet(parquet_path)
        reused_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else build_summary(existing_manifest, reused_df)
        reused_gaps = pd.read_parquet(gaps_path) if gaps_path.exists() else gaps_df
        return S00Result(
            reused_existing=True, df=reused_df, gaps_df=reused_gaps,
            manifest=existing_manifest, summary=reused_summary,
            parquet_path=parquet_path, parquet_sha256=sha256_file(parquet_path),
            manifest_path=manifest_path, summary_path=summary_path, gaps_path=gaps_path,
            source_manifest_csv_path=source_manifest_csv_path,
            source_records=existing_manifest.get("sources", source_records),
            extraordinary_gaps=existing_manifest.get("gaps", {}).get("extraordinary_cases", []),
        )

    write_result = atomic_write_parquet(df, parquet_path)
    if not gaps_df.empty:
        atomic_write_parquet(gaps_df.reset_index(drop=True), gaps_path)
    else:
        gaps_df.to_parquet(gaps_path, index=False)

    summary = build_summary(manifest, df)
    write_json_atomic(manifest, manifest_path)
    write_json_atomic(summary, summary_path)
    write_source_manifest_csv(source_records, source_manifest_csv_path)

    return S00Result(
        reused_existing=False, df=df, gaps_df=gaps_df,
        manifest=manifest, summary=summary,
        parquet_path=parquet_path, parquet_sha256=write_result["sha256"],
        manifest_path=manifest_path, summary_path=summary_path, gaps_path=gaps_path,
        source_manifest_csv_path=source_manifest_csv_path,
        source_records=source_records,
        extraordinary_gaps=manifest["gaps"]["extraordinary_cases"],
    )
