// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Benchmarks for AggregateExec with dictionary-encoded group keys.
//!
//! Isolates the RowConverter overhead when grouping on a
//! Dictionary(Int32, Utf8) column vs plain Utf8.
//!
//! Builds physical plans by hand — no SQL, no optimizer:
//!
//!   AggregateExec (Single, GROUP BY pod, COUNT(*))
//!     DataSourceExec (ParquetSource, streams from disk)
//!
//! Reuses the parquet files written by repartition_dict.rs:
//!   benches/data/access_log_dict.parquet   — Dictionary(Int32, Utf8) columns
//!   benches/data/access_log_plain.parquet  — same data, dict columns cast to Utf8
//!
//! Run:
//!   cargo bench -p datafusion --bench aggregate_dict
//!
//! Profile with samply:
//!   Step 1 — build (once):
//!     RUSTFLAGS="-C force-frame-pointers=yes" \
//!     cargo bench -p datafusion --bench aggregate_dict \
//!       --profile profiling --no-run
//!
//!   Step 2 — record (run directly against the binary):
//!     samply record ./target/profiling/deps/aggregate_dict-<HASH> \
//!       --bench "dict_pod_key" --profile-time 30

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource_parquet::source::ParquetSource;
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_functions_aggregate::count::count_udaf;
use datafusion_physical_expr::aggregate::AggregateExprBuilder;
use datafusion_physical_expr::expressions::{col, lit};
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use futures::StreamExt;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use test_utils::AccessLogGenerator;
use tokio::runtime::Runtime;

/// Total rows written to each parquet file.
/// NOTE: kept below 150M to avoid RowConverter u32 offset overflow with
/// high-cardinality string group keys (~13 bytes/row * 150M > u32::MAX).
const NUM_ROWS: usize = 10_000_000;

/// Rows per batch during generation.
const BATCH_SIZE: usize = 8_192;

/// Write an iterator of RecordBatches to a parquet file.
/// Skips writing if the file already exists.
fn ensure_parquet(
    path: &PathBuf,
    label: &str,
    batches: impl Iterator<Item = RecordBatch>,
) -> SchemaRef {
    if path.exists() {
        println!("Reusing {label} parquet: {}", path.display());
        let file = File::open(path).unwrap();
        let reader =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                .unwrap();
        return reader.schema().clone();
    }

    println!("Writing {label} parquet ({NUM_ROWS} rows) → {} …", path.display());
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer: Option<ArrowWriter<File>> = None;
    let mut schema: Option<SchemaRef> = None;

    for batch in batches {
        if writer.is_none() {
            schema = Some(batch.schema());
            writer = Some(
                ArrowWriter::try_new(
                    file.try_clone().unwrap(),
                    batch.schema(),
                    Some(props.clone()),
                )
                .unwrap(),
            );
        }
        writer.as_mut().unwrap().write(&batch).unwrap();
    }
    writer.unwrap().close().unwrap();
    println!("Done.");
    schema.unwrap()
}

/// Cast all Dictionary(Int32, Utf8) columns in a batch to plain Utf8.
fn batch_dict_to_plain(batch: &RecordBatch, plain_schema: &SchemaRef) -> RecordBatch {
    let new_cols: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .zip(batch.schema().fields().iter())
        .map(|(arr, field)| match field.data_type() {
            DataType::Dictionary(_, _) => Arc::new(
                arrow::compute::cast(arr, &DataType::Utf8).unwrap(),
            ) as ArrayRef,
            _ => Arc::clone(arr),
        })
        .collect();
    RecordBatch::try_new(plain_schema.clone(), new_cols).unwrap()
}

/// Derive a plain-Utf8 schema from a dict schema.
fn plain_schema(dict_schema: &SchemaRef) -> SchemaRef {
    let fields: Vec<Field> = dict_schema
        .fields()
        .iter()
        .map(|f| match f.data_type() {
            DataType::Dictionary(_, _) => {
                Field::new(f.name(), DataType::Utf8, f.is_nullable())
            }
            _ => f.as_ref().clone(),
        })
        .collect();
    Arc::new(Schema::new(fields))
}

/// Build and execute:
///   AggregateExec (Single, GROUP BY group_col, COUNT(*))
///     DataSourceExec (ParquetSource)
fn run_aggregate(
    rt: &Runtime,
    task_ctx: Arc<datafusion::execution::TaskContext>,
    schema: SchemaRef,
    parquet_path: &PathBuf,
    group_col: &str,
) {
    let file_size = std::fs::metadata(parquet_path).unwrap().len();
    let pfile = PartitionedFile::new(parquet_path.to_str().unwrap().to_owned(), file_size);

    let source = Arc::new(ParquetSource::new(schema.clone()));
    let scan_config = FileScanConfigBuilder::new(
        ObjectStoreUrl::local_filesystem(),
        source,
    )
    .with_file(pfile)
    .build();

    let scan = DataSourceExec::from_data_source(scan_config);

    let group_by = PhysicalGroupBy::new_single(vec![(
        col(group_col, &schema).unwrap(),
        group_col.to_string(),
    )]);

    let count_expr = Arc::new(
        AggregateExprBuilder::new(count_udaf(), vec![lit(1i64)])
            .schema(schema.clone())
            .alias("COUNT(*)")
            .build()
            .unwrap(),
    );

    let agg = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![count_expr],
            vec![None],
            scan,
            schema,
        )
        .unwrap(),
    );

    rt.block_on(async {
        let mut stream = agg.execute(0, Arc::clone(&task_ctx)).unwrap();
        while let Some(batch) = stream.next().await {
            std::hint::black_box(batch.unwrap());
        }
    });
}

fn bench_aggregate(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let session_ctx = SessionContext::new();
    let task_ctx = session_ctx.task_ctx();

    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches")
        .join("data");

    let dict_path = data_dir.join("access_log_dict_10m.parquet");
    let plain_path = data_dir.join("access_log_plain_10m.parquet");

    // High-cardinality generator: ~90% distinct pod values per batch.
    // pods_per_host=500..600 gives hundreds of distinct pods per host.
    // entries_per_container=1..2 means each pod/container combo contributes
    // only 1 row, so a batch of 8192 rows sees ~8000+ distinct pod values.
    let high_card_gen = || {
        AccessLogGenerator::new()
            .with_row_limit(NUM_ROWS)
            .with_max_batch_size(BATCH_SIZE)
            .with_pods_per_host(500..600)
            .with_entries_per_container(1..2)
    };

    // Dict file — native output of AccessLogGenerator (reused from repartition_dict if exists)
    let dict_schema = ensure_parquet(&dict_path, "dict", high_card_gen());

    // Plain file — same data with dict columns cast to Utf8
    let p_schema = plain_schema(&dict_schema);
    let p_schema_clone = p_schema.clone();
    ensure_parquet(
        &plain_path,
        "plain",
        high_card_gen().map(move |b| batch_dict_to_plain(&b, &p_schema_clone)),
    );

    let mut group = c.benchmark_group("aggregate");
    group.sample_size(10);

    // -----------------------------------------------------------------------
    // Dictionary(Int32, Utf8) group key — RowConverter must resolve dict lookups
    // -----------------------------------------------------------------------
    group.bench_function("dict_pod_key", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();
                run_aggregate(
                    &rt,
                    Arc::clone(&task_ctx),
                    dict_schema.clone(),
                    &dict_path,
                    "pod",
                );
                total += start.elapsed();
            }
            total
        })
    });

    // -----------------------------------------------------------------------
    // Plain Utf8 group key — RowConverter encodes string bytes directly
    // -----------------------------------------------------------------------
    group.bench_function("plain_pod_key", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();
                run_aggregate(
                    &rt,
                    Arc::clone(&task_ctx),
                    p_schema.clone(),
                    &plain_path,
                    "pod",
                );
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

criterion_group!(benches, bench_aggregate);
criterion_main!(benches);
