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

//! Helpers for working with the `arrow.timestamp_with_offset` canonical Arrow extension type.
//!
//! The physical layout of a `TimestampWithOffset` field is:
//!
//! ```text
//! Struct {
//!     timestamp:      Timestamp(unit, "UTC"),   -- non-nullable
//!     offset_minutes: Int16,                    -- non-nullable, per-row offset east of UTC
//! }
//! ```
//!
//! See <https://arrow.apache.org/docs/format/CanonicalExtensions.html#timestamp-with-offset>.
//!
//! Most datetime functions can operate on the UTC `timestamp` child directly.  For functions that
//! need to display or truncate in *local* time (e.g. `to_char`, coarse `date_trunc` granularities)
//! the offset must be applied before delegating to the per-row chrono logic.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int16Array, StructArray};
use arrow::datatypes::{DataType, Field, Fields, TimeUnit};
use datafusion_common::{DataFusionError, Result};

/// The Arrow extension type name for `TimestampWithOffset`.
///
/// Defined by the Arrow spec:
/// <https://arrow.apache.org/docs/format/CanonicalExtensions.html#timestamp-with-offset>
pub const TIMESTAMP_WITH_OFFSET_NAME: &str = "arrow.timestamp_with_offset";

/// Key used in Arrow field metadata to record the extension type name.
pub const EXTENSION_TYPE_NAME_KEY: &str = "ARROW:extension:name";

/// Returns `true` if `data_type` is the physical `Struct` that backs a
/// `TimestampWithOffset` extension field.
///
/// The check is purely structural (field names + types); it does not require
/// the `ARROW:extension:name` metadata key to be present on the outer field.
pub fn is_timestamp_with_offset(data_type: &DataType) -> bool {
    match data_type {
        DataType::Struct(fields) if fields.len() == 2 => {
            let ts_field = &fields[0];
            let off_field = &fields[1];
            ts_field.name() == "timestamp"
                && matches!(
                    ts_field.data_type(),
                    DataType::Timestamp(_, Some(tz)) if tz.as_ref() == "UTC"
                )
                && !ts_field.is_nullable()
                && off_field.name() == "offset_minutes"
                && matches!(off_field.data_type(), DataType::Int16)
                && !off_field.is_nullable()
        }
        _ => false,
    }
}

/// Extract the UTC `timestamp` child column from a `TimestampWithOffset` struct array.
///
/// Returns an error if the array does not have the expected structural layout.
pub fn timestamp_child(array: &dyn Array) -> Result<ArrayRef> {
    let struct_array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            DataFusionError::Execution(format!(
                "TimestampWithOffset: expected StructArray, got {:?}",
                array.data_type()
            ))
        })?;

    if struct_array.num_columns() < 1 {
        return Err(DataFusionError::Execution(
            "TimestampWithOffset: struct has no columns".into(),
        ));
    }

    Ok(Arc::clone(struct_array.column(0)))
}

/// Extract the `offset_minutes` child column (Int16) from a `TimestampWithOffset` struct array.
pub fn offset_minutes_child(array: &dyn Array) -> Result<&Int16Array> {
    let struct_array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            DataFusionError::Execution(format!(
                "TimestampWithOffset: expected StructArray, got {:?}",
                array.data_type()
            ))
        })?;

    if struct_array.num_columns() < 2 {
        return Err(DataFusionError::Execution(
            "TimestampWithOffset: struct has fewer than 2 columns".into(),
        ));
    }

    struct_array
        .column(1)
        .as_any()
        .downcast_ref::<Int16Array>()
        .ok_or_else(|| {
            DataFusionError::Execution(
                "TimestampWithOffset: offset_minutes column is not Int16".into(),
            )
        })
}

/// Reconstruct a `TimestampWithOffset` struct array from a (possibly recomputed) UTC timestamp
/// column and the original `offset_minutes` column.
///
/// `ts_array`  – a `Timestamp(unit, "UTC")` array
/// `off_array` – an `Int16Array` of per-row offsets in minutes
/// `nulls`     – nullness from the original outer struct (propagated as-is)
pub fn rebuild_timestamp_with_offset(
    ts_array: ArrayRef,
    off_array: ArrayRef,
    nulls: Option<arrow::buffer::NullBuffer>,
) -> Result<ArrayRef> {
    let ts_type = ts_array.data_type().clone();
    let off_type = off_array.data_type().clone();

    match &ts_type {
        DataType::Timestamp(_, Some(tz)) if tz.as_ref() == "UTC" => {}
        other => {
            return Err(DataFusionError::Execution(format!(
                "TimestampWithOffset rebuild: timestamp child must be Timestamp(_,\"UTC\"), got {other}"
            )));
        }
    }
    if off_type != DataType::Int16 {
        return Err(DataFusionError::Execution(format!(
            "TimestampWithOffset rebuild: offset_minutes child must be Int16, got {off_type}"
        )));
    }

    let fields = Fields::from(vec![
        Field::new("timestamp", ts_type, false),
        Field::new("offset_minutes", DataType::Int16, false),
    ]);

    let struct_array = StructArray::new(fields, vec![ts_array, off_array], nulls);

    Ok(Arc::new(struct_array) as ArrayRef)
}

/// Build the Arrow `Field` for a `TimestampWithOffset` column with the correct
/// `ARROW:extension:name` metadata so IPC readers can recognise the type.
pub fn timestamp_with_offset_field(
    name: &str,
    time_unit: TimeUnit,
    nullable: bool,
) -> Field {
    let ts_field = Field::new(
        "timestamp",
        DataType::Timestamp(time_unit, Some("UTC".into())),
        false,
    );
    let off_field = Field::new("offset_minutes", DataType::Int16, false);

    let struct_type = DataType::Struct(Fields::from(vec![ts_field, off_field]));

    let mut metadata = HashMap::new();
    metadata.insert(
        EXTENSION_TYPE_NAME_KEY.to_owned(),
        TIMESTAMP_WITH_OFFSET_NAME.to_owned(),
    );

    Field::new(name, struct_type, nullable).with_metadata(metadata)
}
