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

//! Memory-tracked wrappers for Arrow compute kernels
//!
//! This module provides wrappers around Arrow compute kernels that automatically
//! claim the resulting arrays in the DataFusion memory pool. This ensures that:
//!
//! 1. All array allocations are tracked by DataFusion's memory management
//! 2. Intermediate allocations (like null buffers, offset buffers) are tracked
//! 3. Memory limits are enforced during kernel execution, not just after
//! 4. Developers don't need to remember to manually claim arrays
//!
//! # Usage
//!
//! ```ignore
//! use datafusion_execution::memory_pool::compute;
//!
//! // Instead of:
//! // let result = arrow::compute::filter(&array, &predicate)?;
//!
//! // Use:
//! let result = compute::filter(&array, &predicate, &reservation)?;
//! ```
//!
//! # Design
//!
//! These wrappers follow a simple pattern:
//! 1. Call the Arrow compute kernel
//! 2. Claim all buffers in the result using the provided memory pool
//! 3. Return the result
//!
//! The memory tracking happens automatically via RAII - when the array is dropped,
//! the memory is automatically released back to the pool.

use crate::memory_pool::arrow::claim_array;
use crate::memory_pool::MemoryReservation;
use arrow::array::{Array, ArrayRef, ArrowPrimitiveType, BooleanArray, RecordBatch};
use arrow::compute;
use arrow::datatypes::Schema;
use datafusion_common::Result;
use std::sync::Arc;

/// Helper macro to create a memory-tracked wrapper for Arrow compute kernels
/// that return a single ArrayRef.
///
/// The pattern is:
/// 1. Call the Arrow kernel
/// 2. Claim the result array
/// 3. Return it
macro_rules! wrap_compute_array {
    (
        $(#[$meta:meta])*
        $fn_name:ident($($param:ident: $param_ty:ty),*) => $arrow_fn:path
    ) => {
        $(#[$meta])*
        pub fn $fn_name($($param: $param_ty,)* reservation: &MemoryReservation) -> Result<ArrayRef> {
            let result = $arrow_fn($($param),*)?;
            claim_array(result.as_ref(), reservation.arrow_pool().as_ref());
            Ok(result)
        }
    };
}

/// Helper macro to create a memory-tracked wrapper for Arrow compute kernels
/// that return a RecordBatch.
///
/// Claims all arrays in the result batch.
macro_rules! wrap_compute_batch {
    (
        $(#[$meta:meta])*
        $fn_name:ident($($param:ident: $param_ty:ty),*) => $arrow_fn:path
    ) => {
        $(#[$meta])*
        pub fn $fn_name($($param: $param_ty,)* reservation: &MemoryReservation) -> Result<RecordBatch> {
            let result = $arrow_fn($($param),*)?;

            // Claim all arrays in the result batch
            for array in result.columns() {
                claim_array(array.as_ref(), reservation.arrow_pool().as_ref());
            }

            Ok(result)
        }
    };
}

/// Helper macro for kernels that return typed arrays (not ArrayRef).
///
/// Used for kernels like lexsort_to_indices that return UInt32Array.
macro_rules! wrap_compute_typed {
    (
        $(#[$meta:meta])*
        $fn_name:ident($($param:ident: $param_ty:ty),*) -> $ret_ty:ty => $arrow_fn:path
    ) => {
        $(#[$meta])*
        pub fn $fn_name($($param: $param_ty,)* reservation: &MemoryReservation) -> Result<$ret_ty> {
            let result = $arrow_fn($($param),*)?;
            claim_array(&result, reservation.arrow_pool().as_ref());
            Ok(result)
        }
    };
}

wrap_compute_array!(
    filter(array: &dyn Array, predicate: &BooleanArray) => compute::filter
);

wrap_compute_array!(
    concat(arrays: &[&dyn Array]) => compute::concat
);

wrap_compute_batch!(
    filter_record_batch(batch: &RecordBatch, predicate: &BooleanArray) => compute::filter_record_batch
);

wrap_compute_typed!(
    is_null(array: &dyn Array) -> BooleanArray => compute::is_null
);

wrap_compute_typed!(
    is_not_null(array: &dyn Array) -> BooleanArray => compute::is_not_null
);

wrap_compute_typed!(
    lexsort_to_indices(arrays: &[compute::SortColumn], limit: Option<usize>) -> arrow::array::UInt32Array => compute::lexsort_to_indices
);

// For kernels with complex signatures (extra parameters, generics, lifetimes),
// we define them manually since they don't fit the simple macro pattern

/// Take elements from an array using indices, with memory tracking.
///
/// This is a wrapper around [`arrow::compute::take`] that automatically
/// claims the resulting array in the memory pool.
pub fn take(
    array: &dyn Array,
    indices: &dyn Array,
    reservation: &MemoryReservation,
) -> Result<ArrayRef> {
    let result = compute::take(array, indices, None)?;
    claim_array(result.as_ref(), reservation.arrow_pool().as_ref());
    Ok(result)
}

/// Cast array to a different data type, with memory tracking.
///
/// This is a wrapper around [`arrow::compute::cast`] that automatically
/// claims the resulting array in the memory pool.
pub fn cast(
    array: &dyn Array,
    to_type: &arrow::datatypes::DataType,
    reservation: &MemoryReservation,
) -> Result<ArrayRef> {
    let result = compute::cast(array, to_type)?;
    claim_array(result.as_ref(), reservation.arrow_pool().as_ref());
    Ok(result)
}

/// Concatenate record batches, with memory tracking.
///
/// This is a wrapper around [`arrow::compute::concat_batches`] that automatically
/// claims the resulting batch's arrays in the memory pool.
pub fn concat_batches<'a>(
    schema: &Arc<Schema>,
    batches: impl IntoIterator<Item = &'a RecordBatch>,
    reservation: &MemoryReservation,
) -> Result<RecordBatch> {
    let result = compute::concat_batches(schema, batches)?;

    // Claim all arrays in the result batch
    for array in result.columns() {
        claim_array(array.as_ref(), reservation.arrow_pool().as_ref());
    }

    Ok(result)
}

/// Helper macro to create memory-tracked builder finish functions.
///
/// This generates a function that takes a mutable reference to a builder,
/// finishes it, claims the result in the memory pool, and returns the array.
macro_rules! wrap_builder_finish {
    (
        $(#[$meta:meta])*
        $fn_name:ident($builder_ty:ty) -> $array_ty:ty
    ) => {
        $(#[$meta])*
        pub fn $fn_name(
            builder: &mut $builder_ty,
            reservation: &MemoryReservation,
        ) -> $array_ty {
            let array = builder.finish();
            claim_array(&array, reservation.arrow_pool().as_ref());
            array
        }
    };
}

wrap_builder_finish!(finish_string_builder(arrow::array::StringBuilder) -> arrow::array::StringArray);
wrap_builder_finish!(finish_large_string_builder(arrow::array::LargeStringBuilder) -> arrow::array::LargeStringArray);
wrap_builder_finish!(finish_binary_builder(arrow::array::BinaryBuilder) -> arrow::array::BinaryArray);
wrap_builder_finish!(finish_large_binary_builder(arrow::array::LargeBinaryBuilder) -> arrow::array::LargeBinaryArray);
wrap_builder_finish!(finish_fixed_size_binary_builder(arrow::array::FixedSizeBinaryBuilder) -> arrow::array::FixedSizeBinaryArray);
wrap_builder_finish!(finish_boolean_builder(arrow::array::BooleanBuilder) -> BooleanArray);

/// Finish a PrimitiveBuilder and claim the resulting PrimitiveArray in the memory pool.
pub fn finish_primitive_builder<T: ArrowPrimitiveType>(
    builder: &mut arrow::array::PrimitiveBuilder<T>,
    reservation: &MemoryReservation,
) -> arrow::array::PrimitiveArray<T> {
    let array = builder.finish();
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Finish a GenericByteBuilder and claim the resulting array in the memory pool.
pub fn finish_generic_byte_builder<T: arrow::array::types::ByteArrayType>(
    builder: &mut arrow::array::GenericByteBuilder<T>,
    reservation: &MemoryReservation,
) -> arrow::array::GenericByteArray<T> {
    let array = builder.finish();
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create a RecordBatch and claim all its arrays in the memory pool.
pub fn record_batch_try_new(
    schema: Arc<Schema>,
    columns: Vec<ArrayRef>,
    reservation: &MemoryReservation,
) -> Result<RecordBatch> {
    let batch = RecordBatch::try_new(schema, columns)?;

    for array in batch.columns() {
        claim_array(array.as_ref(), reservation.arrow_pool().as_ref());
    }

    Ok(batch)
}

/// Finish a BooleanBufferBuilder and claim the resulting BooleanBuffer in the memory pool.
pub fn finish_boolean_buffer_builder(
    builder: &mut arrow::array::BooleanBufferBuilder,
    reservation: &MemoryReservation,
) -> arrow::buffer::BooleanBuffer {
    let buffer = builder.finish();
    buffer.inner().claim(reservation.arrow_pool().as_ref());
    buffer
}

/// Finish a NullBufferBuilder and claim the resulting NullBuffer in the memory pool (if any).
pub fn finish_null_buffer_builder(
    builder: &mut arrow::array::NullBufferBuilder,
    reservation: &MemoryReservation,
) -> Option<arrow::buffer::NullBuffer> {
    let null_buffer = builder.finish();
    if let Some(ref nb) = null_buffer {
        nb.inner().inner().claim(reservation.arrow_pool().as_ref());
    }
    null_buffer
}

/// Create an Int32Array from a vector of values, with memory tracking.
pub fn int32_array_from_vec(
    data: Vec<i32>,
    reservation: &MemoryReservation,
) -> arrow::array::Int32Array {
    let array = arrow::array::Int32Array::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create an Int32Array from a vector of optional values, with memory tracking.
pub fn int32_array_from_option_vec(
    data: Vec<Option<i32>>,
    reservation: &MemoryReservation,
) -> arrow::array::Int32Array {
    let array = arrow::array::Int32Array::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create an Int64Array from a vector of values, with memory tracking.
pub fn int64_array_from_vec(
    data: Vec<i64>,
    reservation: &MemoryReservation,
) -> arrow::array::Int64Array {
    let array = arrow::array::Int64Array::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create an Int64Array from a vector of optional values, with memory tracking.
pub fn int64_array_from_option_vec(
    data: Vec<Option<i64>>,
    reservation: &MemoryReservation,
) -> arrow::array::Int64Array {
    let array = arrow::array::Int64Array::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create a UInt32Array from a vector of values, with memory tracking.
pub fn uint32_array_from_vec(
    data: Vec<u32>,
    reservation: &MemoryReservation,
) -> arrow::array::UInt32Array {
    let array = arrow::array::UInt32Array::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create a UInt64Array from a vector of values, with memory tracking.
pub fn uint64_array_from_vec(
    data: Vec<u64>,
    reservation: &MemoryReservation,
) -> arrow::array::UInt64Array {
    let array = arrow::array::UInt64Array::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create a Float32Array from a vector of values, with memory tracking.
pub fn float32_array_from_vec(
    data: Vec<f32>,
    reservation: &MemoryReservation,
) -> arrow::array::Float32Array {
    let array = arrow::array::Float32Array::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create a Float64Array from a vector of values, with memory tracking.
pub fn float64_array_from_vec(
    data: Vec<f64>,
    reservation: &MemoryReservation,
) -> arrow::array::Float64Array {
    let array = arrow::array::Float64Array::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create a BooleanArray from a vector of values, with memory tracking.
pub fn boolean_array_from_vec(
    data: Vec<bool>,
    reservation: &MemoryReservation,
) -> BooleanArray {
    let array = BooleanArray::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create a BooleanArray from a vector of optional values, with memory tracking.
pub fn boolean_array_from_option_vec(
    data: Vec<Option<bool>>,
    reservation: &MemoryReservation,
) -> BooleanArray {
    let array = BooleanArray::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create a StringArray from a vector of string slices, with memory tracking.
pub fn string_array_from_vec(
    data: Vec<&str>,
    reservation: &MemoryReservation,
) -> arrow::array::StringArray {
    let array = arrow::array::StringArray::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create a StringArray from a vector of optional string slices, with memory tracking.
pub fn string_array_from_option_vec(
    data: Vec<Option<&str>>,
    reservation: &MemoryReservation,
) -> arrow::array::StringArray {
    let array = arrow::array::StringArray::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Create a BinaryArray from a vector of byte slices, with memory tracking.
pub fn binary_array_from_vec(
    data: Vec<&[u8]>,
    reservation: &MemoryReservation,
) -> arrow::array::BinaryArray {
    let array = arrow::array::BinaryArray::from(data);
    claim_array(&array, reservation.arrow_pool().as_ref());
    array
}

/// Macro to generate from_iter_values wrappers for primitive array types
macro_rules! wrap_array_from_iter_values {
    (
        $(#[$meta:meta])*
        $fn_name:ident($item_ty:ty) => $array_ty:ty
    ) => {
        $(#[$meta])*
        pub fn $fn_name(
            iter: impl IntoIterator<Item = $item_ty>,
            reservation: &MemoryReservation,
        ) -> $array_ty {
            let array = <$array_ty>::from_iter_values(iter);
            claim_array(&array, reservation.arrow_pool().as_ref());
            array
        }
    };
}

wrap_array_from_iter_values!(int64_array_from_iter_values(i64) => arrow::array::Int64Array);

wrap_array_from_iter_values!(
    int32_array_from_iter_values(i32) => arrow::array::Int32Array
);

wrap_array_from_iter_values!(
    int16_array_from_iter_values(i16) => arrow::array::Int16Array
);

wrap_array_from_iter_values!(
    int8_array_from_iter_values(i8) => arrow::array::Int8Array
);

wrap_array_from_iter_values!(
    uint64_array_from_iter_values(u64) => arrow::array::UInt64Array
);

wrap_array_from_iter_values!(
    uint32_array_from_iter_values(u32) => arrow::array::UInt32Array
);

wrap_array_from_iter_values!(
    uint16_array_from_iter_values(u16) => arrow::array::UInt16Array
);

wrap_array_from_iter_values!(
    uint8_array_from_iter_values(u8) => arrow::array::UInt8Array
);

wrap_array_from_iter_values!(
    float64_array_from_iter_values(f64) => arrow::array::Float64Array
);

wrap_array_from_iter_values!(
    float32_array_from_iter_values(f32) => arrow::array::Float32Array
);

wrap_array_from_iter_values!(
    date32_array_from_iter_values(i32) => arrow::array::Date32Array
);

wrap_array_from_iter_values!(
    date64_array_from_iter_values(i64) => arrow::array::Date64Array
);

wrap_array_from_iter_values!(
    timestamp_second_array_from_iter_values(i64) => arrow::array::TimestampSecondArray
);

wrap_array_from_iter_values!(
    timestamp_millisecond_array_from_iter_values(i64) => arrow::array::TimestampMillisecondArray
);

wrap_array_from_iter_values!(
    timestamp_microsecond_array_from_iter_values(i64) => arrow::array::TimestampMicrosecondArray
);

wrap_array_from_iter_values!(
    timestamp_nanosecond_array_from_iter_values(i64) => arrow::array::TimestampNanosecondArray
);

wrap_array_from_iter_values!(
    duration_second_array_from_iter_values(i64) => arrow::array::DurationSecondArray
);

wrap_array_from_iter_values!(
    duration_millisecond_array_from_iter_values(i64) => arrow::array::DurationMillisecondArray
);

wrap_array_from_iter_values!(
    duration_microsecond_array_from_iter_values(i64) => arrow::array::DurationMicrosecondArray
);

wrap_array_from_iter_values!(
    duration_nanosecond_array_from_iter_values(i64) => arrow::array::DurationNanosecondArray
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_pool::{MemoryConsumer, MemoryPool, UnboundedMemoryPool};
    use arrow::datatypes::{DataType, Field};

    #[test]
    fn test_array_construction_wrappers() {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let consumer = MemoryConsumer::new("test");
        let reservation = consumer.register(&pool);

        assert_eq!(pool.reserved(), 0);

        // Create first array - pool size increases
        let int_array = int32_array_from_vec(vec![1, 2, 3, 4, 5], &reservation);
        let after_int = pool.reserved();
        assert!(after_int > 0, "Pool should track int array");

        // Create second array - pool size increases more
        let bool_array =
            boolean_array_from_vec(vec![true, false, true, false, true], &reservation);
        let after_bool = pool.reserved();
        assert!(after_bool > after_int, "Pool should track bool array");

        // Create third array - pool size increases more
        let str_array = string_array_from_vec(vec!["hello", "world"], &reservation);
        let after_str = pool.reserved();
        assert!(after_str > after_bool, "Pool should track string array");

        // Drop arrays - pool size decreases
        drop(int_array);
        let after_drop_int = pool.reserved();
        assert!(
            after_drop_int < after_str,
            "Pool should release int array memory"
        );

        drop(bool_array);
        let after_drop_bool = pool.reserved();
        assert!(
            after_drop_bool < after_drop_int,
            "Pool should release bool array memory"
        );

        drop(str_array);
        let after_drop_all = pool.reserved();
        assert_eq!(after_drop_all, 0, "Pool should be empty after all drops");
    }

    #[test]
    fn test_compute_kernel_wrappers() {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let consumer = MemoryConsumer::new("test");
        let reservation = consumer.register(&pool);

        assert_eq!(pool.reserved(), 0);

        // Create input arrays
        let data = int32_array_from_vec(vec![10, 20, 30, 40, 50], &reservation);
        let predicate =
            boolean_array_from_vec(vec![true, false, true, false, true], &reservation);
        let after_inputs = pool.reserved();
        assert!(after_inputs > 0, "Input arrays should be tracked");

        // Use compute wrapper - output array is also tracked, since there is no array reuse, consumed memory will increase
        let filtered = filter(&data, &predicate, &reservation).unwrap();
        let after_filter = pool.reserved();
        assert!(
            after_filter > after_inputs,
            "Filtered array should increase pool usage"
        );

        // Drop input arrays - pool decreases but filtered array remains
        drop(data);
        drop(predicate);
        let after_drop_inputs = pool.reserved();
        assert!(
            after_drop_inputs < after_filter,
            "Pool should decrease after dropping inputs"
        );
        assert!(
            after_drop_inputs > 0,
            "Pool should still track filtered array"
        );

        // Drop filtered array - pool goes to zero
        drop(filtered);
        assert_eq!(pool.reserved(), 0, "Pool should be empty");
    }

    #[test]
    fn test_builder_finish_wrappers() {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let consumer = MemoryConsumer::new("test");
        let reservation = consumer.register(&pool);

        assert_eq!(pool.reserved(), 0);

        // Build array incrementally (builder itself doesn't track memory yet)
        let mut string_builder = arrow::array::StringBuilder::new();
        string_builder.append_value("first");
        string_builder.append_null();
        string_builder.append_value("third");

        // After finish, memory is tracked in the pool
        let string_array = finish_string_builder(&mut string_builder, &reservation);
        let after_finish = pool.reserved();
        assert!(after_finish > 0, "Finished array should be tracked");

        // Drop array - pool goes back to zero
        drop(string_array);
        assert_eq!(pool.reserved(), 0, "Pool should be empty after drop");
    }

    #[test]
    fn test_complete_workflow_with_wrappers() {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let consumer = MemoryConsumer::new("aggregation");
        let reservation = consumer.register(&pool);

        assert_eq!(pool.reserved(), 0);

        // Step 1: Create input data
        let values = int32_array_from_vec(vec![100, 200, 300, 400], &reservation);
        let mask = boolean_array_from_vec(vec![true, true, false, true], &reservation);
        let after_inputs = pool.reserved();
        assert!(after_inputs > 0, "Inputs should be tracked");

        // Step 2: Filter - pool increases
        let filtered = filter(&values, &mask, &reservation).unwrap();
        let after_filter = pool.reserved();
        assert!(
            after_filter > after_inputs,
            "Filter output should be tracked"
        );

        // Step 3: Cast - pool increases more
        let casted = cast(&filtered, &DataType::Int64, &reservation).unwrap();
        let after_cast = pool.reserved();
        assert!(after_cast > after_filter, "Cast output should be tracked");

        // Step 4: Build result - pool increases more
        let mut result_builder = arrow::array::Int64Builder::new();
        let casted_i64 = casted
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        for i in 0..casted_i64.len() {
            result_builder.append_value(casted_i64.value(i) * 2);
        }
        let final_result = finish_primitive_builder(&mut result_builder, &reservation);
        let after_build = pool.reserved();
        assert!(after_build > after_cast, "Built array should be tracked");

        // Drop intermediate results - pool decreases
        drop(values);
        drop(mask);
        drop(filtered);
        drop(casted);
        let after_drop_intermediates = pool.reserved();
        assert!(
            after_drop_intermediates < after_build,
            "Pool should decrease after dropping intermediates"
        );
        assert!(
            after_drop_intermediates > 0,
            "Final result should still be tracked"
        );

        // Drop final result - pool goes to zero
        drop(final_result);
        assert_eq!(pool.reserved(), 0, "Pool should be empty");
    }

    #[test]
    fn test_record_batch_workflow() {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let consumer = MemoryConsumer::new("batch_op");
        let reservation = consumer.register(&pool);

        assert_eq!(pool.reserved(), 0);

        // Create schema and arrays
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let ids = int32_array_from_vec(vec![1, 2, 3, 4], &reservation);
        let values = float64_array_from_vec(vec![1.5, 2.5, 3.5, 4.5], &reservation);
        let after_arrays = pool.reserved();
        assert!(after_arrays > 0, "Arrays should be tracked");

        // Create RecordBatch - pool increases (batch claims the arrays)
        let batch = record_batch_try_new(
            Arc::clone(&schema),
            vec![Arc::new(ids), Arc::new(values)],
            &reservation,
        )
        .unwrap();
        let after_batch = pool.reserved();
        assert!(
            after_batch >= after_arrays,
            "Batch arrays should be tracked"
        );

        // Filter batch - creates new arrays, pool increases
        let predicate =
            boolean_array_from_vec(vec![true, false, true, false], &reservation);
        let filtered_batch =
            filter_record_batch(&batch, &predicate, &reservation).unwrap();
        let after_filter = pool.reserved();
        assert!(
            after_filter > after_batch,
            "Filtered batch should increase pool"
        );

        // Drop everything - pool goes to zero
        drop(batch);
        drop(predicate);
        drop(filtered_batch);
        assert_eq!(pool.reserved(), 0, "Pool should be empty");
    }
}
