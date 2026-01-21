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

//! Adapter for integrating DataFusion's [`MemoryPool`] with Arrow's memory tracking APIs.

use crate::memory_pool::{MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation};
use std::fmt::Debug;
use std::sync::Arc;

/// An adapter that implements Arrow's [`arrow_buffer::MemoryPool`] trait
/// by wrapping a DataFusion [`MemoryPool`].
///
/// This allows DataFusion's memory management system to be used with Arrow's
/// memory allocation APIs. Each reservation made through this pool will be
/// tracked using the provided [`MemoryConsumer`], enabling DataFusion to
/// monitor and limit memory usage across Arrow operations.
///
/// This is useful when you want Arrow operations (such as array builders
/// or compute kernels) to participate in DataFusion's memory management
/// and respect the same memory limits as DataFusion operators.
#[derive(Debug)]
pub struct ArrowMemoryPool {
    inner: Arc<dyn MemoryPool>,
    consumer: MemoryConsumer,
}

impl ArrowMemoryPool {
    /// Creates a new [`ArrowMemoryPool`] that wraps the given DataFusion [`MemoryPool`]
    /// and tracks allocations under the specified [`MemoryConsumer`].
    pub fn new(inner: Arc<dyn MemoryPool>, consumer: MemoryConsumer) -> Self {
        Self { inner, consumer }
    }
}

impl arrow_buffer::MemoryReservation for MemoryReservation {
    fn size(&self) -> usize {
        MemoryReservation::size(self)
    }

    fn resize(&mut self, new_size: usize) {
        MemoryReservation::resize(self, new_size)
    }
}

impl arrow_buffer::MemoryPool for ArrowMemoryPool {
    fn reserve(&self, size: usize) -> Box<dyn arrow_buffer::MemoryReservation> {
        let consumer = self.consumer.clone_with_new_id();
        let reservation = consumer.register(&self.inner);
        reservation.grow(size);

        Box::new(reservation)
    }

    fn available(&self) -> isize {
        // The pool may be overfilled, so this method might return a negative value.
        (self.capacity() as i128 - self.used() as i128)
            .try_into()
            .unwrap_or(isize::MIN)
    }

    fn used(&self) -> usize {
        self.inner.reserved()
    }

    fn capacity(&self) -> usize {
        match self.inner.memory_limit() {
            MemoryLimit::Infinite | MemoryLimit::Unknown => usize::MAX,
            MemoryLimit::Finite(capacity) => capacity,
        }
    }
}

/// Claims all buffers in an array using the provided memory pool.
///
/// This recursively claims all buffers used by the array, including:
/// - Data buffers
/// - Null buffers
/// - Child array buffers (for nested types)
///
/// Buffers are tracked via Arc reference counting, so if multiple arrays
/// share the same underlying buffer, only one claim will succeed and the
/// others will be no-ops.
pub fn claim_array(array: &dyn arrow::array::Array, pool: &dyn arrow_buffer::MemoryPool) {
    let array_data = array.to_data();

    // Claim data buffers
    for buffer in array_data.buffers() {
        buffer.claim(pool);
    }

    // Claim null buffer if present
    if let Some(null_buffer) = array_data.nulls() {
        null_buffer.inner().inner().claim(pool);
    }

    // Recursively claim child arrays (for nested types like List, Struct, etc.)
    for child in array_data.child_data() {
        claim_array(&arrow::array::make_array(child.clone()), pool);
    }
}

/// Claims all arrays in a RecordBatch using the provided memory pool.
///
/// This is a convenience function that calls [`claim_array`] on each column
/// in the batch. Useful for claiming batches returned from operations that
/// create new arrays (e.g., coalescing, sorting, joining).
///
/// # Example
/// ```ignore
/// use datafusion_execution::memory_pool::arrow::claim_batch;
///
/// let batch = arrow::compute::concat_batches(&schema, &batches)?;
/// claim_batch(&batch, pool);
/// ```
pub fn claim_batch(
    batch: &arrow::record_batch::RecordBatch,
    pool: &dyn arrow_buffer::MemoryPool,
) {
    for column in batch.columns() {
        claim_array(column.as_ref(), pool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_pool::UnboundedMemoryPool;
    use arrow::array::{Array, Int32Array};
    use arrow_buffer::MemoryPool as ArrowMemoryPoolTrait;

    #[test]
    pub fn can_claim_array() {
        let pool = Arc::new(UnboundedMemoryPool::default());

        let consumer = MemoryConsumer::new("arrow");
        let arrow_pool = ArrowMemoryPool::new(pool, consumer);

        let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        claim_array(&array, &arrow_pool);

        assert_eq!(arrow_pool.used(), array.get_buffer_memory_size());

        let slice = array.slice(0, 2);

        // This should be a no-op since it shares the same buffer
        claim_array(&slice, &arrow_pool);

        assert_eq!(arrow_pool.used(), array.get_buffer_memory_size());
    }

    #[test]
    pub fn test_array_memory_released_on_drop() {
        // Create a DataFusion memory pool
        let df_pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());

        // Initial state: pool should be empty
        assert_eq!(df_pool.reserved(), 0);

        // Create consumer and register with the pool
        let consumer = MemoryConsumer::new("test_operator");
        let reservation = consumer.register(&df_pool);

        // Get the arrow pool from the reservation
        let arrow_pool = reservation.arrow_pool();

        let array_size;
        {
            // Create an array
            let array = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            array_size = array.get_buffer_memory_size();

            // Claim the array using the arrow pool
            claim_array(&array, arrow_pool.as_ref());

            // Verify memory is tracked in DataFusion pool
            assert_eq!(
                df_pool.reserved(),
                array_size,
                "DataFusion pool should track the array memory after claim"
            );

            // Also verify arrow pool reports same usage
            assert_eq!(
                arrow_pool.used(),
                array_size,
                "Arrow pool should report same usage as DataFusion pool"
            );

            // Array is about to be dropped here
        }

        // After array is dropped, memory should be released from DataFusion pool
        assert_eq!(
            df_pool.reserved(),
            0,
            "DataFusion pool should have memory released after array is dropped"
        );

        // Arrow pool should also show 0 usage
        assert_eq!(
            arrow_pool.used(),
            0,
            "Arrow pool should show 0 usage after array is dropped"
        );
    }

    #[test]
    pub fn test_multiple_arrays_memory_tracking() {
        let df_pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let consumer = MemoryConsumer::new("test_operator");
        let reservation = consumer.register(&df_pool);
        let arrow_pool = reservation.arrow_pool();

        // Create first array
        let array1 = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let size1 = array1.get_buffer_memory_size();
        claim_array(&array1, arrow_pool.as_ref());

        assert_eq!(df_pool.reserved(), size1);

        // Create second array
        let array2 = Int32Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80]);
        let size2 = array2.get_buffer_memory_size();
        claim_array(&array2, arrow_pool.as_ref());

        // Both arrays' memory should be tracked
        assert_eq!(
            df_pool.reserved(),
            size1 + size2,
            "Pool should track both arrays"
        );

        // Drop first array
        drop(array1);

        // Only second array's memory should remain
        assert_eq!(
            df_pool.reserved(),
            size2,
            "Only second array's memory should remain after dropping first"
        );

        // Drop second array
        drop(array2);

        // All memory should be released
        assert_eq!(
            df_pool.reserved(),
            0,
            "All memory should be released after dropping both arrays"
        );
    }

    #[test]
    pub fn test_shared_buffer_not_double_counted() {
        let df_pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let consumer = MemoryConsumer::new("test_operator");
        let reservation = consumer.register(&df_pool);
        let arrow_pool = reservation.arrow_pool();

        // Create original array
        let array = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let array_size = array.get_buffer_memory_size();
        claim_array(&array, arrow_pool.as_ref());

        assert_eq!(df_pool.reserved(), array_size);

        // Create slices that share the same buffer
        let slice1 = array.slice(0, 5);
        let slice2 = array.slice(5, 5);

        // Claim the slices - these should be no-ops because buffer already claimed
        claim_array(&slice1, arrow_pool.as_ref());
        claim_array(&slice2, arrow_pool.as_ref());

        // Memory usage should still be the same (not triple counted)
        assert_eq!(
            df_pool.reserved(),
            array_size,
            "Slices sharing buffer should not increase memory usage"
        );

        // Drop slices (should not affect memory since they share buffer)
        drop(slice1);
        drop(slice2);

        // Memory should still be tracked (original array still alive)
        assert_eq!(
            df_pool.reserved(),
            array_size,
            "Memory should still be tracked while original array is alive"
        );

        // Drop original array
        drop(array);

        // Now memory should be released
        assert_eq!(
            df_pool.reserved(),
            0,
            "Memory should be released after original array is dropped"
        );
    }

    #[test]
    pub fn test_mixed_datafusion_and_arrow_tracking() {
        // Create a DataFusion memory pool
        let df_pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());

        // Initial state: pool should be empty
        assert_eq!(df_pool.reserved(), 0);

        // Create consumer and register with the pool
        let consumer = MemoryConsumer::new("test_operator");
        let reservation = consumer.register(&df_pool);

        // Add some regular DataFusion memory tracking (e.g., from a Vec or other data structure)
        reservation.grow(5);
        assert_eq!(
            df_pool.reserved(),
            5,
            "DataFusion pool should track the 5 bytes we added"
        );

        let array_size;
        {
            // Now also track an Arrow array
            let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
            array_size = array.get_buffer_memory_size();

            // Get the arrow pool and claim the array
            let arrow_pool = reservation.arrow_pool();
            claim_array(&array, arrow_pool.as_ref());

            // Both the regular reservation and the array should be tracked
            assert_eq!(df_pool.reserved(), 5 + array_size,
                "DataFusion pool should track both the regular 5 bytes and the array memory");

            // Array is about to be dropped here
        }

        // After array is dropped, only the regular reservation should remain
        assert_eq!(df_pool.reserved(), 5,
            "DataFusion pool should still have the 5 bytes, but array memory should be released");

        // Clean up the regular reservation
        reservation.free();
        assert_eq!(df_pool.reserved(), 0, "All memory should be released");
    }

    #[test]
    pub fn test_two_reservations_same_array() {
        // This test verifies what happens when two different MemoryReservations
        // try to claim the same array
        let df_pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());

        // Create array ONCE
        let array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let array_size = array.get_buffer_memory_size();

        // Consumer 1 registers
        let consumer1 = MemoryConsumer::new("operator1");
        let reservation1 = consumer1.register(&df_pool);
        let arrow_pool1 = reservation1.arrow_pool();

        // Consumer 2 registers
        let consumer2 = MemoryConsumer::new("operator2");
        let reservation2 = consumer2.register(&df_pool);
        let arrow_pool2 = reservation2.arrow_pool();

        // First claim
        claim_array(&array, arrow_pool1.as_ref());
        assert_eq!(
            df_pool.reserved(),
            array_size,
            "First claim should track the array"
        );

        // Second claim on the SAME array with a DIFFERENT arrow pool
        claim_array(&array, arrow_pool2.as_ref());

        // The key question: is it double-counted?
        // Arrow's claim() should detect that the buffer already has a reservation
        // and NOT create a second one, so we should still see only array_size
        assert_eq!(
            df_pool.reserved(),
            array_size,
            "Second claim should NOT double-count - buffer already has a reservation"
        );
    }
}
