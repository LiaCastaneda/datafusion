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

//! Utilities for passing information from the build side of a hash join to the probe side.
//! This is also known as "sideways information passing", and enables optimizations that take
//! of the smaller build side to reduce data processed on the larger probe side.
//!
//! As an example, let's consider TPC-H Query 17:
//!
//! ```sql
//! select
//! sum(l_extendedprice) / 7.0 as avg_yearly
//! from
//!     lineitem,
//!     part
//! where
//!         p_partkey = l_partkey
//!   and p_brand = 'Brand#23'
//!   and p_container = 'MED BOX'
//!   and l_quantity < (
//!     select
//!             0.2 * avg(l_quantity)
//!     from
//!         lineitem
//!     where
//!             l_partkey = p_partkey
//! );
//! ```
//! The join portion of the query should look something like this:
//!
//! ```text
//!                            │                                                         
//!                            │                                                         
//!                 2044 Rows  │                                                         
//!                            │                                                         
//!                            ▼                                                         
//!                    ┌────────────────┐                                                 
//!                    │    HashJoin    │                                                 
//!                    │   p_partkey =  │                                                 
//!                    │   l_partkey    │                                                 
//!                    └──┬─────────┬───┘                     
//!              2M Rows  │         │  60M Rows              
//!                       │         │                          
//!                       │         │                          
//!              ┌────────┘         └─────────┐               
//!              │                            │               This scan decodes 60M values of l_quantity and l_extendedprice,
//!              ▼                            ▼               even though all but 2044 are filtered by the join!
//!      ┌──────────────────┐        ┌─────────────────────┐                                
//!      │Scan: part        │        │Scan: lineitem       │                  │             
//!      │projection:       │        │projection:          │                  │              
//!      │  p_partkey       │        │  l_quantity,        │                  │             
//!      │filters:          │        │  l_extendedprice,   │◀─ ─ ─ ─ ─ ─ ─ ─ ─              
//!      │  p_brand = ..    │        │  l_partkey          │                                
//!      │  p_container = ..│        │filters:             │                                
//!      │                  │        │  NONE               │                                
//!      └──────────────────┘        └─────────────────────┘                                
//! ```
//!
//! The key observation is that the scan of `lineitem` produces 60 million rows, but only 2044 of them
//! will pass the join condition. If we can push down a filter to the scan of `lineitem` that only limits
//! the number of rows produced, we can avoid a lot of unnecessary work.
//!
//! Given that in a hash join, we fully process the build side (in this case, `part`) before scanning partitions
//! of the probe side (`lineitem`), we can collect information from the build side that helps us construct filters
//! for the probe side. This allows us to transform the above plan into something like this:
//!
//! ```text
//!                            │                                                         
//!                            │                                                         
//!                 2044 Rows  │                                                         
//!                            │                                                         
//!                            ▼                                                         
//!                    ┌────────────────┐                                                 
//!                    │    HashJoin    │                                                 
//!                    │   p_partkey =  │                                                 
//!                    │   l_partkey    │                                                 
//!                    └──┬─────────┬───┘                     
//!              2M Rows  │         │  60M Rows              
//!                       │         │                          
//!                       │         │                          
//!              ┌────────┘         └─────────┐               
//!              │                            │               
//!              ▼                            ▼               
//!      ┌──────────────────┐        ┌──────────────────────────────┐                                
//!      │Scan: part        │        │Scan: lineitem                │         Now, the scan contains a filter that takes into account                      
//!      │projection:       │        │projection:                   │         min/max bounds from the left side. This enables scans that                       
//!      │  p_partkey       │        │  l_quantity,                 │         take advantage of late materialization to avoid decoding                     
//!      │filters:          │        │  l_extendedprice,            │         l_quantity and l_extendedprice for rows that do not match.                       
//!      │  p_brand = ..    │        │  l_partkey                   │                   │             
//!      │  p_container = ..│        │filters:                      │                   │             
//!      └──────────────────┘        │  l_partkey >= min(p_partkey) │                   │
//!                                  │    and                       │ ◀─ ─ ─ ─ ─ ─ ─ ─ ─
//!                                  │  l_partkey <= max(p_partkey) │
//!                                  │    and                       │
//!                                  │  ...                         │                                
//!                                  └──────────────────────────────┘                                
//! ```
//!
//! Dynamic filters are expressions that allow us to pass information sideways in a query plan. They essentially start out
//! as dummy filters that are always true (resulting in no selectivity), and may be updated later during query execution.
//! In the case of a hash join, we update dynamic filters after fully processing the build side, and before scanning the probe side.
//!
//! References:
//! - <https://github.com/apache/datafusion/issues/7955>
//! - <https://datafusion.apache.org/blog/2025/09/10/dynamic-filters/>

use std::{any::Any, fmt, hash::Hash, sync::Arc};

use crate::joins::utils::JoinHashMapType;
use crate::joins::PartitionMode;
use crate::ExecutionPlan;
use crate::ExecutionPlanProperties;

use ahash::RandomState;
use arrow::{
    array::BooleanArray,
    buffer::MutableBuffer,
    datatypes::{DataType, Schema},
    util::bit_util,
};
use datafusion_common::{hash_utils::create_hashes, Result, ScalarValue};
use datafusion_expr::{ColumnarValue, Operator};
use datafusion_physical_expr::expressions::{lit, BinaryExpr, DynamicFilterPhysicalExpr};
use datafusion_physical_expr::PhysicalExpr;

use itertools::Itertools;
use parking_lot::Mutex;
use tokio::sync::Barrier;

/// Random state used by RepartitionExec for partitioning data.
/// We use the same seeds to compute which partition a row belongs to,
/// ensuring consistency with how DataFusion partitions data across streams.
static REPARTITION_RANDOM_STATE: RandomState = RandomState::with_seeds(0, 0, 0, 0);

/// Represents the minimum and maximum values for a specific column.
/// Used in dynamic filter pushdown to establish value boundaries.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ColumnBounds {
    /// The minimum value observed for this column
    min: ScalarValue,
    /// The maximum value observed for this column  
    max: ScalarValue,
}

impl ColumnBounds {
    pub(crate) fn new(min: ScalarValue, max: ScalarValue) -> Self {
        Self { min, max }
    }
}

/// Represents the bounds for all join key columns from a single partition.
/// This contains the min/max values computed from one partition's build-side data.
#[derive(Debug, Clone)]
pub(crate) struct PartitionBounds {
    /// Partition identifier for debugging and determinism (not strictly necessary)
    partition: usize,
    /// Min/max bounds for each join key column in this partition.
    /// Index corresponds to the join key expression index.
    column_bounds: Vec<ColumnBounds>,
}

impl PartitionBounds {
    pub(crate) fn new(partition: usize, column_bounds: Vec<ColumnBounds>) -> Self {
        Self {
            partition,
            column_bounds,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.column_bounds.len()
    }

    pub(crate) fn get_column_bounds(&self, index: usize) -> Option<&ColumnBounds> {
        self.column_bounds.get(index)
    }
}

/// Coordinates information collected from the build side of the join, across multiple partitions
///
/// This structure ensures that dynamic filters are built with complete information from all
/// relevant partitions before being applied to probe-side scans. Incomplete filters would
/// incorrectly eliminate valid join results.
///
/// ## Synchronization Strategy
///
/// 1. Each partition relays values computed from its build-side data (e.g. min/max bounds, available hashes)
/// 2. The per-partition information is accumulated in shared state
/// 3. A [`Barrier`] is used to track reporters. Once the last partition reports, the information is merged and the filter is updated exactly once
///
/// ## Partition Counting
///
/// The `total_partitions` count represents how many times `collect_build_side` will be called:
///
/// - **CollectLeft**: Number of output partitions (each accesses shared build data)
/// - **Partitioned**: Number of input partitions (each builds independently)
///
/// ## Thread Safety
///
/// All fields use a single mutex to ensure correct coordination between concurrent
/// partition executions.
pub(crate) struct SharedBuildAccumulator {
    /// Shared state protected by a single mutex to avoid ordering concerns
    inner: Mutex<SharedBuildState>,
    /// Synchronization barrier to track when all partitions have reported
    barrier: Barrier,
    /// Dynamic filter for pushdown to probe side
    dynamic_filter: Arc<DynamicFilterPhysicalExpr>,
    /// Right side join expressions needed for creating filter bounds
    on_right: Vec<Arc<dyn PhysicalExpr>>,
    /// Random state used for hash computation
    random_state: &'static RandomState,
}

/// State protected by SharedBuildAccumulator's mutex
struct SharedBuildState {
    /// Bounds from completed partitions.
    /// Each element represents the column bounds computed by one partition.
    bounds: Vec<PartitionBounds>,
    /// Hash tables from the left (build) side, if enabled.
    /// - For CollectLeft mode: Vec with 1 element at index 0
    /// - For Partitioned mode: Vec with N elements, where hash_tables[i] is partition i's table
    ///   Uses Option to handle partitions that haven't reported yet or have no hashes
    hash_tables: Vec<Option<Arc<dyn JoinHashMapType>>>,
    /// Partition mode to determine how to create the filter (Single vs Partitioned)
    partition_mode: PartitionMode,
    /// Number of partitions (for Partitioned mode hash routing)
    num_partitions: usize,
}

impl SharedBuildAccumulator {
    /// Creates a new [SharedBuildAccumulator] configured for the given partition mode
    ///
    /// This method calculates how many times `collect_build_side` will be called based on the
    /// partition mode's execution pattern. This count is critical for determining when we have
    /// complete information from all partitions to build the dynamic filter.
    ///
    /// ## Partition Mode Execution Patterns
    ///
    /// - **CollectLeft**: Build side is collected ONCE from partition 0 and shared via `OnceFut`
    ///   across all output partitions. Each output partition calls `collect_build_side` to access the shared build data.
    ///   Although this results in multiple invocations, the  `report_partition_bounds` function contains deduplication logic to handle them safely.
    ///   Expected calls = number of output partitions.
    ///
    ///
    /// - **Partitioned**: Each partition independently builds its own hash table by calling
    ///   `collect_build_side` once. Expected calls = number of build partitions.
    ///
    /// - **Auto**: Placeholder mode resolved during optimization. Uses 1 as safe default since
    ///   the actual mode will be determined and a new bounds_accumulator created before execution.
    ///
    /// ## Why This Matters
    ///
    /// We cannot build a partial filter from some partitions - it would incorrectly eliminate
    /// valid join results. We must wait until we have complete bounds information from ALL
    /// relevant partitions before updating the dynamic filter.
    pub(crate) fn new_from_partition_mode(
        partition_mode: PartitionMode,
        left_child: &dyn ExecutionPlan,
        right_child: &dyn ExecutionPlan,
        dynamic_filter: Arc<DynamicFilterPhysicalExpr>,
        on_right: Vec<Arc<dyn PhysicalExpr>>,
        random_state: &'static RandomState,
    ) -> Self {
        // Troubleshooting: If partition counts are incorrect, verify this logic matches
        // the actual execution pattern in collect_build_side()
        let expected_calls = match partition_mode {
            // Each output partition accesses shared build data
            PartitionMode::CollectLeft => {
                right_child.output_partitioning().partition_count()
            }
            // Each partition builds its own data
            PartitionMode::Partitioned => {
                left_child.output_partitioning().partition_count()
            }
            // Default value, will be resolved during optimization (does not exist once `execute()` is called; will be replaced by one of the other two)
            PartitionMode::Auto => unreachable!("PartitionMode::Auto should not be present at execution time. This is a bug in DataFusion, please report it!"),
        };
        let num_partitions = match partition_mode {
            PartitionMode::CollectLeft => 1,
            PartitionMode::Partitioned => {
                left_child.output_partitioning().partition_count()
            }
            PartitionMode::Auto => unreachable!(),
        };

        Self {
            inner: Mutex::new(SharedBuildState {
                bounds: Vec::with_capacity(expected_calls),
                // Pre-allocate Vec with None placeholders for each partition
                hash_tables: vec![None; num_partitions],
                partition_mode,
                num_partitions,
            }),
            barrier: Barrier::new(expected_calls),
            dynamic_filter,
            on_right,
            random_state,
        }
    }

    /// Create a filter expression from individual partition bounds using OR logic.
    ///
    /// This creates a filter where each partition's bounds form a conjunction (AND)
    /// of column range predicates, and all partitions are combined with OR.
    ///
    /// For example, with 2 partitions and 2 columns:
    /// ((col0 >= p0_min0 AND col0 <= p0_max0 AND col1 >= p0_min1 AND col1 <= p0_max1)
    ///  OR
    ///  (col0 >= p1_min0 AND col0 <= p1_max0 AND col1 >= p1_min1 AND col1 <= p1_max1))
    pub(crate) fn create_filter_from_partition_bounds(
        &self,
        bounds: &[PartitionBounds],
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if bounds.is_empty() {
            return Ok(lit(true));
        }

        // Create a predicate for each partition
        let mut partition_predicates = Vec::with_capacity(bounds.len());

        for partition_bounds in bounds.iter().sorted_by_key(|b| b.partition) {
            // Create range predicates for each join key in this partition
            let mut column_predicates = Vec::with_capacity(partition_bounds.len());

            for (col_idx, right_expr) in self.on_right.iter().enumerate() {
                if let Some(column_bounds) = partition_bounds.get_column_bounds(col_idx) {
                    // Create predicate: col >= min AND col <= max
                    let min_expr = Arc::new(BinaryExpr::new(
                        Arc::clone(right_expr),
                        Operator::GtEq,
                        lit(column_bounds.min.clone()),
                    )) as Arc<dyn PhysicalExpr>;
                    let max_expr = Arc::new(BinaryExpr::new(
                        Arc::clone(right_expr),
                        Operator::LtEq,
                        lit(column_bounds.max.clone()),
                    )) as Arc<dyn PhysicalExpr>;
                    let range_expr =
                        Arc::new(BinaryExpr::new(min_expr, Operator::And, max_expr))
                            as Arc<dyn PhysicalExpr>;
                    column_predicates.push(range_expr);
                }
            }

            // Combine all column predicates for this partition with AND
            if !column_predicates.is_empty() {
                let partition_predicate = column_predicates
                    .into_iter()
                    .reduce(|acc, pred| {
                        Arc::new(BinaryExpr::new(acc, Operator::And, pred))
                            as Arc<dyn PhysicalExpr>
                    })
                    .unwrap();
                partition_predicates.push(partition_predicate);
            }
        }

        // Combine all partition predicates with OR
        let combined_predicate = partition_predicates
            .into_iter()
            .reduce(|acc, pred| {
                Arc::new(BinaryExpr::new(acc, Operator::Or, pred))
                    as Arc<dyn PhysicalExpr>
            })
            .unwrap_or_else(|| lit(true));

        Ok(combined_predicate)
    }

    /// Report information from a completed partition and update dynamic filter if all partitions are done
    ///
    /// This method coordinates the dynamic filter updates across all partitions. It stores the
    /// information from the current partition, increments the completion counter, and when all
    /// partitions have reported, updates the dynamic filter with the appropriate expressions.
    ///
    /// This method is async and uses a [`tokio::sync::Barrier`] to wait for all partitions
    /// to report their bounds. Once that occurs, the method will resolve for all callers and the
    /// dynamic filter will be updated exactly once.
    ///
    /// # Note
    ///
    /// As barriers are reusable, it is likely an error to call this method more times than the
    /// total number of partitions - as it can lead to pending futures that never resolve. We rely
    /// on correct usage from the caller rather than imposing additional checks here. If this is a concern,
    /// consider making the resulting future shared so the ready result can be reused.
    ///
    /// # Arguments
    /// * `left_side_partition_id` - The identifier for the **left-side** partition reporting its bounds
    /// * `partition_bounds` - The bounds computed by this partition (if enabled)
    /// * `left_hash_map` - The hashes from the build side for this partition (if enabled)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err if filter update failed
    pub(crate) async fn report_information(
        &self,
        left_side_partition_id: usize,
        partition_bounds: Option<Vec<ColumnBounds>>,
        left_hash_map: Option<Arc<dyn JoinHashMapType>>,
    ) -> Result<()> {
        // Store bounds in the accumulator - this runs once per partition
        if let Some(bounds) = partition_bounds {
            let mut guard = self.inner.lock();

            let should_push = if let Some(last_bound) = guard.bounds.last() {
                // In `PartitionMode::CollectLeft`, all streams on the left side share the same partition id (0).
                // Since this function can be called multiple times for that same partition, we must deduplicate
                // by checking against the last recorded bound.
                last_bound.partition != left_side_partition_id
            } else {
                true
            };

            if should_push {
                guard
                    .bounds
                    .push(PartitionBounds::new(left_side_partition_id, bounds));
            }
        }

        if let Some(left_hash_map) = left_hash_map {
            if left_hash_map.num_hashes() > 0 {
                let mut inner = self.inner.lock();
                // Store hash table at the correct partition index
                // This ensures hash_tables[partition_id] corresponds to that partition's data
                inner.hash_tables[left_side_partition_id] =
                    Some(Arc::clone(&left_hash_map));
            }
        }

        if self.barrier.wait().await.is_leader() {
            // All partitions have reported, so we can update the filter
            let inner = self.inner.lock();
            let maybe_bounds_expr = if !inner.bounds.is_empty() {
                Some(self.create_filter_from_partition_bounds(&inner.bounds)?)
            } else {
                None
            };

            // Check if we have any hash tables to use
            let has_hash_tables = inner.hash_tables.iter().any(|opt| opt.is_some());

            let maybe_hash_eval_expr = if has_hash_tables {
                // Create the appropriate strategy based on partition mode
                let strategy = match inner.partition_mode {
                    PartitionMode::CollectLeft => {
                        // Single hash table shared by all probe partitions
                        // Unwrap is safe because we checked has_hash_tables above
                        HashCheckStrategy::Single(Arc::clone(
                            inner.hash_tables[0].as_ref().unwrap(),
                        ))
                    }
                    PartitionMode::Partitioned => {
                        // Multiple hash tables - need partition routing
                        // Keep the Vec structure intact (with Options) so partition_id
                        // correctly maps to hash_tables[partition_id]
                        HashCheckStrategy::Partitioned {
                            hash_tables: inner.hash_tables.clone(),
                            num_partitions: inner.num_partitions,
                        }
                    }
                    PartitionMode::Auto => unreachable!(),
                };

                Some(Arc::new(HashComparePhysicalExpr::new(
                    self.on_right.clone(),
                    strategy,
                    self.random_state,
                )) as Arc<dyn PhysicalExpr>)
            } else {
                None
            };

            let filter_expr = match (maybe_bounds_expr, maybe_hash_eval_expr) {
                (None, Some(expr)) | (Some(expr), None) => expr,
                // Bounds collection and hash collection are current mutually exclusive, so this branch should not be
                // taken at the moment. However, if they are ever enabled together, this is how we would combine them.
                (Some(bounds_expr), Some(hash_eval_expr)) => Arc::new(BinaryExpr::new(
                    bounds_expr,
                    Operator::And,
                    Arc::clone(&hash_eval_expr),
                )),
                _ => return Ok(()),
            };

            self.dynamic_filter.update(filter_expr)?;
        }

        Ok(())
    }
}

impl fmt::Debug for SharedBuildAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SharedBuildAccumulator")
    }
}

/// Strategy for checking if a hash exists in the build side.
/// Determines whether to use a single hash table or route to multiple partitioned tables.
enum HashCheckStrategy {
    /// Single hash table (CollectLeft mode)
    /// All probe rows check against one shared hash table
    Single(Arc<dyn JoinHashMapType>),

    /// Multiple hash tables with partition routing (Partitioned mode)
    /// Probe rows are routed to the correct partition's hash table based on hash value.
    ///
    /// Uses `Option` because:
    /// 1. Empty partitions: A partition with no rows won't report a hash table
    /// 2. Zero hashes: A partition where `num_hashes() == 0` won't store a table (see line 437)
    ///    In evaluate(), we treat `None` as "no matches in this partition" (skip the row).
    Partitioned {
        hash_tables: Vec<Option<Arc<dyn JoinHashMapType>>>,
        num_partitions: usize,
    },
}

/// A [`PhysicalExpr`] that evaluates to a boolean array indicating which rows in a batch
/// have hashes that exist in the build-side hash tables.
///
/// This is currently used to implement hash-based dynamic filters in hash joins. That is,
/// this expression can be pushed down to the probe side of a hash join to filter out rows
/// that do not have a matching hash in the build side, allowing the scan to emit only the rows
/// that are guaranteed to have at least one match.
struct HashComparePhysicalExpr {
    /// Expressions that will be evaluated to compute hashes for filtering
    exprs: Vec<Arc<dyn PhysicalExpr>>,
    /// Strategy for checking hashes (single table or partitioned)
    strategy: HashCheckStrategy,
    /// Random state for hash computation
    random_state: &'static RandomState,
}

impl HashComparePhysicalExpr {
    pub fn new(
        exprs: Vec<Arc<dyn PhysicalExpr>>,
        strategy: HashCheckStrategy,
        random_state: &'static RandomState,
    ) -> Self {
        Self {
            exprs,
            strategy,
            random_state,
        }
    }
}

impl fmt::Debug for HashComparePhysicalExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HashComparePhysicalExpr [ {:?} ]", self.exprs)
    }
}

impl Hash for HashComparePhysicalExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.exprs.hash(state);
    }
}

impl PartialEq for HashComparePhysicalExpr {
    fn eq(&self, other: &Self) -> bool {
        // TODO: We limit comparison to uphold the equality property w.r.t `Hash`.
        // However, this means we may consider two expressions equal when they are actually not.
        // Since this is currently just used internally for dynamic filters, this may be acceptable
        // for now. If this expression is ever exposed more broadly, we should revisit this.
        self.exprs == other.exprs
    }
}

impl Eq for HashComparePhysicalExpr {}

impl fmt::Display for HashComparePhysicalExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let exprs = self
            .exprs
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "HashComparePhysicalExpr [ {exprs} ]")
    }
}

impl PhysicalExpr for HashComparePhysicalExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.exprs.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        // Clone the strategy - for Single it clones the Arc
        // for Partitioned it clones the Vec of Arcs
        let strategy = match &self.strategy {
            HashCheckStrategy::Single(table) => {
                HashCheckStrategy::Single(Arc::clone(table))
            }
            HashCheckStrategy::Partitioned {
                hash_tables,
                num_partitions,
            } => HashCheckStrategy::Partitioned {
                hash_tables: hash_tables.clone(),
                num_partitions: *num_partitions,
            },
        };

        Ok(Arc::new(Self {
            exprs: children,
            strategy,
            random_state: self.random_state,
        }))
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(
        &self,
        batch: &arrow::record_batch::RecordBatch,
    ) -> Result<ColumnarValue> {
        let num_rows = batch.num_rows();

        let expr_values = self
            .exprs
            .iter()
            .map(|col| col.evaluate(batch)?.into_array(num_rows))
            .collect::<Result<Vec<_>>>()?;

        // Compute join hashes for each probe row based on the evaluated expressions
        // These are used for the actual hash table lookup
        let mut join_hashes = vec![0; num_rows];
        create_hashes(&expr_values, self.random_state, &mut join_hashes)?;

        // Create a boolean array based on the strategy
        let mut buf = MutableBuffer::from_len_zeroed(bit_util::ceil(num_rows, 8));

        match &self.strategy {
            HashCheckStrategy::Single(hash_table) => {
                // Simple case: check against one hash table
                for (idx, join_hash) in join_hashes.into_iter().enumerate() {
                    if hash_table.contains_hash(&join_hash) {
                        bit_util::set_bit(buf.as_slice_mut(), idx);
                    }
                }
            }
            HashCheckStrategy::Partitioned {
                hash_tables,
                num_partitions,
            } => {
                // Partitioned case: use two-hash approach for correct partition routing
                //
                // Hash 1 (partition_hashes): Computed with same random state as RepartitionExec
                //         to determine which partition this row belongs to
                // Hash 2 (join_hashes): Computed with join's random state for actual lookup
                //         in the hash table
                let mut partition_hashes = vec![0; num_rows];
                create_hashes(
                    &expr_values,
                    &REPARTITION_RANDOM_STATE,
                    &mut partition_hashes,
                )?;

                for idx in 0..num_rows {
                    // Use partition hash to determine which hash table to check
                    let partition_id =
                        (partition_hashes[idx] % (*num_partitions as u64)) as usize;

                    // Use join hash to look up in that specific partition's hash table
                    let join_hash = join_hashes[idx];

                    // Check the hash table for that partition (if it exists)
                    if let Some(hash_table) = &hash_tables[partition_id] {
                        if hash_table.contains_hash(&join_hash) {
                            bit_util::set_bit(buf.as_slice_mut(), idx);
                        }
                    }
                }
            }
        }

        Ok(ColumnarValue::Array(Arc::new(
            BooleanArray::new_from_packed(buf, 0, num_rows),
        )))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}
