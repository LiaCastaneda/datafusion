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

//! `MAP_AGG` aggregate implementation: [`MapAgg`]

use std::collections::{HashSet, VecDeque};
use std::mem::{size_of, size_of_val, take};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, MapArray, NullBufferBuilder, StructArray,
    new_empty_array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, FieldRef, Fields};

use datafusion_common::utils::{SingleRowListArrayBuilder, compare_rows, get_row_at_idx};
use datafusion_common::{Result, ScalarValue, exec_err};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::format_state_name;
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, EmitTo, GroupsAccumulator, Signature,
    Volatility,
};
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::nulls::filter_to_nulls;
use datafusion_functions_aggregate_common::merge_arrays::merge_ordered_arrays;
use datafusion_functions_aggregate_common::order::AggregateOrderSensitivity;
use datafusion_functions_aggregate_common::utils::ordering_fields;
use datafusion_macros::user_doc;
use datafusion_physical_expr_common::sort_expr::LexOrdering;

use crate::utils::{map_row_to_scalars, struct_to_rows};

make_udaf_expr_and_func!(
    MapAgg,
    map_agg,
    "Aggregate key-value pairs into a map",
    map_agg_udaf
);

#[user_doc(
    doc_section(label = "General Functions"),
    description = "Aggregate key-value pairs from two columns into a single map per group. Pairs with a NULL key are skipped; NULL values are retained. For duplicate keys, the first value in aggregate order wins; use ORDER BY to make that order deterministic.",
    syntax_example = "map_agg(key, value [ORDER BY expression])",
    sql_example = r#"
```sql
> SELECT map_agg(name, score) FROM scores GROUP BY department;
+-------------------------------+
| map_agg(name, score)          |
+-------------------------------+
| {Alice: 95, Bob: 87}          |
+-------------------------------+
```
"#,
    standard_argument(name = "key",),
    standard_argument(name = "value",)
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct MapAgg {
    signature: Signature,
    /// Whether the optimizer guarantees the input is already ordered by the
    /// `ORDER BY` clause, letting the accumulator skip its internal sort.
    is_input_pre_ordered: bool,
}

impl Default for MapAgg {
    fn default() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
            is_input_pre_ordered: false,
        }
    }
}

impl AggregateUDFImpl for MapAgg {
    fn name(&self) -> &str {
        "map_agg"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(map_type(&arg_types[0], &arg_types[1]))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let key_type = args.input_fields[0].data_type();
        let value_type = args.input_fields[1].data_type();

        let mut fields = vec![Arc::new(Field::new(
            format_state_name(args.name, "map_agg"),
            map_type(key_type, value_type),
            true,
        ))];

        if !args.ordering_fields.is_empty() {
            fields.push(Arc::new(Field::new_list(
                format_state_name(args.name, "map_agg_orderings"),
                Field::new_list_field(
                    DataType::Struct(Fields::from(args.ordering_fields.to_vec())),
                    true,
                ),
                false,
            )));
        }

        Ok(fields)
    }

    fn order_sensitivity(&self) -> AggregateOrderSensitivity {
        AggregateOrderSensitivity::SoftRequirement
    }

    fn with_beneficial_ordering(
        self: Arc<Self>,
        beneficial_ordering: bool,
    ) -> Result<Option<Arc<dyn AggregateUDFImpl>>> {
        Ok(Some(Arc::new(Self {
            signature: self.signature.clone(),
            is_input_pre_ordered: beneficial_ordering,
        })))
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let key_type = acc_args.expr_fields[0].data_type().clone();
        let value_type = acc_args.expr_fields[1].data_type().clone();

        let Some(ordering) = LexOrdering::new(acc_args.order_bys.to_vec()) else {
            return Ok(Box::new(MapAggAccumulator::new(key_type, value_type)));
        };

        let ordering_dtypes = ordering
            .iter()
            .map(|e| e.expr.data_type(acc_args.schema))
            .collect::<Result<Vec<_>>>()?;

        Ok(Box::new(OrderSensitiveMapAggAccumulator::new(
            key_type,
            value_type,
            ordering_dtypes,
            ordering,
            self.is_input_pre_ordered,
        )))
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        args.order_bys.is_empty()
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        let key_type = args.expr_fields[0].data_type().clone();
        let value_type = args.expr_fields[1].data_type().clone();
        Ok(Box::new(MapAggGroupsAccumulator::new(key_type, value_type)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn map_type(key_type: &DataType, value_type: &DataType) -> DataType {
    let fields = Fields::from(vec![
        Field::new("key", key_type.clone(), false),
        Field::new("value", value_type.clone(), true),
    ]);
    let entries_field = Arc::new(Field::new("entries", DataType::Struct(fields), false));
    DataType::Map(entries_field, false)
}

fn build_single_map(
    keys: Vec<ScalarValue>,
    values: Vec<ScalarValue>,
    key_type: &DataType,
    value_type: &DataType,
) -> Result<ArrayRef> {
    debug_assert_eq!(keys.len(), values.len());

    let fields = Fields::from(vec![
        Field::new("key", key_type.clone(), false),
        Field::new("value", value_type.clone(), true),
    ]);
    let entries_field = Arc::new(Field::new(
        "entries",
        DataType::Struct(fields.clone()),
        false,
    ));

    let len = keys.len();
    let key_array = if len == 0 {
        new_empty_array(key_type)
    } else {
        ScalarValue::iter_to_array(keys)?
    };
    let value_array = if len == 0 {
        new_empty_array(value_type)
    } else {
        ScalarValue::iter_to_array(values)?
    };

    let entries = StructArray::try_new(fields, vec![key_array, value_array], None)?;

    // With no entries, emit a single NULL map
    let nulls = (len == 0).then(|| NullBuffer::new_null(1));

    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0i32, len as i32]));
    Ok(Arc::new(MapArray::try_new(
        entries_field,
        offsets,
        entries,
        nulls,
        false,
    )?))
}

/// Plain accumulator used when there is no `ORDER BY`.
#[derive(Debug)]
pub struct MapAggAccumulator {
    key_type: DataType,
    value_type: DataType,
    keys: Vec<ScalarValue>,
    values: Vec<ScalarValue>,
}

impl MapAggAccumulator {
    pub fn new(key_type: DataType, value_type: DataType) -> Self {
        Self {
            key_type,
            value_type,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// De-duplicates parallel key/value vectors, keeping the first value seen
    /// for each key. Surviving keys retain their first-seen order.
    fn dedup_first_wins(
        keys: Vec<ScalarValue>,
        values: Vec<ScalarValue>,
    ) -> (Vec<ScalarValue>, Vec<ScalarValue>) {
        // First pass: mark each position that is the first occurrence of its key.
        let mut seen = HashSet::with_capacity(keys.len());
        let keep: Vec<bool> = keys.iter().map(|k| seen.insert(k)).collect();

        // Second pass: keep only the first-occurrence positions.
        let out_keys = keys
            .into_iter()
            .zip(&keep)
            .filter_map(|(k, &keep)| keep.then_some(k))
            .collect();
        let out_values = values
            .into_iter()
            .zip(&keep)
            .filter_map(|(v, &keep)| keep.then_some(v))
            .collect();
        (out_keys, out_values)
    }
}

impl Accumulator for MapAggAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() != 2 {
            return exec_err!("map_agg expects 2 columns, got {}", values.len());
        }
        let keys = &values[0];
        let vals = &values[1];

        for i in 0..keys.len() {
            // NULL keys cannot exist in a map; skip the whole pair.
            if keys.is_null(i) {
                continue;
            }
            self.keys
                .push(ScalarValue::try_from_array(keys, i)?.compacted());
            self.values
                .push(ScalarValue::try_from_array(vals, i)?.compacted());
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }
        let map_array = states[0].as_map();
        for row in 0..map_array.len() {
            if map_array.is_null(row) {
                continue;
            }
            let (keys, values) = map_row_to_scalars(map_array, row)?;
            self.keys.extend(keys);
            self.values.extend(values);
        }
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let (keys, values) =
            Self::dedup_first_wins(self.keys.clone(), self.values.clone());
        let map_array = build_single_map(keys, values, &self.key_type, &self.value_type)?;
        ScalarValue::try_from_array(&map_array, 0)
    }

    fn size(&self) -> usize {
        size_of_val(self) + ScalarValue::size_of_vec(&self.keys) - size_of_val(&self.keys)
            + ScalarValue::size_of_vec(&self.values)
            - size_of_val(&self.values)
            + self.key_type.size()
            - size_of_val(&self.key_type)
            + self.value_type.size()
            - size_of_val(&self.value_type)
    }
}

struct OrderSensitiveMapAggRows {
    keys: Vec<ScalarValue>,
    values: Vec<ScalarValue>,
    ordering_values: Vec<Vec<ScalarValue>>,
}

/// Accumulator used when `map_agg` has an `ORDER BY`. Stores the ordering column
/// values alongside each pair so the input can be globally sorted (across
/// partitions).
#[derive(Debug)]
pub struct OrderSensitiveMapAggAccumulator {
    key_type: DataType,
    value_type: DataType,
    keys: Vec<ScalarValue>,
    values: Vec<ScalarValue>,
    /// Ordering-expression values for each pair, parallel to `keys`/`values`.
    ordering_values: Vec<Vec<ScalarValue>>,
    ordering_dtypes: Vec<DataType>,
    ordering_req: LexOrdering,
    is_input_pre_ordered: bool,
}

impl OrderSensitiveMapAggAccumulator {
    pub fn new(
        key_type: DataType,
        value_type: DataType,
        ordering_dtypes: Vec<DataType>,
        ordering_req: LexOrdering,
        is_input_pre_ordered: bool,
    ) -> Self {
        Self {
            key_type,
            value_type,
            keys: Vec::new(),
            values: Vec::new(),
            ordering_values: Vec::new(),
            ordering_dtypes,
            ordering_req,
            is_input_pre_ordered,
        }
    }

    fn sort_options(&self) -> Vec<SortOptions> {
        self.ordering_req.iter().map(|s| s.options).collect()
    }

    /// Sorts the accumulated pairs by their ordering values, then applies
    /// first-wins de-duplication. Returns the surviving keys, values, and
    /// ordering values, all aligned so they describe the same rows.
    fn sorted_deduped(&self) -> Result<OrderSensitiveMapAggRows> {
        let mut rows: Vec<usize> = (0..self.keys.len()).collect();

        if !self.is_input_pre_ordered {
            let sort_options = self.sort_options();
            let mut cmp_err = Ok(());
            rows.sort_by(|&a, &b| {
                compare_rows(
                    &self.ordering_values[a],
                    &self.ordering_values[b],
                    &sort_options,
                )
                .unwrap_or_else(|e| {
                    cmp_err = Err(e);
                    std::cmp::Ordering::Equal
                })
            });
            cmp_err?;
        }

        // Keep the first occurrence of each key in sorted order, and project the
        // keys, values, and ordering values through the same surviving indices
        // so all three stay aligned.
        let mut seen = HashSet::with_capacity(rows.len());
        let mut keys = Vec::new();
        let mut values = Vec::new();
        let mut ordering_values = Vec::new();
        for &i in &rows {
            if seen.insert(&self.keys[i]) {
                keys.push(self.keys[i].clone());
                values.push(self.values[i].clone());
                ordering_values.push(self.ordering_values[i].clone());
            }
        }

        Ok(OrderSensitiveMapAggRows {
            keys,
            values,
            ordering_values,
        })
    }

    /// Builds the `List<Struct<ordering...>>` state column from the given
    /// ordering values. These must be the de-duplicated ordering values that
    /// align with the map state, so both pieces of state describe the same rows.
    fn evaluate_orderings(
        &self,
        ordering_values: &[Vec<ScalarValue>],
    ) -> Result<ScalarValue> {
        let fields = ordering_fields(&self.ordering_req, &self.ordering_dtypes);
        let struct_field = Fields::from(fields.clone());

        let mut column_wise: Vec<ArrayRef> = Vec::with_capacity(fields.len());
        for (col_idx, field) in fields.iter().enumerate() {
            if ordering_values.is_empty() {
                column_wise.push(new_empty_array(field.data_type()));
            } else {
                let col_vals = ordering_values.iter().map(|row| row[col_idx].clone());
                column_wise.push(ScalarValue::iter_to_array(col_vals)?);
            }
        }

        let struct_array = StructArray::try_new(struct_field, column_wise, None)?;
        Ok(SingleRowListArrayBuilder::new(Arc::new(struct_array)).build_list_scalar())
    }
}

impl Accumulator for OrderSensitiveMapAggAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() < 2 {
            return exec_err!("map_agg expects at least 2 columns, got {}", values.len());
        }
        let keys = &values[0];
        let vals = &values[1];
        let ordering_cols = &values[2..];

        for i in 0..keys.len() {
            // NULL keys cannot exist in a map; skip the whole pair.
            if keys.is_null(i) {
                continue;
            }
            self.keys
                .push(ScalarValue::try_from_array(keys, i)?.compacted());
            self.values
                .push(ScalarValue::try_from_array(vals, i)?.compacted());
            self.ordering_values.push(
                get_row_at_idx(ordering_cols, i)?
                    .into_iter()
                    .map(|v| v.compacted())
                    .collect(),
            );
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }
        if states.len() != 2 {
            return exec_err!(
                "map_agg ordered merge expects 2 state columns, got {}",
                states.len()
            );
        }

        let map_array = states[0].as_map();
        let orderings = states[1].as_list::<i32>();

        // Each partition contributes one map row plus a parallel list of
        // ordering-value structs. Collect them, then merge by ordering.
        let mut partition_keys: Vec<VecDeque<ScalarValue>> =
            vec![take(&mut self.keys).into()];
        let mut partition_orderings: Vec<VecDeque<Vec<ScalarValue>>> =
            vec![take(&mut self.ordering_values).into()];
        let mut partition_values: Vec<VecDeque<ScalarValue>> =
            vec![take(&mut self.values).into()];

        // Push keys and values from each partition's state into the merge buffers.
        for row in 0..map_array.len() {
            if map_array.is_null(row) {
                continue;
            }
            let (keys, values) = map_row_to_scalars(map_array, row)?;
            let ord_vals = struct_to_rows(orderings.value(row).as_struct())?;

            partition_keys.push(keys.into());
            partition_values.push(values.into());
            partition_orderings.push(ord_vals.into());
        }

        // Merge keys and values along the ordering. `merge_ordered_arrays`
        // merges a single value stream; run it once for keys and once for
        // values using the same ordering inputs so they stay aligned.
        let sort_options = self.sort_options();

        let (merged_keys, merged_orderings) = merge_ordered_arrays(
            &mut partition_keys,
            &mut partition_orderings.clone(),
            &sort_options,
        )?;
        let (merged_values, _) = merge_ordered_arrays(
            &mut partition_values,
            &mut partition_orderings,
            &sort_options,
        )?;

        self.keys = merged_keys;
        self.values = merged_values;
        self.ordering_values = merged_orderings;
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let rows = self.sorted_deduped()?;
        let orderings = self.evaluate_orderings(&rows.ordering_values)?;
        let map_array =
            build_single_map(rows.keys, rows.values, &self.key_type, &self.value_type)?;
        Ok(vec![ScalarValue::try_from_array(&map_array, 0)?, orderings])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let rows = self.sorted_deduped()?;
        let map_array =
            build_single_map(rows.keys, rows.values, &self.key_type, &self.value_type)?;
        ScalarValue::try_from_array(&map_array, 0)
    }

    fn size(&self) -> usize {
        let mut total = size_of_val(self) + ScalarValue::size_of_vec(&self.keys)
            - size_of_val(&self.keys)
            + ScalarValue::size_of_vec(&self.values)
            - size_of_val(&self.values)
            + self.key_type.size()
            - size_of_val(&self.key_type)
            + self.value_type.size()
            - size_of_val(&self.value_type);

        // ordering_values: Vec spine plus the heap owned by each row's scalars.
        total += size_of::<Vec<ScalarValue>>() * self.ordering_values.capacity();
        for row in &self.ordering_values {
            total += ScalarValue::size_of_vec(row) - size_of_val(row);
        }

        // ordering_dtypes: Vec spine plus the heap owned by each DataType.
        total += size_of::<DataType>() * self.ordering_dtypes.capacity();
        for dtype in &self.ordering_dtypes {
            total += dtype.size() - size_of_val(dtype);
        }

        total
    }
}

/// Vectorized groups accumulator for the non-ORDER BY case of `map_agg`.
///
/// Each input batch is stored as a `StructArray` with two columns (key, value), together with a
/// `(group_idx, row_idx)` entry list. `evaluate` uses a counting sort to
/// scatter entries into group order and then calls `arrow::compute::interleave`
/// to build the per-group flat key/value arrays that back a `MapArray`.
struct MapAggGroupsAccumulator {
    key_type: DataType,
    value_type: DataType,
    /// Each element is a `StructArray` with columns `[key, value]`.
    batches: Vec<ArrayRef>,
    /// Per-batch `(group_idx, row_idx)` pairs (parallel to `batches`).
    batch_entries: Vec<Vec<(u32, u32)>>,
    num_groups: usize,
}

impl MapAggGroupsAccumulator {
    fn new(key_type: DataType, value_type: DataType) -> Self {
        Self {
            key_type,
            value_type,
            batches: Vec::new(),
            batch_entries: Vec::new(),
            num_groups: 0,
        }
    }

    fn entries_field(&self) -> Arc<Field> {
        Arc::new(Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", self.key_type.clone(), false),
                Field::new("value", self.value_type.clone(), true),
            ])),
            false,
        ))
    }

    fn clear_state(&mut self) {
        self.batches = Vec::new();
        self.batch_entries = Vec::new();
        self.num_groups = 0;
    }

    fn compact_retained_state(&mut self, emit_groups: usize) -> Result<()> {
        let emit_groups = emit_groups as u32;
        let old_batches = take(&mut self.batches);
        let old_entries = take(&mut self.batch_entries);

        let mut batches = Vec::new();
        let mut batch_entries = Vec::new();

        for (batch, entries) in old_batches.into_iter().zip(old_entries) {
            let retained_len = entries.iter().filter(|(g, _)| *g >= emit_groups).count();

            if retained_len == 0 {
                continue;
            }

            if retained_len == entries.len() {
                let mut retained = entries;
                for (g, _) in &mut retained {
                    *g -= emit_groups;
                }
                retained.shrink_to_fit();
                batches.push(batch);
                batch_entries.push(retained);
                continue;
            }

            let mut retained_entries = Vec::with_capacity(retained_len);
            let mut retained_rows = Vec::with_capacity(retained_len);

            for (g, r) in entries {
                if g >= emit_groups {
                    retained_entries.push((g - emit_groups, retained_rows.len() as u32));
                    retained_rows.push(r);
                }
            }

            let batch = {
                let retained_rows = arrow::array::UInt32Array::from(retained_rows);
                arrow::compute::take(batch.as_ref(), &retained_rows, None)?
            };

            batches.push(batch);
            batch_entries.push(retained_entries);
        }

        self.batches = batches;
        self.batch_entries = batch_entries;
        self.num_groups -= emit_groups as usize;

        Ok(())
    }

    /// Scatter entries for `emit_groups` groups into group order using a
    /// counting sort and call `interleave` on the struct batches. Returns the
    /// flat `StructArray` and the per-group offsets/nulls for the `MapArray`.
    fn build_map_array(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let emit_groups = match emit_to {
            EmitTo::All => self.num_groups,
            EmitTo::First(n) => n,
        };

        // Step 1: count entries per group.
        let mut counts = vec![0u32; emit_groups];
        for entries in &self.batch_entries {
            for &(g, _) in entries {
                let g = g as usize;
                if g < emit_groups {
                    counts[g] += 1;
                }
            }
        }

        // Step 2: prefix sum → offsets, null flags, write positions.
        let mut offsets = Vec::<i32>::with_capacity(emit_groups + 1);
        offsets.push(0);
        let mut nulls_builder = NullBufferBuilder::new(emit_groups);
        let mut write_positions = Vec::with_capacity(emit_groups);
        let mut cur_offset = 0u32;
        for &count in &counts {
            if count == 0 {
                nulls_builder.append_null();
            } else {
                nulls_builder.append_non_null();
            }
            write_positions.push(cur_offset);
            cur_offset += count;
            offsets.push(cur_offset as i32);
        }
        let total_rows = cur_offset as usize;

        // Step 3: counting sort → interleave.
        let entries_field = self.entries_field();
        let flat_entries = if total_rows == 0 {
            new_empty_array(entries_field.data_type())
        } else {
            let mut interleave_indices = vec![(0usize, 0usize); total_rows];
            for (batch_idx, entries) in self.batch_entries.iter().enumerate() {
                for &(g, r) in entries {
                    let g = g as usize;
                    if g < emit_groups {
                        let wp = write_positions[g] as usize;
                        interleave_indices[wp] = (batch_idx, r as usize);
                        write_positions[g] += 1;
                    }
                }
            }
            let sources: Vec<&dyn Array> =
                self.batches.iter().map(|b| b.as_ref()).collect();
            arrow::compute::interleave(&sources, &interleave_indices)?
        };

        // Step 4: dedup each group (first-wins within the group's flat slice).
        // For each group we slice the flat entries, dedup, then append to output.
        let flat_struct = flat_entries
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                datafusion_common::internal_datafusion_err!(
                    "interleave result must be StructArray"
                )
            })?;
        let flat_keys = flat_struct.column(0);
        let flat_values_col = flat_struct.column(1);

        // Build deduped per-group key and value ScalarValue vecs.
        let mut deduped_keys: Vec<ScalarValue> = Vec::with_capacity(total_rows);
        let mut deduped_values: Vec<ScalarValue> = Vec::with_capacity(total_rows);
        let mut deduped_offsets = Vec::<i32>::with_capacity(emit_groups + 1);
        deduped_offsets.push(0);
        let mut deduped_nulls = NullBufferBuilder::new(emit_groups);
        let mut prev_deduped = 0i32;

        for g in 0..emit_groups {
            let start = offsets[g] as usize;
            let end = offsets[g + 1] as usize;
            let mut seen: HashSet<ScalarValue> = HashSet::with_capacity(end - start);
            for row in start..end {
                let k = ScalarValue::try_from_array(flat_keys, row)?;
                if k.is_null() {
                    continue;
                }
                if seen.insert(k.clone()) {
                    deduped_keys.push(k);
                    deduped_values
                        .push(ScalarValue::try_from_array(flat_values_col, row)?);
                }
            }
            let new_end = deduped_keys.len() as i32;
            if new_end == prev_deduped {
                deduped_nulls.append_null();
            } else {
                deduped_nulls.append_non_null();
            }
            prev_deduped = new_end;
            deduped_offsets.push(new_end);
        }

        // Step 5: build the MapArray output.
        match emit_to {
            EmitTo::All => self.clear_state(),
            EmitTo::First(_) => self.compact_retained_state(emit_groups)?,
        }

        let fields = Fields::from(vec![
            Field::new("key", self.key_type.clone(), false),
            Field::new("value", self.value_type.clone(), true),
        ]);
        let n = deduped_keys.len();
        let key_array = if n == 0 {
            new_empty_array(&self.key_type)
        } else {
            ScalarValue::iter_to_array(deduped_keys)?
        };
        let value_array = if n == 0 {
            new_empty_array(&self.value_type)
        } else {
            ScalarValue::iter_to_array(deduped_values)?
        };
        let flat_struct_out =
            StructArray::try_new(fields, vec![key_array, value_array], None)?;

        let offsets_buf = OffsetBuffer::new(ScalarBuffer::from(deduped_offsets));
        let map_array = MapArray::try_new(
            entries_field,
            offsets_buf,
            flat_struct_out,
            deduped_nulls.finish(),
            false,
        )?;
        Ok(Arc::new(map_array))
    }
}

impl GroupsAccumulator for MapAggGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 2, "map_agg expects 2 columns");
        let keys = &values[0];
        let vals = &values[1];

        self.num_groups = self.num_groups.max(total_num_groups);

        let mut entries = Vec::new();
        for (row_idx, &group_idx) in group_indices.iter().enumerate() {
            if opt_filter.is_some_and(|f| f.is_null(row_idx) || !f.value(row_idx)) {
                continue;
            }
            // NULL keys are not allowed in a map; skip the pair.
            if keys.is_null(row_idx) {
                continue;
            }
            entries.push((group_idx as u32, row_idx as u32));
        }

        if !entries.is_empty() {
            let fields = Fields::from(vec![
                Field::new("key", self.key_type.clone(), false),
                Field::new("value", self.value_type.clone(), true),
            ]);
            let struct_arr = StructArray::try_new(
                fields,
                vec![Arc::clone(keys), Arc::clone(vals)],
                None,
            )?;
            self.batches.push(Arc::new(struct_arr));
            self.batch_entries.push(entries);
        }

        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        self.build_map_array(emit_to)
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        // Emit a MapArray as state; merge_batch re-ingests it via merge.
        Ok(vec![self.build_map_array(emit_to)?])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "one state column for merge");
        let map_array = values[0].as_map();

        self.num_groups = self.num_groups.max(total_num_groups);

        // Expand each map row into individual (group_idx, entry_pos) pairs,
        // storing the underlying entries StructArray as a single batch.
        let entries_struct = map_array.entries();
        let offsets = map_array.offsets();

        let mut entries = Vec::new();
        for (row_idx, &group_idx) in group_indices.iter().enumerate() {
            if map_array.is_null(row_idx) {
                continue;
            }
            let start = offsets[row_idx] as u32;
            let end = offsets[row_idx + 1] as u32;
            for pos in start..end {
                entries.push((group_idx as u32, pos));
            }
        }

        if !entries.is_empty() {
            self.batches
                .push(Arc::new(entries_struct.clone()) as ArrayRef);
            self.batch_entries.push(entries);
        }

        Ok(())
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        assert_eq!(values.len(), 2, "map_agg expects 2 columns");
        let keys = &values[0];
        let vals = &values[1];

        // Build a 1-entry MapArray per row so merge_batch can re-ingest it.
        let n = keys.len();
        let entries_field = Arc::new(Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", self.key_type.clone(), false),
                Field::new("value", self.value_type.clone(), true),
            ])),
            false,
        ));
        let fields = Fields::from(vec![
            Field::new("key", self.key_type.clone(), false),
            Field::new("value", self.value_type.clone(), true),
        ]);

        let struct_arr =
            StructArray::try_new(fields, vec![Arc::clone(keys), Arc::clone(vals)], None)?;
        let offsets = OffsetBuffer::from_repeated_length(1, n);

        // Null-key rows and filtered rows become null map entries.
        let key_nulls = keys.logical_nulls();
        let filter_nulls = opt_filter.map(filter_to_nulls);
        let nulls = NullBuffer::union(key_nulls.as_ref(), filter_nulls.as_ref());

        let map_array =
            MapArray::try_new(entries_field, offsets, struct_arr, nulls, false)?;
        Ok(vec![Arc::new(map_array)])
    }

    fn supports_convert_to_state(&self) -> bool {
        true
    }

    fn size(&self) -> usize {
        self.batches
            .iter()
            .map(|arr| arr.to_data().get_slice_memory_size().unwrap_or_default())
            .sum::<usize>()
            + self.batches.capacity() * size_of::<ArrayRef>()
            + self
                .batch_entries
                .iter()
                .map(|e| e.capacity() * size_of::<(u32, u32)>())
                .sum::<usize>()
            + self.batch_entries.capacity() * size_of::<Vec<(u32, u32)>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{Int32Type, Schema};
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;

    fn make_acc() -> MapAggAccumulator {
        MapAggAccumulator::new(DataType::Utf8, DataType::Int32)
    }

    fn str_arr(vals: &[&str]) -> ArrayRef {
        Arc::new(StringArray::from(vals.to_vec()))
    }

    fn int_arr(vals: &[i32]) -> ArrayRef {
        Arc::new(Int32Array::from(vals.to_vec()))
    }

    fn str_arr_nullable(vals: Vec<Option<&str>>) -> ArrayRef {
        Arc::new(StringArray::from(vals))
    }

    fn extract_map(sv: ScalarValue) -> Vec<(String, Option<i32>)> {
        let ScalarValue::Map(arr) = sv else {
            panic!("expected ScalarValue::Map, got {sv:?}");
        };
        let entries = arr.value(0);
        let entries = entries.as_any().downcast_ref::<StructArray>().unwrap();
        let keys = entries.column(0).as_string::<i32>();
        let vals = entries.column(1).as_primitive::<Int32Type>();
        (0..keys.len())
            .map(|i| {
                let v = if vals.is_null(i) {
                    None
                } else {
                    Some(vals.value(i))
                };
                (keys.value(i).to_string(), v)
            })
            .collect()
    }

    #[test]
    fn collects_distinct_pairs_in_order() -> Result<()> {
        let mut acc = make_acc();
        acc.update_batch(&[str_arr(&["a", "b", "c"]), int_arr(&[1, 2, 3])])?;
        let pairs = extract_map(acc.evaluate()?);
        assert_eq!(
            pairs,
            vec![
                ("a".into(), Some(1)),
                ("b".into(), Some(2)),
                ("c".into(), Some(3)),
            ]
        );
        Ok(())
    }

    #[test]
    fn null_key_skipped() -> Result<()> {
        let mut acc = make_acc();
        // A null-key pair is dropped (matches Trino map_agg).
        acc.update_batch(&[
            str_arr_nullable(vec![Some("a"), None, Some("c")]),
            int_arr(&[1, 2, 3]),
        ])?;
        let pairs = extract_map(acc.evaluate()?);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "a");
        assert_eq!(pairs[1].0, "c");
        Ok(())
    }

    #[test]
    fn null_value_retained() -> Result<()> {
        let mut acc = make_acc();
        let vals: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]));
        acc.update_batch(&[str_arr(&["a", "b", "c"]), vals])?;
        let pairs = extract_map(acc.evaluate()?);
        assert_eq!(pairs[1], ("b".into(), None));
        Ok(())
    }

    #[test]
    fn duplicate_key_first_wins() -> Result<()> {
        let mut acc = make_acc();
        // a appears twice; the first value (1) wins, later one is dropped.
        acc.update_batch(&[str_arr(&["a", "b", "a"]), int_arr(&[1, 2, 3])])?;
        let pairs = extract_map(acc.evaluate()?);
        assert_eq!(pairs, vec![("a".into(), Some(1)), ("b".into(), Some(2))]);
        Ok(())
    }

    #[test]
    fn empty_produces_null_map() -> Result<()> {
        let mut acc = make_acc();
        let ScalarValue::Map(arr) = acc.evaluate()? else {
            panic!("expected ScalarValue::Map");
        };
        assert_eq!(arr.len(), 1);
        assert!(arr.is_null(0));
        Ok(())
    }

    #[test]
    fn merge_two_partitions() -> Result<()> {
        let mut acc1 = make_acc();
        let mut acc2 = make_acc();

        acc1.update_batch(&[str_arr(&["a", "b"]), int_arr(&[1, 2])])?;
        acc2.update_batch(&[str_arr(&["c", "d"]), int_arr(&[3, 4])])?;

        let state2 = acc2
            .state()?
            .into_iter()
            .map(|sv| sv.to_array())
            .collect::<Result<Vec<_>>>()?;
        acc1.merge_batch(&state2)?;

        let pairs = extract_map(acc1.evaluate()?);
        assert_eq!(pairs.len(), 4);
        Ok(())
    }

    #[test]
    fn merge_duplicate_key_across_partitions_first_wins() -> Result<()> {
        let mut acc1 = make_acc();
        let mut acc2 = make_acc();

        acc1.update_batch(&[str_arr(&["a"]), int_arr(&[1])])?;
        acc2.update_batch(&[str_arr(&["a"]), int_arr(&[2])])?;

        let state2 = acc2
            .state()?
            .into_iter()
            .map(|sv| sv.to_array())
            .collect::<Result<Vec<_>>>()?;
        acc1.merge_batch(&state2)?;

        let pairs = extract_map(acc1.evaluate()?);
        // acc1's pair comes first in the merged stream, so value 1 wins.
        assert_eq!(pairs, vec![("a".into(), Some(1))]);
        Ok(())
    }

    /// Builds an order-sensitive accumulator that orders by a single Int32
    /// column `ord` (column index 2 in the update batch).
    fn make_ordered_acc(descending: bool) -> OrderSensitiveMapAggAccumulator {
        let schema = Schema::new(vec![
            Field::new("k", DataType::Utf8, true),
            Field::new("v", DataType::Int32, true),
            Field::new("ord", DataType::Int32, true),
        ]);
        let ord_expr = Arc::new(Column::new_with_schema("ord", &schema).unwrap());
        let ordering = LexOrdering::new(vec![PhysicalSortExpr::new(
            ord_expr,
            SortOptions::new(descending, false),
        )])
        .unwrap();
        OrderSensitiveMapAggAccumulator::new(
            DataType::Utf8,
            DataType::Int32,
            vec![DataType::Int32],
            ordering,
            false,
        )
    }

    #[test]
    fn ordered_dup_key_asc_first_wins() -> Result<()> {
        let mut acc = make_ordered_acc(false);
        // key "a" twice: ord=10 -> v=1, ord=20 -> v=2. ASC sort puts v=1
        // (ord=10) first, so first-wins keeps v=1.
        acc.update_batch(&[str_arr(&["a", "a"]), int_arr(&[1, 2]), int_arr(&[10, 20])])?;
        let pairs = extract_map(acc.evaluate()?);
        assert_eq!(pairs, vec![("a".into(), Some(1))]);
        Ok(())
    }

    #[test]
    fn ordered_dup_key_desc_flips_winner() -> Result<()> {
        let mut acc = make_ordered_acc(true);
        // Same input, DESC sort puts v=2 (ord=20) first, so first-wins keeps v=2.
        acc.update_batch(&[str_arr(&["a", "a"]), int_arr(&[1, 2]), int_arr(&[10, 20])])?;
        let pairs = extract_map(acc.evaluate()?);
        assert_eq!(pairs, vec![("a".into(), Some(2))]);
        Ok(())
    }

    #[test]
    fn ordered_merge_two_partitions() -> Result<()> {
        // Partition 1 sees ord=20, partition 2 sees ord=10 for the same key.
        // After merge + ASC sort, the ord=10 row is first, so its value wins.
        let mut acc1 = make_ordered_acc(false);
        let mut acc2 = make_ordered_acc(false);

        acc1.update_batch(&[str_arr(&["a"]), int_arr(&[2]), int_arr(&[20])])?;
        acc2.update_batch(&[str_arr(&["a"]), int_arr(&[1]), int_arr(&[10])])?;

        let state2 = acc2
            .state()?
            .into_iter()
            .map(|sv| sv.to_array())
            .collect::<Result<Vec<_>>>()?;
        acc1.merge_batch(&state2)?;

        let pairs = extract_map(acc1.evaluate()?);
        assert_eq!(pairs, vec![("a".into(), Some(1))]);
        Ok(())
    }

    #[test]
    fn ordered_merge_with_intra_partition_duplicates() -> Result<()> {
        let mut acc1 = make_ordered_acc(false);
        let mut acc2 = make_ordered_acc(false);

        // P1: a -> {1@10, 3@30}, b -> {2@20}  => after dedup: a=1, b=2
        acc1.update_batch(&[
            str_arr(&["a", "b", "a"]),
            int_arr(&[1, 2, 3]),
            int_arr(&[10, 20, 30]),
        ])?;
        // P2: a -> {4@40, 5@50}, c -> {6@60}  => after dedup: a=4, c=6
        acc2.update_batch(&[
            str_arr(&["a", "a", "c"]),
            int_arr(&[4, 5, 6]),
            int_arr(&[40, 50, 60]),
        ])?;

        let state2 = acc2
            .state()?
            .into_iter()
            .map(|sv| sv.to_array())
            .collect::<Result<Vec<_>>>()?;
        acc1.merge_batch(&state2)?;

        let mut pairs = extract_map(acc1.evaluate()?);
        pairs.sort();
        // Global first-wins by ord: a@10=1, b@20=2, c@60=6.
        assert_eq!(
            pairs,
            vec![
                ("a".into(), Some(1)),
                ("b".into(), Some(2)),
                ("c".into(), Some(6)),
            ]
        );
        Ok(())
    }
}
