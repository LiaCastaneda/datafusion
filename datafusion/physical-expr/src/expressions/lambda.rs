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

//! Physical lambda expression: [`LambdaExpr`]

use std::hash::Hash;
use std::sync::Arc;

use crate::{
    ScalarFunctionExpr,
    expressions::{Column, LambdaVariable},
    physical_expr::PhysicalExpr,
};
use arrow::{
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};
use datafusion_common::{
    HashMap, plan_err,
    tree_node::{Transformed, TreeNode, TreeNodeRecursion, TreeNodeVisitor},
};
use datafusion_common::{HashSet, Result, internal_err};
use datafusion_expr::ColumnarValue;

/// Represents a lambda with the given parameters names and body
#[derive(Debug, Eq, Clone)]
pub struct LambdaExpr {
    params: Vec<String>,
    body: Arc<dyn PhysicalExpr>,
    projected_body: Arc<dyn PhysicalExpr>,
    projection: Vec<usize>,
}

// Manually derive PartialEq and Hash to work around https://github.com/rust-lang/rust/issues/78808 [https://github.com/apache/datafusion/issues/13196]
impl PartialEq for LambdaExpr {
    fn eq(&self, other: &Self) -> bool {
        self.params.eq(&other.params) && self.body.eq(&other.body)
    }
}

impl Hash for LambdaExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.params.hash(state);
        self.body.hash(state);
    }
}

impl LambdaExpr {
    /// Create a new lambda expression with the given parameters and body
    pub fn try_new(params: Vec<String>, body: Arc<dyn PhysicalExpr>) -> Result<Self> {
        if !all_unique(&params) {
            return plan_err!(
                "lambda params must be unique, got ({})",
                params.join(", ")
            );
        }

        check_async_udf(&body)?;

        Ok(Self::new(params, body))
    }

    fn new(params: Vec<String>, body: Arc<dyn PhysicalExpr>) -> Self {
        let own_params: HashSet<String> = params.iter().cloned().collect();

        // Walk the body once to collect:
        // - used outer Column indices (for the captures projection)
        // - used own parameter names (with nested-lambda shadowing)
        let mut visitor = CollectUsedVisitor {
            own_params: &own_params,
            used_column_indices: HashSet::new(),
            used_param_names: HashSet::new(),
            shadow_stack: Vec::new(),
        };
        body.visit(&mut visitor).expect("visitor is infallible");
        let CollectUsedVisitor {
            used_column_indices,
            ..
        } = visitor;

        let mut projection: Vec<usize> = used_column_indices.into_iter().collect();
        projection.sort();

        // Map original outer-column indices → dense positions in the captures batch.
        let col_index_map: HashMap<usize, usize> = projection
            .iter()
            .copied()
            .enumerate()
            .map(|(new_idx, original)| (original, new_idx))
            .collect();

        // Map each *declared* param name → its slot in the merged evaluation batch.
        // Captures occupy 0..projection.len(); then all declared params follow in
        // declaration order (HOF passes all declared param closures in that order).
        // Unused params still occupy a slot — they're just never read by the body.
        let projected_body = {
            let captures_len = projection.len();
            let param_slot_map: HashMap<&str, usize> = params
                .iter()
                .enumerate()
                .map(|(i, name)| (name.as_str(), captures_len + i))
                .collect();

            Arc::clone(&body)
                .transform_down(|e| {
                    // Don't descend into nested lambdas — they are
                    // self-contained and were already rewritten by their own
                    // LambdaExpr::new call with their own param_slot_map.
                    if e.downcast_ref::<LambdaExpr>().is_some() {
                        return Ok(Transformed::new(e, false, TreeNodeRecursion::Jump));
                    }
                    if let Some(column) = e.downcast_ref::<Column>() {
                        let original = column.index();
                        let projected = *col_index_map.get(&original).unwrap();
                        if projected != original {
                            return Ok(Transformed::yes(Arc::new(Column::new(
                                column.name(),
                                projected,
                            ))));
                        }
                    } else if let Some(var) = e.downcast_ref::<LambdaVariable>() {
                        if let Some(&slot) = param_slot_map.get(var.name()) {
                            if slot != var.index() {
                                return Ok(Transformed::yes(Arc::new(
                                    LambdaVariable::new(slot, Arc::clone(var.field())),
                                )));
                            }
                        } else if let Some(&new_idx) = col_index_map.get(&var.index())
                            && new_idx != var.index()
                        {
                            return Ok(Transformed::yes(Arc::new(LambdaVariable::new(
                                new_idx,
                                Arc::clone(var.field()),
                            ))));
                        }
                    }
                    Ok(Transformed::no(e))
                })
                .expect("closure should be infallible")
                .data
        };

        Self {
            params,
            body,
            projected_body,
            projection,
        }
    }

    /// Get the lambda's params names
    pub fn params(&self) -> &[String] {
        &self.params
    }

    /// Get the lambda's body
    pub fn body(&self) -> &Arc<dyn PhysicalExpr> {
        &self.body
    }

    pub(crate) fn projection(&self) -> &[usize] {
        &self.projection
    }

    pub(crate) fn projected_body(&self) -> &Arc<dyn PhysicalExpr> {
        &self.projected_body
    }
}

/// Walks a lambda body once and collects:
/// - `used_column_indices`: every outer `Column` index referenced anywhere in
///   the tree (drives the captures projection).
/// - `used_param_names`: the subset of *this* lambda's own params referenced
///   by the body, accounting for nested-lambda shadowing (an inner lambda that
///   re-declares a param name shadows the outer one).
struct CollectUsedVisitor<'a> {
    own_params: &'a HashSet<String>,
    used_column_indices: HashSet<usize>,
    used_param_names: HashSet<String>,
    shadow_stack: Vec<HashSet<String>>,
}

impl TreeNodeVisitor<'_> for CollectUsedVisitor<'_> {
    type Node = Arc<dyn PhysicalExpr>;

    fn f_down(&mut self, node: &Self::Node) -> Result<TreeNodeRecursion> {
        if let Some(col) = node.downcast_ref::<Column>() {
            self.used_column_indices.insert(col.index());
        } else if let Some(var) = node.downcast_ref::<LambdaVariable>() {
            let name = var.name();
            let shadowed = self.shadow_stack.iter().any(|frame| frame.contains(name));
            if !shadowed && self.own_params.contains(name) {
                self.used_param_names.insert(name.to_string());
            } else {
                self.used_column_indices.insert(var.index());
            }
        } else if let Some(nested) = node.downcast_ref::<LambdaExpr>() {
            self.shadow_stack
                .push(nested.params.iter().cloned().collect());
        }
        Ok(TreeNodeRecursion::Continue)
    }

    fn f_up(&mut self, node: &Self::Node) -> Result<TreeNodeRecursion> {
        if node.downcast_ref::<LambdaExpr>().is_some() {
            self.shadow_stack.pop();
        }
        Ok(TreeNodeRecursion::Continue)
    }
}

impl std::fmt::Display for LambdaExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}) -> {}", self.params.join(", "), self.body)
    }
}

impl PhysicalExpr for LambdaExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Null)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(true)
    }

    fn evaluate(&self, _batch: &RecordBatch) -> Result<ColumnarValue> {
        internal_err!("LambdaExpr::evaluate() should not be called")
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.body]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let [body] = children.as_slice() else {
            return internal_err!(
                "LambdaExpr expects exactly 1 child, got {}",
                children.len()
            );
        };

        check_async_udf(body)?;

        Ok(Arc::new(Self::new(self.params.clone(), Arc::clone(body))))
    }

    fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) -> {}", self.params.join(", "), self.body)
    }
}

/// Create a lambda expression
pub fn lambda(
    params: impl IntoIterator<Item = impl Into<String>>,
    body: Arc<dyn PhysicalExpr>,
) -> Result<Arc<dyn PhysicalExpr>> {
    Ok(Arc::new(LambdaExpr::try_new(
        params.into_iter().map(Into::into).collect(),
        body,
    )?))
}

fn all_unique(params: &[String]) -> bool {
    match params.len() {
        0 | 1 => true,
        2 => params[0] != params[1],
        _ => {
            let mut set = HashSet::with_capacity(params.len());

            params.iter().all(|p| set.insert(p.as_str()))
        }
    }
}

fn check_async_udf(body: &Arc<dyn PhysicalExpr>) -> Result<()> {
    if body.exists(|expr| {
        Ok(expr
            .downcast_ref::<ScalarFunctionExpr>()
            .is_some_and(|udf| udf.fun().as_async().is_some()))
    })? {
        return plan_err!(
            "Async functions in lambdas aren't supported, see https://github.com/apache/datafusion/issues/22091"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::expressions::{Column, LambdaVariable, NoOp, lambda::lambda};
    use arrow::{
        array::{Int32Array, RecordBatch},
        datatypes::{DataType, Field, Schema},
    };
    use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion_expr::ColumnarValue;
    use datafusion_expr::LambdaArgument;
    use std::sync::Arc;

    use super::LambdaExpr;

    #[test]
    fn test_lambda_evaluate() {
        let lambda = lambda(["a"], Arc::new(NoOp::new())).unwrap();
        let batch = RecordBatch::new_empty(Arc::new(Schema::empty()));
        assert!(lambda.evaluate(&batch).is_err());
    }

    #[test]
    fn test_lambda_duplicate_name() {
        assert!(lambda(["a", "a"], Arc::new(NoOp::new())).is_err());
    }

    /// `(k, v) -> v`: only `v` (index 1) is referenced. The projection should
    /// contain only index 1, and `used_params` should be `["v"]` only.
    #[test]
    fn test_unused_first_param_projection() {
        let v_field = Arc::new(Field::new("v", DataType::Int32, true));
        // LambdaVariable index 1 = "v" as declared
        let body = Arc::new(LambdaVariable::new(1, Arc::clone(&v_field)));
        let lambda =
            LambdaExpr::try_new(vec!["k".to_string(), "v".to_string()], body).unwrap();

        // No outer columns captured — projection is empty.
        assert_eq!(lambda.projection(), &[] as &[usize]);
        // projected_body's LambdaVariable for `v` must be at slot 1
        // (captures_len=0 + declaration index 1), not compressed to slot 0.
        let mut found_slot = None;
        lambda
            .projected_body()
            .apply(|e| {
                if let Some(var) = e.downcast_ref::<LambdaVariable>()
                    && var.name() == "v"
                {
                    found_slot = Some(var.index());
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
        assert_eq!(found_slot, Some(1), "v must remain at declaration slot 1");
    }

    /// `(k, v) -> v` evaluated end-to-end: LambdaArgument must return the `v`
    /// column, not `k`. This is the exact runtime bug this fix addresses.
    #[test]
    fn test_unused_first_param_evaluates_correctly() {
        let k_field = Arc::new(Field::new("k", DataType::Int32, true));
        let v_field = Arc::new(Field::new("v", DataType::Int32, true));

        let body = Arc::new(LambdaVariable::new(1, Arc::clone(&v_field)));
        let lambda_expr =
            LambdaExpr::try_new(vec!["k".to_string(), "v".to_string()], body).unwrap();

        // HOF passes all declared params to LambdaArgument::new.
        // The fix is in projected_body: `v`'s LambdaVariable is remapped to
        // slot captures_len + 1 (its declaration position), not slot 0.
        let params = vec![k_field, v_field];
        let arg =
            LambdaArgument::new(params, Arc::clone(lambda_expr.projected_body()), None);

        let k_array = Arc::new(Int32Array::from(vec![1, 2, 3])) as _;
        let v_array = Arc::new(Int32Array::from(vec![10, 20, 30])) as _;

        // HOF passes all declared param closures in declaration order.
        // The projected body already maps `v`'s LambdaVariable to slot
        // captures_len + 1, so it reads `v`'s array, not `k`'s.
        let k_fn: &dyn Fn() -> datafusion_common::Result<arrow::array::ArrayRef> =
            &|| Ok(Arc::clone(&k_array));
        let v_fn: &dyn Fn() -> datafusion_common::Result<arrow::array::ArrayRef> =
            &|| Ok(Arc::clone(&v_array));
        let result = arg
            .evaluate(&[k_fn, v_fn], |_| unreachable!("no captures"))
            .unwrap();

        let ColumnarValue::Array(result_arr) = result else {
            panic!("expected array result");
        };
        let result_i32 = result_arr.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(result_i32.values(), &[10, 20, 30]);
    }

    /// Nested-lambda shadowing: `(k, v) -> col + (k, v2) -> k + v2 + v`.
    /// The inner lambda re-declares `k`, shadowing the outer one. The outer
    /// lambda's `CollectUsedVisitor` must not treat the inner `k` as a
    /// reference to the outer `k` — only outer `v` is actually used by the
    /// outer lambda's own body. This test verifies construction succeeds and
    /// the outer lambda's projected body is built correctly (no panic).
    #[test]
    fn test_shadowed_param_construction_succeeds() {
        let outer_k = Arc::new(Field::new("k", DataType::Int32, true));
        let outer_v = Arc::new(Field::new("v", DataType::Int32, true));
        let inner_v2 = Arc::new(Field::new("v2", DataType::Int32, true));

        let inner_body: Arc<dyn crate::PhysicalExpr> =
            Arc::new(crate::expressions::BinaryExpr::new(
                Arc::new(crate::expressions::BinaryExpr::new(
                    Arc::new(LambdaVariable::new(1, Arc::clone(&outer_k))),
                    datafusion_expr::Operator::Plus,
                    Arc::new(LambdaVariable::new(2, Arc::clone(&inner_v2))),
                )),
                datafusion_expr::Operator::Plus,
                Arc::new(LambdaVariable::new(0, Arc::clone(&outer_v))),
            ));
        let inner_lambda = Arc::new(
            LambdaExpr::try_new(vec!["k".to_string(), "v2".to_string()], inner_body)
                .unwrap(),
        );

        let outer_body: Arc<dyn crate::PhysicalExpr> =
            Arc::new(crate::expressions::BinaryExpr::new(
                Arc::new(Column::new("col", 0)),
                datafusion_expr::Operator::Plus,
                inner_lambda,
            ));

        // Must not panic — shadowing means outer's `k` is NOT flagged as used
        // and its param slot is assigned by declaration position correctly.
        LambdaExpr::try_new(vec!["k".to_string(), "v".to_string()], outer_body).unwrap();
    }
}
