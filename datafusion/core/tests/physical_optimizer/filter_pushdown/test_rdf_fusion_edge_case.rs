// Test case that reproduces the RDF-Fusion edge case where dynamic filters
// don't increment the inner Arc count when a custom DataSource extracts
// and stores them directly without wrapping in conjunction_opt.

use std::any::Any;
use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_catalog::memory::DataSourceExec;
use datafusion_common::{JoinType, Result, config::ConfigOptions};
use datafusion_datasource::source::DataSource;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::expressions::{DynamicFilterPhysicalExpr, col};
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion_physical_plan::filter_pushdown::{FilterPushdownPropagation, PushedDown};
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayFormatType, ExecutionPlan, Partitioning, Statistics,
};
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_optimizer::filter_pushdown::FilterPushdown;

/// Custom DataSource that mimics RDF-Fusion behavior:
/// - Directly extracts and stores dynamic filters WITHOUT wrapping them
/// - Does NOT call conjunction_opt or create wrapper expressions
#[derive(Debug, Clone)]
struct RdfLikeDataSource {
    schema: SchemaRef,
    // Store the dynamic filter directly (like RDF-Fusion does)
    stored_filter: Option<Arc<DynamicFilterPhysicalExpr>>,
}

impl RdfLikeDataSource {
    fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            stored_filter: None,
        }
    }
}

impl DataSource for RdfLikeDataSource {
    fn open(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Return empty stream for testing
        let stream = futures::stream::empty();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(f, "RdfLikeDataSource")
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(Arc::clone(&self.schema))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Statistics> {
        Ok(Statistics::new_unknown(&self.schema))
    }

    fn with_fetch(&self, _limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        None
    }

    fn fetch(&self) -> Option<usize> {
        None
    }

    fn try_swapping_with_projection(
        &self,
        _: &datafusion_physical_expr::projection::ProjectionExprs,
    ) -> Result<Option<Arc<dyn DataSource>>> {
        Ok(None)
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn DataSource>>> {
        // THIS IS THE KEY: Extract dynamic filter directly without wrapping
        // (mimics RDF-Fusion's MemStoragePredicateExpr::try_from behavior)
        let mut stored_filter = None;
        let mut push_results = vec![];

        for filter in &filters {
            // Check if it's a dynamic filter
            if filter.as_any().downcast_ref::<DynamicFilterPhysicalExpr>().is_some() {
                // OLD RDF-Fusion code (WITHOUT workaround):
                // Just clone the outer Arc - inner Arc count NOT incremented
                let cloned = Arc::clone(filter);

                // Downcast to concrete type
                if let Ok(dynamic_filter) = (cloned as Arc<dyn Any + Send + Sync>)
                    .downcast::<DynamicFilterPhysicalExpr>()
                {
                    stored_filter = Some(dynamic_filter);
                    push_results.push(PushedDown::Yes);
                    continue;
                }
            }
            push_results.push(PushedDown::No);
        }

        if stored_filter.is_some() {
            let new_source = Arc::new(RdfLikeDataSource {
                schema: Arc::clone(&self.schema),
                stored_filter,
            });
            Ok(FilterPushdownPropagation {
                filters: push_results,
                updated_node: Some(new_source),
            })
        } else {
            Ok(FilterPushdownPropagation {
                filters: push_results,
                updated_node: None,
            })
        }
    }
}

#[tokio::test]
async fn test_rdf_fusion_edge_case_is_used_returns_false() {
    let build_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, false),
        Field::new("b", DataType::Utf8, false),
    ]));

    let probe_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, false),
        Field::new("b", DataType::Utf8, false),
    ]));

    // Create build side (empty exec for testing)
    let build_exec = Arc::new(
        datafusion_physical_plan::empty::EmptyExec::new(Arc::clone(&build_schema))
    ) as Arc<dyn ExecutionPlan>;

    // Create probe side using RdfLikeDataSource (mimics RDF-Fusion)
    let rdf_source: Arc<dyn DataSource> = Arc::new(RdfLikeDataSource::new(Arc::clone(&probe_schema)));
    let probe_exec = Arc::new(DataSourceExec::new(rdf_source)) as Arc<dyn ExecutionPlan>;

    // Create HashJoin
    let on = vec![
        (
            col("a", &build_schema).unwrap(),
            col("a", &probe_schema).unwrap(),
        ),
        (
            col("b", &build_schema).unwrap(),
            col("b", &probe_schema).unwrap(),
        ),
    ];

    let hash_join = Arc::new(
        HashJoinExec::try_new(
            build_exec,
            probe_exec,
            on,
            None,
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            datafusion_common::NullEquality::NullEqualsNothing,
        )
        .unwrap(),
    ) as Arc<dyn ExecutionPlan>;

    // Apply filter pushdown optimization (Post phase for dynamic filters)
    let mut config = ConfigOptions::default();
    config.optimizer.enable_join_dynamic_filter_pushdown = true;

    let optimized_plan = FilterPushdown::new_post_optimization()
        .optimize(hash_join, &config)
        .unwrap();

    // Get the HashJoinExec from optimized plan
    let hash_join_exec = optimized_plan
        .as_any()
        .downcast_ref::<HashJoinExec>()
        .expect("Plan should be HashJoinExec");

    // Get the dynamic filter
    let dynamic_filter = hash_join_exec
        .dynamic_filter_for_test()
        .expect("Dynamic filter should be created");

    // THIS IS THE BUG: is_used() returns FALSE!
    // Because RdfLikeDataSource only cloned the outer Arc,
    // the inner Arc count is still 1
    println!("is_used() = {}", dynamic_filter.is_used());

    // This assertion will FAIL, demonstrating the edge case
    assert!(
        dynamic_filter.is_used(),
        "BUG: is_used() returns false because inner Arc was never cloned. \
         RdfLikeDataSource extracted the filter directly without calling \
         with_new_children, so the inner Arc count stayed at 1."
    );
}