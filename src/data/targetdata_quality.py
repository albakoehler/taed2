import great_expectations as gx
from ruamel import yaml
import pandas as pd
from great_expectations.data_context.data_context import DataContext
from great_expectations.core.batch import BatchRequest
from great_expectations.profile.user_configurable_profiler import UserConfigurableProfiler
from great_expectations.checkpoint.checkpoint import SimpleCheckpoint

context = DataContext(context_root_dir='great_expectations')
expectation_suite_name = "sentiment expectations"

#########################
### ytrain checkpoint ###
#########################

batch_request = {
    "datasource_name": "processed_data",
    "data_connector_name": "default_inferred_data_connector_name",
    "data_asset_name": "y_train.csv",
}

validator = context.get_validator(
    batch_request=BatchRequest(**batch_request),
    expectation_suite_name=expectation_suite_name
)

# Review and save our Expectation Suite
validator.get_expectation_suite(discard_failed_expectations=False)
validator.save_expectation_suite(discard_failed_expectations=False)

# Set up and run a Simple Checkpoint for ad hoc validation of our data
checkpoint_config = {
    "class_name": "y train checkpoint",
    "validations": [
        {
            "batch_request": batch_request,
            "expectation_suite_name": expectation_suite_name,
        }
    ],
}
checkpoint = SimpleCheckpoint(
    f"{validator.active_batch_definition.data_asset_name}_{expectation_suite_name}", context, **checkpoint_config
)
checkpoint_result = checkpoint.run()

# Build Data Docs
context.build_data_docs()

# Get the only validation_result_identifier from our SimpleCheckpoint run, and open Data Docs to that page
validation_result_identifier = checkpoint_result.list_validation_result_identifiers()[0]
context.open_data_docs(resource_identifier=validation_result_identifier)


#########################
### Xtest checkpoint ###
#########################

batch_request = {
    "datasource_name": "processed_data",
    "data_connector_name": "default_inferred_data_connector_name",
    "data_asset_name": "y_test.csv",
}

validator = context.get_validator(
    batch_request=BatchRequest(**batch_request),
    expectation_suite_name=expectation_suite_name
)

# Review and save our Expectation Suite
validator.get_expectation_suite(discard_failed_expectations=False)
validator.save_expectation_suite(discard_failed_expectations=False)

# Set up and run a Simple Checkpoint for ad hoc validation of our data
checkpoint_config = {
    "class_name": "y test checkpoint",
    "validations": [
        {
            "batch_request": batch_request,
            "expectation_suite_name": expectation_suite_name,
        }
    ],
}

checkpoint = SimpleCheckpoint(
    f"{validator.active_batch_definition.data_asset_name}_{expectation_suite_name}", context, **checkpoint_config
)
checkpoint_result = checkpoint.run()

# Build Data Docs
context.build_data_docs()

# Get the only validation_result_identifier from our SimpleCheckpoint run, and open Data Docs to that page
validation_result_identifier = checkpoint_result.list_validation_result_identifiers()[0]
context.open_data_docs(resource_identifier=validation_result_identifier)
