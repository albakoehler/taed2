import great_expectations as ge
import pandas as pd
from great_expectations.data_context.data_context import DataContext
from great_expectations.data_context.types.resource_identifiers import ExpectationSuiteIdentifier
from great_expectations.data_context.data_context import DataContext
import json

data = pd.read_csv("processed/X_train.csv")
df = ge.dataset.PandasDataset(data)
context = DataContext(context_root_dir='great_expectations')

###################################
#### Define text expectations: ####
###################################

# Presence of specific features
df.expect_table_columns_to_match_ordered_list(column_list=["text"])

# Missing values
df.expect_column_values_to_not_be_null(column="text")

# Type adherence
df.expect_column_values_to_be_of_type(column="text", type_="str")



# Create and save expectations suite
expectation_suite = df.get_expectation_suite(discard_failed_expectations=False)
expectation_suite_name = 'text expectations'
context.save_expectation_suite(expectation_suite=expectation_suite, expectation_suite_name=expectation_suite_name)


#####################################
#### Define target expectations: ####
#####################################

data = pd.read_csv("processed/y_train.csv")
df = ge.dataset.PandasDataset(data)
context = DataContext(context_root_dir='great_expectations')

#### Define sentiment expectations:

# Presence of specific features
df.expect_table_columns_to_match_ordered_list(column_list=["sentiment"])

# Missing values
df.expect_column_values_to_not_be_null(column="sentiment")

# Type adherence
df.expect_column_values_to_be_of_type(column="sentiment", type_="str")

# categorical value options
df.expect_column_values_to_be_in_set(column="sentiment", value_set=['Positive','Negative'])

#### Create and save expectations suite
expectation_suite = df.get_expectation_suite(discard_failed_expectations=False)
expectation_suite_name = 'sentiment expectations'
context.save_expectation_suite(expectation_suite=expectation_suite, expectation_suite_name=expectation_suite_name)
