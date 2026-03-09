# Test Guidelines

## Debugging Test Failures

If a test fails because of an issue associated with the dataframe (e.g., missing data, unexpected values, weights summing to zero), it is likely that the test is correct and the dataframe has been corrupted. Always check with the user before modifying a test in this situation.
