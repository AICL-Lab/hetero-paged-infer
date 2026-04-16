# Testing Specifications: Heterogeneous Inference Engine

## Overview

This document defines the Behavior-Driven Development (BDD) testing specifications for the Heterogeneous Inference Engine.

## Test Categories

### 1. Unit Tests

Each module must have unit tests covering core functionality and edge cases.

#### KV Cache Manager Tests

```gherkin
Feature: KV Cache Management

  Scenario: Allocate blocks for new sequence
    Given a KV cache manager with 100 free blocks
    When a new sequence with 20 tokens is submitted
    Then 2 logical blocks should be allocated (block_size = 16)
    And each logical block should map to a distinct physical block

  Scenario: Free sequence blocks
    Given a sequence with 3 allocated blocks
    When the sequence completes
    Then all 3 blocks should return to the free pool
    And total blocks should remain unchanged
```

#### Scheduler Tests

```gherkin
Feature: Continuous Batching Scheduler

  Scenario: Schedule prefill requests
    Given a scheduler with 3 pending requests
    When schedule() is called
    Then a batch containing all 3 requests should be created
    And all requests should transition to Prefill state

  Scenario: Decode priority over prefill
    Given 2 decode requests and 2 prefill requests pending
    When schedule() is called with capacity for 3 requests
    Then both decode requests should be scheduled first
    And only 1 prefill request should be scheduled
```

### 2. Property Tests

Property tests verify invariants across many generated inputs using `proptest`.

#### Property Test Configuration

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    
    // All property tests must run minimum 100 iterations
}
```

#### Required Properties

| Property ID | Description | Validates |
|-------------|-------------|-----------|
| PROP-1 | Request ID Uniqueness | REQ-1.2 |
| PROP-2 | Parameter Validation Correctness | REQ-1.3 |
| PROP-3 | Block Allocation on Sequence Start | REQ-2.2 |
| PROP-4 | Block Allocation on Growth | REQ-2.3 |
| PROP-5 | Block Count Invariant | REQ-2.4, REQ-2.5 |
| PROP-6 | Scheduler Queue State Consistency | REQ-3.1 |
| PROP-7 | Batch Size Constraints | REQ-3.5 |
| PROP-8 | Decode Priority over Prefill | REQ-3.7 |
| PROP-9 | Prefill to Decode Transition | REQ-3.3 |
| PROP-10 | Completion Condition | REQ-3.4 |
| PROP-11 | Variable Sequence Length Handling | REQ-4.2 |
| PROP-12 | Memory Statistics Invariant | REQ-6.2 |
| PROP-13 | Memory Pressure Response | REQ-6.3 |
| PROP-14 | Configuration Validation | REQ-7.2 |
| PROP-15 | Tokenizer Round-Trip | REQ-8.4 |

### 3. Integration Tests

End-to-end tests validating complete system behavior.

```gherkin
Feature: End-to-End Inference

  Scenario: Submit request and get response
    Given an initialized InferenceEngine
    When a request "Hello, world!" is submitted
    And the engine runs until completion
    Then the request should complete successfully
    And output tokens should be generated
    And KV cache should be freed

  Scenario: Continuous batching with multiple requests
    Given an initialized InferenceEngine
    When 5 requests are submitted at staggered times
    And the engine runs until all complete
    Then all 5 requests should generate output
    And mixed prefill/decode batches should form

  Scenario: Memory pressure handling
    Given an InferenceEngine with limited KV cache
    When memory utilization exceeds threshold
    Then new prefill requests should be rejected
    And completed requests should free memory
    And new prefill requests should be accepted again
```

## Test Coverage Requirements

| Test Type | Minimum Count | Coverage Target |
|-----------|---------------|-----------------|
| Unit Tests | 78 | Core modules |
| Property Tests | 15 | Invariant verification |
| Integration Tests | 13 | End-to-end flows |
| Doc Tests | 29 | API examples |

## Test Execution

```bash
# Run all tests
cargo test

# Run property tests only
cargo test property_tests

# Run integration tests only
cargo test --test integration_tests

# Run with coverage
cargo tarpaulin --out Html
```
