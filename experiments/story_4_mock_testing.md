# Story 4: Mock Testing and Resource Profiling

## User Story
As a model developer, I want a comprehensive mock test suite that validates the TC-AdaptFormer model with synthetic data, measures resource consumption, and catches dimension mismatches, so that I can ensure the model is ready for real training.

## Context
- **Model**: TC-AdaptFormer from Story 3
- **Dataset**: AgriMultimodalDataset from Story 2
- **Goal**: Validate forward/backward pass, measure FLOPs, memory, and timing
- **No Real Data Required**: Use `torch.randn` for all inputs

## Acceptance Criteria

### AC1: Mock Data Generation
- [ ] File `tests/test_model_mock.py` created
- [ ] Function `generate_mock_batch(batch_size=8, T=5)` returns:
  - `video: torch.randn(B, T, 3, 224, 224)`
  - `gnss: torch.randn(B, 7)`
  - `labels: torch.randint(0, 11, (B,))`
- [ ] Verify shapes match expected dimensions

### AC2: Forward Pass Validation
- [ ] Instantiate `TCAdaptFormer(num_classes=11)`
- [ ] Run forward pass: `logits = model(video, gnss)`
- [ ] Assert output shape: `logits.shape == (B, 11)`
- [ ] Assert no NaN values: `torch.isnan(logits).sum() == 0`
- [ ] Assert logits are finite: `torch.isfinite(logits).all()`
- [ ] Print: "✓ Forward pass successful"

### AC3: Backward Pass Validation
- [ ] Compute loss: `loss = F.cross_entropy(logits, labels)`
- [ ] Run backward: `loss.backward()`
- [ ] Check gradients exist for trainable params:
  ```python
  for name, param in model.named_parameters():
      if param.requires_grad:
          assert param.grad is not None, f"No gradient for {name}"
  ```
- [ ] Verify frozen params have no gradients:
  ```python
  for name, param in model.named_parameters():
      if not param.requires_grad:
          assert param.grad is None, f"Unexpected gradient for frozen {name}"
  ```
- [ ] Print: "✓ Backward pass successful"

### AC4: Parameter Count Verification
- [ ] Count total parameters: `sum(p.numel() for p in model.parameters())`
- [ ] Count trainable parameters: `sum(p.numel() for p in model.parameters() if p.requires_grad)`
- [ ] Assert trainable params in range [1.8M, 2.5M]
- [ ] Print breakdown:
  ```
  Total parameters: 88.1M
  Trainable parameters: 2.1M (2.4%)
  Frozen parameters: 86.0M (97.6%)
  ```

### AC5: Memory Footprint Estimation
- [ ] Use `torch.cuda.memory_allocated()` if GPU available
- [ ] Measure memory before and after forward pass
- [ ] Estimate peak memory during training (forward + backward + optimizer states)
- [ ] Print: "Peak memory: XXX MB"
- [ ] Assert memory < 8GB for batch_size=8 (fits on consumer GPU)

### AC6: Inference Timing
- [ ] Measure forward pass time over 100 iterations
- [ ] Use `torch.cuda.synchronize()` if GPU
- [ ] Compute mean and std of inference time
- [ ] Print: "Inference time: XX.X ± Y.Y ms per sample"
- [ ] Assert inference time < 100ms per sample (real-time capable)

### AC7: Training Step Timing
- [ ] Measure full training step (forward + backward + optimizer.step()) over 50 iterations
- [ ] Use dummy optimizer: `torch.optim.AdamW(model.parameters(), lr=1e-4)`
- [ ] Print: "Training step time: XX.X ± Y.Y ms"
- [ ] Estimate throughput: samples/second

### AC8: Dimension Mismatch Detection
- [ ] Test with wrong input shapes and verify errors are caught:
  - `video.shape = (8, 3, 3, 224, 224)` (missing T dimension) → should raise AssertionError
  - `gnss.shape = (8, 5)` (wrong feature count) → should raise AssertionError
  - `video.shape = (8, 5, 3, 256, 256)` (wrong resolution) → should handle or error
- [ ] Verify error messages are informative

### AC9: Batch Size Scalability Test
- [ ] Test with batch_sizes = [1, 4, 8, 16, 32]
- [ ] Measure memory and time for each
- [ ] Print table:
  ```
  Batch Size | Memory (MB) | Time (ms) | Throughput (samples/s)
  -----------|-------------|-----------|----------------------
  1          | 1200        | 45        | 22
  4          | 2100        | 120       | 33
  8          | 3500        | 210       | 38
  16         | 6200        | 380       | 42
  32         | OOM         | -         | -
  ```
- [ ] Identify optimal batch size for available GPU

### AC10: Automated Test Report
- [ ] Generate `tests/mock_test_report.txt` with all results
- [ ] Include:
  - Model architecture summary
  - Parameter counts
  - Memory footprint
  - Timing benchmarks
  - Batch size scalability
  - Pass/Fail status for all tests
- [ ] Report should be human-readable and copy-pasteable into paper appendix

### AC11: Continuous Integration Ready
- [ ] Test script runs with `pytest tests/test_model_mock.py`
- [ ] All tests pass with exit code 0
- [ ] No warnings or deprecation messages
- [ ] Runtime < 5 minutes on CPU, < 1 minute on GPU

## Definition of Done
- File `tests/test_model_mock.py` exists and runs successfully
- All 11 acceptance criteria verified
- Test report generated at `tests/mock_test_report.txt`
- No dimension mismatches or runtime errors
- Model is validated as ready for real data training
- Report includes resource consumption estimates for paper

## Technical Notes
- Use `pytest` framework for test organization
- Use `@pytest.mark.parametrize` for batch size tests
- Mock test should NOT require real data or GPU (but support GPU if available)
- Catch and fix any dimension mismatches found during testing
- If errors occur, iterate with Story 3 (model implementation) until all tests pass

## Example Test Output
```
============================= test session starts ==============================
tests/test_model_mock.py::test_forward_pass PASSED                       [ 10%]
tests/test_model_mock.py::test_backward_pass PASSED                      [ 20%]
tests/test_model_mock.py::test_parameter_count PASSED                    [ 30%]
tests/test_model_mock.py::test_memory_footprint PASSED                   [ 40%]
tests/test_model_mock.py::test_inference_timing PASSED                   [ 50%]
tests/test_model_mock.py::test_training_timing PASSED                    [ 60%]
tests/test_model_mock.py::test_dimension_errors PASSED                   [ 70%]
tests/test_model_mock.py::test_batch_scalability PASSED                  [ 80%]
tests/test_model_mock.py::test_report_generation PASSED                  [ 90%]
tests/test_model_mock.py::test_no_warnings PASSED                        [100%]

======================== 10 passed in 45.23s ===============================

Mock Test Report Summary:
✓ Forward pass: PASS
✓ Backward pass: PASS
✓ Parameter count: 2.1M trainable (target: ~2M)
✓ Memory footprint: 3.5 GB (batch_size=8)
✓ Inference time: 42.3 ± 2.1 ms/sample
✓ Training throughput: 38 samples/s
✓ Optimal batch size: 8 (for 8GB GPU)
```

## Termination Condition
This story is COMPLETE when:
1. All pytest tests pass (exit code 0)
2. Test report shows no dimension mismatches
3. Resource consumption is within acceptable limits (< 8GB memory, < 100ms inference)
4. Report file exists and contains all required metrics

If any test fails, return to Story 3 to fix model implementation, then re-run Story 4.
