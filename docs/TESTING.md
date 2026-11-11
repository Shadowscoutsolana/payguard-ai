# PayGuard AI — Testing

Last full run: **278 passed, 5 skipped** (the skipped tests are the opt-in live-service tests) ·
line coverage **91%** of `backend/`, `ml/` and `scripts/` (`pytest-cov`).

## How to run

```powershell
# Offline suite: no network, PayPal in mock mode, Gemini blocked, test-only API keys
.\.venv\Scripts\python.exe -m pytest -q

# With coverage
.\.venv\Scripts\python.exe -m pytest -q --cov=backend --cov=ml --cov=scripts --cov-report=term-missing

# Live external services (real PayPal Sandbox, Gemini, Etherscan; uses the keys in .env)
$env:RUN_LIVE_TESTS = "1"; .\.venv\Scripts\python.exe -m pytest tests\test_live_services.py -v
```

Safety guards for the offline suite ([tests/conftest.py](../tests/conftest.py)): the real Gemini API can never be
called, and test-only API keys from [tests/keys.py](../tests/keys.py) replace the real ones.

## Mock and real paths are tested separately

| Path | What runs | Result (last run) |
|---|---|---|
| **Offline / mock** | Whole suite; PayPal mocked or HTTP faked; temporary SQLite databases | 278 passed |
| **Full stack (in-process)** | Real FastAPI server + real Streamlit dashboard, scenarios A/B/C, analyst approval, PDF ([test_dashboard_e2e.py](../tests/test_dashboard_e2e.py)) | 4 passed |
| **Live services** | Real PayPal Sandbox order lifecycle, Gemini report, Etherscan PYUSD fetch ([test_live_services.py](../tests/test_live_services.py)) | 4 passed, 1 skipped (live webhook needs `PAYPAL_WEBHOOK_ID`) |
| **Manual Sandbox end-to-end** | Buyer approves on PayPal's site (cannot be automated) | Scenario A: LOW, auto-captured (capture `77Y92357DW632724T`). Scenario C: HIGH, held and capture refused (403); live Gemini report generated; analyst `analyst.demo` approved with an analyst API key; captured by PayPal Sandbox (capture `85Y243739J442351M`, 4,800.00 USD, confirmed COMPLETED via PayPal's Orders API) |

## Requirement → tests

| # | Required area | Tests (file → test names) |
|---|---|---|
| 1 | PayPal create-order flow | `test_paypal_client`: `test_real_create_order_sends_capture_intent`, `test_mock_order_uses_capture_intent` · `test_integration_gaps`: `test_create_when_paypal_errors_returns_502_and_stores_nothing`, `test_create_when_paypal_returns_no_id_is_502` · live: `test_sandbox_order_lifecycle_without_buyer` |
| 2 | Buyer approval validation | `test_buyer_approval`: `test_real_approval_verified_with_paypal`, `test_forged_return_url_without_paypal_approval_is_rejected`, `test_amount_mismatch_with_paypal_is_rejected`, `test_cancelled_order_cannot_be_approved_or_captured`, `test_risk_analysis_refused_before_approval` |
| 3 | Capture-gate bypass prevention | `test_capture_gate`: `test_client_supplied_score_and_override_are_ignored`, `test_order_cannot_be_rescored`, `test_amount_must_match_order`, `test_capture_policy` · `test_security`: `test_analyst_key_cannot_run_merchant_actions`, `test_protected_endpoints_reject_missing_key` |
| 4 | Missing risk decision | `test_capture_gate`: `test_capture_without_risk_decision_is_rejected`, `test_capture_without_buyer_approval_is_rejected` |
| 5 | High-risk capture prevention | `test_capture_gate`: `test_high_risk_order_is_blocked` · `test_analyst_review`: `test_reject_blocks_capture_permanently` |
| 6 | Medium-risk step-up | `test_capture_gate`: `test_medium_risk_order_requires_step_up` · `test_step_up`: `test_step_up_not_available_for_other_bands` |
| 7 | Successful OTP verification | `test_step_up`: `test_correct_code_unlocks_capture`, `test_code_is_stored_only_as_hash` · e2e: `test_scenario_b_step_up_then_capture` |
| 8 | Failed OTP verification | `test_step_up`: `test_three_wrong_codes_escalate_to_analyst`, `test_expired_code_is_rejected_and_new_code_works`, `test_too_many_codes_escalate`, `test_malformed_code_is_rejected` |
| 9 | Analyst approval and actual capture | `test_analyst_review`: `test_approve_captures_through_the_gate`, `test_approve_reports_real_capture_failure`, `test_capture_can_be_retried_after_failure_with_analyst_approval` · e2e: `test_scenario_c_hold_analyst_approval_capture_and_case_report` |
| 10 | Analyst rejection | `test_analyst_review`: `test_reject_blocks_capture_permanently`, `test_alert_cannot_be_decided_twice`, `test_identity_comes_from_key_not_body` |
| 11 | Webhook signature verification | `test_webhooks`: `test_mock_mode_never_verifies`, `test_missing_webhook_id_fails`, `test_untrusted_cert_url_fails`, `test_forged_event_is_rejected_and_not_stored_under_claimed_id` · `test_integration_gaps`: `test_webhook_postback_interprets_paypal_answer` · live: `test_sandbox_rejects_forged_webhook` (skipped until webhook configured) |
| 12 | Duplicate webhook events | `test_webhooks`: `test_duplicate_event_is_processed_once` |
| 13 | Duplicate capture requests | `test_capture_gate`: `test_duplicate_capture_is_rejected` · `test_integration_gaps`: `test_simultaneous_captures_only_one_reaches_paypal` · `test_paypal_client`: `test_real_capture_uses_order_capture_endpoint_idempotently` |
| 14 | Missing-wallet handling | `test_wallet_policy`: `test_low_score_with_unknown_wallet_requires_step_up`, `test_partial_address_never_matches_known_wallet`, `test_unseen_address_is_unknown_not_clean` |
| 15 | XGBoost prediction | `test_ml_model`: `test_saved_model_is_xgboost_with_matching_features`, `test_prediction_has_no_feature_name_warning`, `test_missing_model_is_unknown_never_low` · `test_integration_gaps`: `test_paysim_loader_has_no_future_leakage`, `test_metrics_threshold_is_chosen_on_validation_not_test` |
| 16 | SHAP explanations | `test_ml_model`: `test_shap_values_are_real_and_additive`, `test_top_three_explanations_with_direction`, `test_rule_factors_are_never_labelled_shap` |
| 17 | Blockchain cycle detection | `test_graph_engine`: `test_cycle_is_found_from_transfers`, `test_no_cycle_longer_than_bound` · `test_blockchain_evidence`: `test_fast_cycle_is_flagged`, `test_slow_cycle_outside_window_is_not_flagged` |
| 18 | Wallet cluster evidence | `test_blockchain_evidence`: `test_analysis_returns_and_stores_evidence`, `test_cluster_endpoint_returns_members_transfers_and_evidence`, `test_pass_through_dispersion_is_flagged`, `test_busy_service_wallet_is_not_fan_out` |
| 19 | Gemini fallback | `test_investigation_agent`: `test_fallback_restates_stored_evidence_only`, `test_gemini_outage_falls_back_after_retries_and_logs_what_was_sent`, `test_ungrounded_gemini_answer_falls_back` · `test_integration_gaps`: `test_report_generator_crash_is_visible_and_audited` |
| 20 | Gemini report evidence validation | `test_investigation_agent`: `test_invalid_reports_are_rejected`, `test_valid_gemini_answer_is_used`, `test_pack_contains_no_identifiers` · live: `test_gemini_grounded_report` |
| 21 | PDF report generation | `test_case_report`: `test_report_contains_every_required_section`, `test_stored_gemini_report_is_rendered_with_citations`, `test_unicode_text_does_not_break_the_pdf`, `test_export_is_audited_with_exporter_identity` |

## Not covered by automated tests (and why)

- **Buyer approval on PayPal's website.** It requires a human logging in to the Sandbox buyer account, so it's verified manually (see the table above).
- **Live webhook delivery.** It needs a public tunnel and a registered webhook ID. The handler logic is fully tested with faked PayPal answers; the live check is a pending manual step.
- **Model training** (`ml/training/train_baseline.py`). It needs the 494 MB PaySim file (about 40 s). It was run and validated manually; its parts (feature loading and leakage rule, time split, metrics) are unit-tested.
