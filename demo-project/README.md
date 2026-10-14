

## Session 1: safe booking tools (local and free)

Requires Python 3.12. No dependency installation, GPU, API key, subscription, phone number, or server is needed. This phase contains no network client or paid provider calls.

Run commands from this folder with your configured Python interpreter:

```powershell
python -m unittest discover -s tests -v
python demo.py --smoke
python demo.py
```

The configured interpreter on this machine is:

```powershell
& 'C:/Users/rtalakoob/AppData/Local/anaconda3/python.exe' demo.py
```

The console uses a temporary SQLite database that disappears after exit. It seeds 21 appointments across the next seven days in UTC, at 09:00, 12:00, and 15:00 each day, for one shared mock service bay. The text agent receives the numbered current-availability list and must map requested dates/times to it. This is deliberately simpler than dealership-local time zones and multi-resource scheduling.

Try this sequence:

```text
slots
offer oil_change 1
confirm
retry
slots
quit
```

Compare the booking IDs for confirm and retry: they should match. Availability should lose exactly one slot. Creating an offer alone must not remove a slot.

## Decision boundaries

The future agent will propose actions, but the scheduler owns validation and commits. The console currently issues explicit commands; it does not understand natural language or use an LLM.

- `available_slots`: read current availability.
- `prepare_offer`: validate the service and slot, then persist an expiring proposal.
- `confirm_booking`: require explicit confirmation, verify the proposal/customer, and atomically create a booking.
- Request keys: a retry returns the original booking; reusing a key for a different offer is rejected.
- Concurrency: a unique slot constraint and transaction allow only one booking winner. An offer does not reserve inventory.

The tests cover confirmation, expiration, unsupported service, customer mismatch, key conflict, retries, and simultaneous callers. Synthetic customer IDs are not authentication. Only UTC timestamps from trusted application code are supported. A deployed system still needs real authorization, tenant isolation, timezone validation, load testing, migration management, logging hygiene, and recovery.

## Five two-hour sessions

1. Booking tools and tests: understand SQLite transactions, confirmation, offer expiration, and idempotency. Explain why these controls belong outside the LLM.
2. Text agent and evaluations: integrate one hosted LLM after budget approval. Use a fixed conversation set; score actual tool outcomes, not just persuasive answers. Add timeouts, maximum tool steps, and schema checks.
3. Browser voice: connect LiveKit and one STT/TTS path after provider/pricing review. Inspect partial transcripts, end-of-turn decisions, interruption handling, and latency. No phone number required.
4. Trained decision unit: build a small intent-classification baseline and held-out evaluation. Compare with an LLM approach. Report limitations of synthetic data and avoid leakage.
5. Regression checks and interview demo: inject tool failures, inspect traces, compare a measured improvement, and explain the architecture. Ray Serve/Dagster are optional extensions only if the core demo is already sound.

## Before any paid integration

Total user-approved ceiling: USD 100. Current phase spend: USD 0. Provider selection and current prices are not yet settled; a monthly subscription is not automatically necessary. Do not enable paid usage before we review it together.

- Use a separate demo project/key with least privileges. Never paste secrets into chat; enter them locally. Exclude `.env` and recordings from version control. This folder is inside OneDrive: prefer process-level secrets or a location outside synced folders.
- Set provider/project hard spending caps where supported; also set low credit balances or manual top-ups with automatic recharge disabled where available. Alerts alone are not hard caps. Verify what each provider actually enforces.
- Keep an application kill switch, per-session timer, single-session concurrency, maximum turns/tool calls, response token limits, and provider timeouts. Enforce these in application code, not prompts. Bound retries; never allow unbounded retry loops.
- Proposed voice demo limits: five minutes/session, one active session, 20 turns, three tool steps/turn, no automatic reconnect loop. These are requirements for the voice phase, NOT implemented yet. Only the 20-command local console limit is implemented now.
- Review estimated and billed usage after each test. Include all components: LiveKit session/inference charges, STT, LLM, and TTS. Estimates can lag billing and are not a guaranteed hard budget limit.
- Use synthetic dealership/customer data. Do not send employer data, customer records, or confidential documents. Limit or disable recording; redact identifiers from traces and control retention/access.
- Keep browser access authenticated with short-lived server-issued credentials. Never expose provider secrets in browser code. Stop active sessions/agent processes immediately after testing.

## Learning checkpoint

Explain these before session 2:

1. Why does confirmation happen after an offer instead of at availability lookup?
2. Why must an offer not guarantee that a slot stays free?
3. What happens if the booking commits but the client loses the response?
4. Why is a request key different from a customer ID?
5. Which tests check application safety, and which future evaluations will check model quality?

## Session 2: constrained text agent and evaluation

The selected project interpreter is now the existing workspace virtual environment (Python 3.13). Phase 1 still needs no external packages. Phase 2 adds the OpenAI SDK, declared in requirements.txt and installed in that environment. From this project folder in PowerShell:

```powershell
& '../.venv/Scripts/python.exe' -m unittest discover -s tests -v
& '../.venv/Scripts/python.exe' evaluate.py
```

Both commands are offline and free. Scripted evaluation supplies expected model outputs: it tests controller/database behavior, NOT natural-language understanding. Live evaluation uses the same ten synthetic cases to score model actions/arguments, event sequences, and actual persisted booking service/slot. It exits nonzero on regression. This tiny developer test set is not a representative benchmark or evidence of production readiness. Add genuinely unseen cases before claiming generalization.

Architecture:

```text
utterance + current slots -> OpenAI structured action -> Python validation
	-> read availability / prepare offer / clarify / mock handoff
exact user command confirm -> trusted controller -> scheduler transaction
```

The model cannot select confirm or provide customer IDs, request keys, or SQL. Replies are templates grounded in controller results; no second generation call is needed. This is a bounded action router, not an autonomous multi-step function-calling agent. The controller keeps structured service and exact slot-ID details plus only the latest confirmed booking in process memory; it does not retain a transcript or persist memory across sessions. This supports split requests and questions about the recent booking. A remembered detail is not booking authorization. Neutral acknowledgements such as `ok` do not confirm or replace a pending offer; only the exact `confirm` command books. Other new natural-language turns clear pending offer/retry context. Cancel and handoff clear pending request details, while the latest booking remains available for a status response until process exit. Voice consent handling will require a separate design.

### Billing checklist BEFORE live mode

No monthly chat subscription is required; API billing is separate. The adapter reads OPENAI_API_KEY from the inherited process environment. It does not load an .env file or print the key.

1. Prefer a separate personal demo project/key, not an employer key. Restrict model usage and endpoint permissions as your account allows.
2. Set a USD 5 project spend limit and ENABLE hard enforcement if available. Configure an early alert too. A threshold without enforcement is only an alert.
3. Disable automatic credit recharges. Prepaid depletion and enforced spend limits can both have accounting/propagation delays; they are not instantaneous cutoffs.
4. Inspect the correct project's usage dashboard after each session. Do not repeatedly restart a failed session without checking the failure and billing.
5. Send synthetic inputs only. store=False disables response storage for this endpoint, not every provider retention/logging mechanism. No conversations are written to local trace files; terminal output includes decisions, timings, and token totals.

Verified reference pricing for standard gpt-4.1-mini: USD 0.40 / million input tokens and USD 1.60 / million output tokens. Example: 1,000 input + 200 output tokens costs about USD 0.00072. Twenty such calls cost about USD 0.0144. Actual token use varies. The local estimate uses these rates and ignores cache discounts; failed/timed-out billed requests may not appear in local usage totals. Limits are per process, not an account-wide USD cap; restarting resets them. Keep provider controls enabled.

Implemented limits: 20 agent turns, 20 model request attempts per router, 10-minute active session check, 1,000 characters per utterance, 6,000 UTF-8 bytes for context JSON, 200 output tokens, 20-second SDK timeout, max_retries=0, fixed model/API endpoint, one request/action per natural-language turn. The SDK timeout is not a guaranteed total wall-clock deadline under all network conditions. The session expiry is checked before requests and after console input returns; it does not forcibly interrupt someone waiting at the input prompt. These bounds reduce exposure but do not guarantee the overall USD 100 budget across providers/processes.

After checking the controls yourself, explicitly enable live calls in your terminal:

```powershell
$env:DEMO_ALLOW_PAID_CALLS = '1'
& '../.venv/Scripts/python.exe' text_demo.py
```

If using Git Bash with the environment already activated:

```bash
export DEMO_ALLOW_PAID_CALLS=1
python text_demo.py
```

Try: `What appointments are available?`, then `Oil change at slot 1 please`, then `confirm`, then `retry`, then `quit`. Pending offers are not reservations. Existing bookings cannot be cancelled yet. Handoff is a local event; no staff member is contacted.

An opt-in model evaluation costs up to twelve model calls for the current fixed set of ten cases:

```powershell
& '../.venv/Scripts/python.exe' evaluate.py --live
Remove-Item Env:DEMO_ALLOW_PAID_CALLS
```

For Bash, disable live mode with `unset DEMO_ALLOW_PAID_CALLS`. We have NOT run a live evaluation or changed your provider billing settings. Model access, API behavior, and actual natural-language accuracy remain unverified until the first approved live run.

### Local speech-to-text experiment

`local_stt.py` transcribes a WAV, MP3, or M4A file with faster-whisper running on CPU using int8. The default `small` model supports English and French; `base` is available for a smaller/faster experiment. The first run downloads model weights to the local model cache. Audio is processed locally by this script and is not sent to OpenAI. This is file-based transcription, not browser streaming. The existing text agent still uses the hosted OpenAI router for decisions.

From this project folder, run:

```powershell
& '../.venv/Scripts/python.exe' local_stt.py path\to\sample.wav
& '../.venv/Scripts/python.exe' local_stt.py path\to\sample.wav --language fr
```

The command reports detected language, transcript segments, model-load time, and transcription time. Compare the transcription time with the recording length before choosing a model for interactive voice use.

References: https://developers.openai.com/api/docs/pricing ; https://developers.openai.com/api/docs/guides/spend-limits ; https://help.openai.com/en/articles/8264778-what-is-prepaid-billing

### Two-hour learning sequence

- 20 minutes: review agent.py. Identify what the model may propose versus what Python alone may commit.
- 20 minutes: run tests and scripted evaluation. Inspect why scripted passes are not model-quality scores.
- 15 minutes: review provider billing controls. Never paste credentials into chat.
- 25 minutes: manually try a live conversation, missing details, unsupported requests, and the explicit confirmation gate.
- 20 minutes: run the live evaluation. Inspect failing decisions and persisted outcomes rather than judging fluent responses.
- 20 minutes: propose one targeted prompt change, compare results, and add fresh edge cases. Do not train/fine-tune on this developer set and call it a held-out test.

### Offline task measurement and callback-priority experiment

The booking evaluator includes deterministic scenarios for punctuated spoken confirmation, ambiguous follow-up while an offer is pending, a wrong-date proposal, a real competing booking against the same SQLite slot, and a booking that commits but whose response is lost. The last case verifies idempotent retry with the same request key. Run it without model calls:

```powershell
& '../.venv/Scripts/python.exe' evaluate.py
```

The report compares expected events and exact persisted `(service, slot_id)` rows. It also reports booking precision/recall, booking success on cases expected to book, no-booking safety on cases expected not to book, and p50/p95 for case, turn, and model-call timing. In scripted-controller mode, the evaluator supplies the expected router decisions: perfect decision scores validate controller/database behavior only. Model-call latency is therefore null, and p95 from a small suite is illustrative rather than a production SLO.

`callback_triage_cases.json` and `callback_triage.py` demonstrate a separate supervised downstream task: rank missed calls for a prompt human callback. The binary label `urgent_callback` is manually assigned in synthetic examples; features represent hypothetical structured signals such as a safety concern, undrivable vehicle, repeat contact, or manager request. The benchmark holds out fixed examples, compares a transparent rule score with logistic regression, and reports precision, recall, F1, confusion matrix, and precision/recall within a 25% callback-capacity budget. Run it with:

```powershell
& '../.venv/Scripts/python.exe' callback_triage.py
```

These examples demonstrate an evaluation workflow, not dealership performance. In a real system, define the label with service operations, use consented and access-controlled records, adjudicate ambiguous labels, split by dealership/time to prevent leakage, compare against current callback policy, and evaluate missed urgent cases and workload at realistic staffing capacity before any operational use.