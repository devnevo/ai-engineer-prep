# The 4-Week Job-Ready AI/ML Engineer Plan

## For Pavan Sai Bellapu — From Senior Backend Engineer to Hired AI/ML Engineer

> **The Rule:** Every single day produces something visible — code pushed, a project deployed, a post written, an application sent. No day ends with only "reading" or "watching." If you read a chapter, you build something from it that same day.

---

# HOW THIS PLAN WORKS

**Each week has:**
- A primary BUILD project that drives all learning
- Exact O'Reilly chapters to read (only when you need them)
- Research papers to read (one every 2 days, 30-45 min each)
- Python/systems concepts to learn BY BUILDING, not by reading docs
- Tools to install and use that day
- A "proof of work" checkpoint — what your GitHub should show by end of week

**Daily structure (8-10 hours):**
- Morning (4-5h): BUILD — write code, push commits
- Afternoon (2-3h): READ — chapters and papers that solve problems you hit in the morning
- Evening (1h): DSA (1 NeetCode problem) + ML System Design (1 problem)
- Night (30 min): Write notes/LinkedIn draft about what you learned

---

# WEEK 1: THE RAG SYSTEM + PYTHON MASTERY

## The Mission
Build a production-quality RAG system from scratch. Not a LangChain tutorial — a system with hybrid search, evaluation, observability, and a deployed API. Every Python concept you need will be learned by building this system.

---

## Day 1: Foundation — FastAPI + Async Python + Project Setup

### What You Build
- FastAPI server with health check, query endpoint, and document upload endpoint
- Async document processing pipeline
- Project structure with proper Python packaging

### Python/Systems Concepts to Learn By Building

**Async/Await (you know Tornado — now learn the modern way):**
- `asyncio` event loop — how it actually works under the hood
- `async def` vs `def` — when to use which
- `asyncio.gather()` for parallel operations
- `httpx.AsyncClient` instead of `requests` (async HTTP calls)
- Key insight: FastAPI runs on `uvicorn` which uses `uvloop` (a C implementation of asyncio). Your Tornado experience maps directly — Tornado's IOLoop IS an event loop, FastAPI just uses the standard library version.

**HTTP Fundamentals:**
- How HTTP/1.1 keep-alive vs HTTP/2 multiplexing works
- What happens at the TCP level when your FastAPI server receives a request
- Status codes: 200, 201, 400, 404, 422 (FastAPI validation error), 500, 503
- Content-Type headers: `application/json` vs `multipart/form-data` (for file uploads)
- CORS — why your frontend will break without it

**Type Hints + Pydantic (backbone of modern Python):**
- Pydantic BaseModel for request/response validation
- `Field()` for documentation and constraints
- Generic types: `list[str]`, `dict[str, Any]`, `Optional[str]`
- Why Pydantic v2 is 5-50x faster than v1 (Rust core via `pydantic-core`)

**Project structure to build:**
```
knowledge-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py             # Pydantic Settings
│   ├── models/
│   │   ├── schemas.py        # Request/Response models
│   ├── api/
│   │   ├── routes.py         # Endpoints
│   ├── services/
│   │   ├── ingestion.py      # Document processing
│   │   ├── retrieval.py      # Search
│   │   ├── generation.py     # LLM calls
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
```

### Read Today
- **AI Engineering** (Chip Huyen) — Ch 1: Intro to Building AI Apps (skim, 45 min). Focus on: the AI stack diagram, planning framework
- FastAPI documentation: Tutorial through Request Body (30 min)

### Paper
**Attention Is All You Need** (Vaswani et al., 2017)
- Focus on: Section 3 (Model Architecture), Figure 1 and 2
- Core insight: self-attention lets every token attend to every other token in O(1) layers vs O(n) for RNNs

### DSA (Evening, 30 min)
NeetCode: Two Sum, Valid Anagram (warm-up — arrays + hashmaps)

---

## Day 2: Document Ingestion + Chunking + Embeddings

### What You Build
- Document loader for Markdown and PDF files
- Three chunking strategies: fixed-size, recursive, semantic
- Embedding generation with OpenAI or local model
- Compare chunk quality with a simple test

### Python/Systems Concepts

**Generators and Iterators (critical for large document sets):**
- `yield` keyword — process one document at a time without loading all into memory
- Generator expressions: `(chunk for doc in docs for chunk in chunk_doc(doc))`
- `itertools.islice()` for batching
- Why: when you process 10K documents, you can't load them all in memory. Generators stream through them.

**String Processing and Encoding:**
- Unicode: `str.encode('utf-8')`, why PDF extraction gives `\xa0` and `\u200b`
- `re.split(r'\n#{1,3}\s', text)` for splitting on markdown headers
- `tiktoken` library: how BPE tokenization actually works
- Key insight: chunk size in tokens ≠ chunk size in characters. Always count tokens.

**Concurrency for Embedding Generation:**
- `asyncio.Semaphore` to limit concurrent API calls
- `asyncio.gather(*tasks)` to embed chunks in parallel
- Batching: send 20 chunks per API call, not 1

### Read Today
- **AI Engineering** (Chip Huyen) — Ch 6: RAG and Agents. Focus on: chunking strategies, embedding models, retrieval methods
- **LLM Engineer's Handbook** (Iusztin & Labonne) — Ch 4: RAG Feature Pipeline. Focus on: ingestion design, chunking handlers

### Paper
**Dense Passage Retrieval (DPR)** (Karpukhin et al., 2020)
- Focus on: Section 3 (dual-encoder), Table 1 (DPR vs BM25)
- Core insight: dense retrieval beats keywords for meaning but fails on exact terms and rare entities → this is why hybrid search exists

### DSA
NeetCode: Group Anagrams, Top K Frequent Elements (hashmaps)

---

## Day 3: Vector Database + Hybrid Search

### What You Build
- PGVector setup with Docker
- HNSW index with proper distance metric
- BM25 keyword search using `rank_bm25` or PostgreSQL `tsvector`
- Hybrid search: combine vector + keyword with Reciprocal Rank Fusion (RRF)
- Compare: pure vector vs pure BM25 vs hybrid on 10 test queries

### Python/Systems Concepts

**Database Connections and Connection Pooling:**
- `asyncpg` for async Postgres connections
- Connection pool: create 10-20 connections, not one per request
- `async with pool.acquire() as conn:` — context managers for safe resource management
- Key insight: every database connection is a TCP socket. Creating one takes ~50ms. Pooling reuses them.

**Context Managers (the `with` statement):**
- `__enter__` and `__exit__` — what happens under the hood
- `@contextmanager` decorator for custom context managers
- `async with` for async resources
- Why: database connections, file handles, HTTP sessions — all need context managers to prevent leaks

**SQL for ML Engineers:**
- `pgvector` syntax: `ORDER BY embedding <=> $1 LIMIT 5` (cosine distance)
- `<=>` = cosine distance, `<->` = L2 distance — know when to use which
- `tsvector` and `tsquery` for PostgreSQL full-text search
- `EXPLAIN ANALYZE` — always check query plans

### Read Today
- PGVector documentation: Indexing guide (HNSW vs IVFFlat)
- **AI Engineering** (Chip Huyen) — Ch 6 continued: retrieval methods, reranking

### Paper
**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020)
- Focus on: Figure 1 (RAG architecture), Section 3
- Core insight: RAG decouples knowledge from model parameters — the model doesn't need to "know" everything, just find it

### DSA
NeetCode: Product of Array Except Self, Encode and Decode Strings

---

## Day 4: LLM Integration + Structured Extraction

### What You Build
- LLM generation with streaming (tokens arrive as they're generated)
- Structured output extraction using Pydantic + function calling
- "Contract clause extractor" mode: upload PDF, extract structured fields
- Retry logic with exponential backoff for API failures

### Python/Systems Concepts

**Streaming and Server-Sent Events (SSE):**
- `async for chunk in response.aiter_text():` — streaming from LLM APIs
- FastAPI `StreamingResponse` — send tokens to client as they arrive
- SSE protocol: `data: {json}\n\n` format
- Why: users perceive streaming as 5x faster even though total time is the same

**Error Handling for Production:**
- Custom exceptions: `class RetrievalError(Exception)`
- `tenacity` library: `@retry(stop=stop_after_attempt(3), wait=wait_exponential())`
- Circuit breaker: if LLM API fails 5x in a row, stop for 60 seconds
- Structured logging with `structlog`: every log is JSON with trace_id, latency, token_count

**Sockets and Networking (understand what's underneath):**
- TCP 3-way handshake — what happens before your first byte of data
- TLS/SSL — why HTTPS adds ~50ms to first request (certificate exchange)
- Keep-alive: `httpx.AsyncClient()` should be created once, not per request
- DNS resolution — what happens when you call `api.openai.com`

### Read Today
- **AI Engineering** (Chip Huyen) — Ch 2: Understanding Foundation Models. Focus on: sampling, structured outputs, probabilistic nature
- OpenAI/Anthropic function calling documentation

### Paper
**Language Models are Few-Shot Learners — GPT-3** (Brown et al., 2020)
- Focus on: Section 1 (scaling laws), Figure 1.2 (in-context learning)
- Core insight: scale creates qualitatively new capabilities. 175B parameters can do things 1B cannot.

### DSA
NeetCode: Valid Parentheses, Min Stack (stacks)

---

## Day 5: LangGraph Agent + Routing

### What You Build
- LangGraph agent with 3 paths: RAG, structured extraction, "I don't know"
- Router node classifying queries to the right path
- Fallback: low-relevance retrieval → "I don't know" instead of hallucinating
- Conversation memory (last 5 turns)

### Python/Systems Concepts

**State Machines and Graph Execution:**
- `TypedDict` for defining agent state
- LangGraph `StateGraph` maps to a finite state machine
- Conditional edges for routing decisions
- Key insight: an agent is just a state machine with an LLM at each node deciding the next transition

**Decorators (understand them, you use them everywhere):**
- `@app.get("/query")` — wraps function in a Route object
- `@retry`, `@cache`, `@trace` — writing your own decorators
- `functools.wraps` — preserving function metadata
- Decorator factories: decorators that take arguments

**Threading vs Asyncio vs Multiprocessing:**
- `asyncio` — IO-bound work (API calls, DB queries). One thread, many coroutines.
- `threading` — IO-bound when libs don't support async. GIL limits CPU parallelism.
- `multiprocessing` — CPU-bound work (local embedding). Separate processes, no GIL.
- Rule: API calls → asyncio. PDF parsing → multiprocessing. Almost never raw threading.

### Read Today
- LangGraph documentation: Quick Start + routing, checkpointing
- **AI Engineering** (Chip Huyen) — Ch 5: Prompt Engineering. Focus on: system prompts, defensive prompting, chain-of-thought

### Paper
**ReAct: Synergizing Reasoning and Acting** (Yao et al., 2022)
- Focus on: Figure 1 (ReAct vs CoT vs Act-only), Section 3
- Core insight: interleave reasoning with actions — more reliable than either alone

### DSA
NeetCode: Binary Search, Search in Rotated Sorted Array

---

## Day 6: Evaluation Pipeline

### What You Build
- Golden dataset: 30+ question-answer-context triples
- Ragas evaluation: faithfulness, answer relevance, context precision, context recall
- LLM-as-a-judge evaluator with custom rubric
- Comparison table: vector vs hybrid vs hybrid+reranker
- Save results as baseline JSON

### Python/Systems Concepts

**Testing Patterns for AI Systems:**
- `pytest` with `pytest-asyncio` for async code
- `@pytest.fixture` for test data and DB setup
- `@pytest.mark.parametrize("query,expected", test_cases)` for data-driven tests
- Key insight: AI systems need TWO kinds of tests — unit tests (does code work?) and eval tests (does AI produce good outputs?). Fundamentally different.

**Data Classes and Protocols:**
- `dataclasses.dataclass` vs Pydantic `BaseModel` — when to use which
- `Protocol` from typing — structural subtyping (duck typing with type safety)
- `Enum` for status codes and categories

### Read Today
- **AI Engineering** (Chip Huyen) — Ch 3: Evaluation Methodology (FULL chapter — the most important chapter for your career)
  - Focus on: entropy, perplexity, exact evaluation, embedding similarity, AI-as-a-judge
- **AI Engineering** — Ch 4: Evaluate AI Systems
  - Focus on: evaluation pipelines, automated evaluation, production evaluation
- Ragas documentation: Metrics guide

### Paper
**Judging LLM-as-a-Judge** (Zheng et al., 2023 — MT-Bench)
- Focus on: Section 3, Section 4 (agreement rates), Table 4 (bias analysis)
- Core insight: GPT-4-as-judge agrees with humans ~80% but has position bias, verbosity bias, self-enhancement bias

### DSA
NeetCode: Invert Binary Tree, Max Depth of Binary Tree (trees)

---

## Day 7: Docker + Deployment + Week 1 Polish

### What You Build
- Multi-stage Dockerfile (build + slim runtime)
- docker-compose.yml: FastAPI app, PGVector, Redis
- Deploy to Google Cloud Run
- Health check endpoint verifying all dependencies
- README with architecture diagram, eval results, setup instructions

### Docker Concepts (Hands-On)

**Dockerfile Best Practices:**
- Multi-stage builds: final image is 10x smaller
- `.dockerignore`: exclude `__pycache__`, `.git`, `tests/`, `*.pyc`
- Layer caching: `COPY requirements.txt` BEFORE `COPY app/` so deps are cached
- Non-root user: `RUN useradd -m appuser && USER appuser`
- Health checks: `HEALTHCHECK CMD curl -f http://localhost:8080/health`

**docker-compose for Local Development:**
- Service dependencies with health checks
- Volumes for persistent data
- `.env` file for environment variables — never hardcode secrets
- Service networking: `postgres:5432` not `localhost:5432`

**Container Fundamentals:**
- A container is a process with isolated namespaces (PID, network, filesystem), NOT a VM
- `cgroups` limit CPU and memory: `--memory=512m --cpus=1`
- Images are layered filesystems: each instruction creates a layer
- `docker exec -it container bash` — debug running containers

### Read Today
- Docker documentation: Best practices for Dockerfiles
- **LLM Engineer's Handbook** — Ch 2: Tooling and Installation (skim for architecture)

### Week 1 Checkpoint

**GitHub shows:**
- Working RAG system with hybrid search
- LangGraph agent with routing
- Structured extraction mode
- Evaluation results with actual numbers
- Dockerized and deployed
- Clean README with architecture diagram

**Papers read:** 6 (Attention, DPR, RAG, GPT-3, ReAct, MT-Bench)
**NeetCode solved:** 12
**Python concepts mastered:** async/await, Pydantic, generators, connection pooling, context managers, streaming, decorators, testing

---

# WEEK 2: FINE-TUNING + MLOPS PIPELINE + OBSERVABILITY

## The Mission
Build a fine-tuning pipeline for a domain-specific task, add MLOps (experiment tracking, CI/CD, monitoring) to your RAG system, and instrument everything with observability.

---

## Day 8: MLflow + Experiment Tracking for RAG

### What You Build
- MLflow server running in Docker
- Log every RAG experiment: prompt template, chunk size, retrieval strategy, eval metrics
- Compare experiments in MLflow UI
- Prompt registry: version your prompts like code

### MLOps Concepts

**Experiment Tracking:**
- `mlflow.log_param("chunk_size", 512)` — log configuration
- `mlflow.log_metric("faithfulness", 0.89)` — log results
- `mlflow.log_artifact("golden_dataset.json")` — log data files
- Runs, experiments, and model registry
- Key insight: every change to your AI system should be trackable. "Why did quality drop Tuesday?" → check MLflow, find the run, see the parameter change.

### Read Today
- **LLM Engineer's Handbook** — Ch 7: Evaluating LLMs (full). Focus on: evaluation frameworks, automated pipelines, metric selection
- MLflow documentation: Tracking guide + LLM evaluation

### Paper
**LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
- Focus on: Section 2 (method), Figure 1, Table 1
- Core insight: instead of updating all 7B parameters, train two small matrices (rank 8-64) that modify attention weights. Same quality, 100x fewer trainable parameters.

### DSA
NeetCode: LRU Cache (hashmaps + linked lists — very common)

---

## Day 9: CI/CD Pipeline + GitHub Actions

### What You Build
- GitHub Actions workflow on every PR:
  1. Lint with `ruff` (fastest Python linter, written in Rust)
  2. Type check with `mypy`
  3. Unit tests with `pytest`
  4. Eval suite against golden dataset
  5. Compare to baseline, fail if quality drops
- Baseline metrics in `eval/baseline.json`
- PR comment bot posting eval results as a table

### Python Quality Tools

**Linting and Formatting:**
- `ruff` — replaces flake8, isort, pyflakes. 10-100x faster.
- `ruff format` — replaces Black. Same output, faster.
- `mypy --strict` — catch type errors before runtime
- `pre-commit` hooks — run checks before every commit

**Testing for AI Systems:**
- Unit tests: chunking, routing logic, API endpoints
- Integration tests: full RAG pipeline with small test dataset
- Eval tests: Ragas on golden dataset, assert metrics above threshold
- `pytest-cov` — code coverage, aim 80%+ on non-AI code

### Read Today
- GitHub Actions documentation: Creating workflows
- **Designing Machine Learning Systems** (Chip Huyen) — Ch 9: Continual Learning and Test in Production. Focus on: testing ML systems, shadow deployment, A/B testing

### DSA
NeetCode: Clone Graph, Number of Islands (BFS/DFS)

---

## Day 10: OpenTelemetry + Observability

### What You Build
- OpenTelemetry SDK on every component:
  - FastAPI middleware (auto-instrumented)
  - PGVector queries (manual spans)
  - LLM API calls (manual spans with token counts)
  - Retrieval step (manual spans with relevance scores)
- Export traces to Arize Phoenix (free, open source)
- Dashboard: p50/p95/p99 latency per component, error rates, cost per query

### Observability Concepts

**Distributed Tracing:**
- Trace = complete request lifecycle
- Span = one operation within a trace (e.g. "embed_query", "vector_search", "llm_generate")
- Context propagation = passing trace ID across service boundaries
- `trace_id` and `span_id` — correlate logs, metrics, traces

**What to Instrument in an AI System:**
- Embedding time, retrieval time, retrieval quality (chunks returned, relevance)
- LLM time (model, tokens), total latency, cost per query

### Read Today
- **Observability Engineering** (Majors, Fong-Jones, Miranda) — Ch 1-2: What Is Observability + Debugging Practices. Focus on: cardinality, dimensionality
- **Observability Engineering** — Ch 5-7: Structured Events + Instrumentation (skim)
- OpenTelemetry Python SDK documentation

### Paper
**Efficient Memory Management for LLM Serving with PagedAttention** (Kwon et al., 2023 — vLLM)
- Focus on: Section 2 (KV cache problem), Section 3 (PagedAttention), Figure 3
- Core insight: KV cache wastes 60-80% of GPU memory through fragmentation. Paging eliminates waste, 2-4x more concurrent requests.

### DSA
NeetCode: Course Schedule (topological sort)

---

## Day 11-12: Fine-Tuning Project (2-Day Sprint)

### What You Build
- Choose domain: legal clause extraction OR clinical note generation
- Create instruction dataset: 200-300 examples
  - Use Claude/GPT-4 to generate initial examples
  - Human-review 50 for quality
  - Format in ChatML/Alpaca format
- Fine-tune with QLoRA using Unsloth on Google Colab (free T4)
- Train Mistral-7B or Llama-3-8B
- Evaluate: base model vs fine-tuned on 50-example test set

### Fine-Tuning Concepts

**Dataset Creation (the part nobody teaches):**
- Instruction format: `{"instruction": "...", "input": "...", "output": "..."}`
- Quality > quantity: 200 high-quality often beat 10,000 noisy
- Decontamination: test examples must not be in training set
- Augmentation: paraphrase instructions, vary complexity

**QLoRA (what's happening under the hood):**
- Base model weights frozen in 4-bit (NF4 quantization)
- LoRA adapters (rank 16-64) trained in FP16/BF16
- 0.1-1% of parameters trainable
- Memory: ~6GB for 7B model (fits free Colab T4)

**Training Configuration:**
- Learning rate: 2e-4 to 2e-5
- Epochs: 1-3 (LLMs overfit fast on small datasets)
- LoRA rank: 16-64 (higher = more capacity, more memory)
- LoRA alpha: usually 2x rank
- Target modules: `q_proj, k_proj, v_proj, o_proj` (attention layers)

### Read Today
- **LLM Engineer's Handbook** — Ch 5: Supervised Fine-Tuning (FULL — your primary guide). Focus on: instruction dataset creation, SFT techniques, LoRA/QLoRA, chat templates
- **LLM Engineer's Handbook** — Ch 6: Preference Alignment (skim DPO concept)
- Unsloth documentation + HuggingFace TRL SFTTrainer docs

### Papers
**QLoRA** (Dettmers et al., 2023)
- Focus on: Section 2 (4-bit NormalFloat), Table 1
- Core insight: 4-bit base + LoRA adapters = same quality as 16-bit full fine-tuning at 1/4 memory

**DPO: Direct Preference Optimization** (Rafailov et al., 2023)
- Focus on: Section 3, Figure 1 (DPO vs RLHF pipeline)
- Core insight: skip the reward model, optimize directly using preference pairs

### DSA
Day 11: Longest Substring Without Repeating Characters (sliding window)
Day 12: Container With Most Water (two pointers)

---

## Day 13: Fine-Tuning Deployment + Inference Optimization

### What You Build
- Deploy fine-tuned model with vLLM or HuggingFace Inference Endpoints
- Quantize with GGUF format for llama.cpp (optional — shows infra knowledge)
- Comparison page: base model vs fine-tuned side-by-side
- Upload model + model card to HuggingFace Hub
- Log training run in MLflow

### Inference Optimization Concepts

**Why Serving LLMs Is Hard:**
- Prefill phase: process entire prompt (compute-bound, parallelizable)
- Decode phase: generate one token at a time (memory-bound, sequential)
- KV cache: stores attention state, grows linearly with sequence length
- Batch size vs latency tradeoff

**Quantization Ladder:**
- FP32 → FP16: 2x less memory, almost no quality loss
- FP16 → INT8: 2x less, minor quality loss
- INT8 → INT4: 2x less, noticeable loss on edge cases
- GPTQ, AWQ, GGUF — know when each is used

### Read Today
- **AI Engineering** (Chip Huyen) — Ch 9: Inference Optimization. Focus on: KV cache, quantization, batching, hardware
- **LLM Engineer's Handbook** — Ch 8: Inference Optimization + Ch 9: Deploying the Inference Pipeline

### Paper
**FlashAttention** (Dao et al., 2022)
- Focus on: Section 1 (IO complexity), Figure 1 (memory hierarchy)
- Core insight: don't materialize the N×N attention matrix in HBM. Tile the computation, keep everything in SRAM. Same result, O(N) memory.

### DSA
NeetCode: Merge Intervals, Insert Interval

---

## Day 14: Online Monitoring + Auto-Rollback + Week 2 Polish

### What You Build
- Nightly batch eval: sample 50 production queries, run Ragas
- Alert if quality drops >5%
- Auto-rollback: faithfulness < 0.85 → revert prompt to last known-good
- Query logging: every query → PostgreSQL with response, chunks, latency, cost, timestamp
- Comprehensive READMEs for both projects

### Monitoring Concepts

**The Production Monitoring Loop:**
1. Every query logged with full context
2. Nightly job samples N queries, runs offline eval
3. Compare to baseline
4. Regression → alert, auto-rollback, incident
5. Weekly report: quality/cost/latency trends

**Alert Thresholds:**
- p99 latency > 3s
- Error rate > 2%
- Faithfulness < 0.85
- Token cost spike > 50% vs last week
- Zero queries for 10 minutes

### Read Today
- **Observability Engineering** — Ch 10: SLOs and Reliable Alerting
- **LLM Engineer's Handbook** — Ch 10: Monitoring and Prompt Tracking
- **LLM Engineer's Handbook** — Ch 11: MLOps and LLMOps (FULL — connects everything)

### Week 2 Checkpoint

**GitHub shows:**
- RAG system WITH MLOps: MLflow, GitHub Actions CI/CD, monitoring
- Fine-tuned model on HuggingFace with model card
- OpenTelemetry traces in Phoenix
- Eval results for both projects
- Two repos with excellent READMEs

**Papers read:** 11 total (added LoRA, QLoRA, DPO, vLLM, FlashAttention)
**NeetCode solved:** 24
**New skills:** MLflow, GitHub Actions, OpenTelemetry, QLoRA fine-tuning, vLLM

---

# WEEK 3: INFRASTRUCTURE + KUBERNETES + DATA ENGINEERING + VISIBILITY

## The Mission
Add production infrastructure to your projects (Kubernetes, Terraform, data pipelines), make yourself visible online, and start the application machine.

---

## Day 15: Kubernetes Fundamentals — Deploy Your RAG System on K8s

### What You Build
- Local Kubernetes cluster with `minikube` or `kind`
- Deployment manifests for: FastAPI app (2 replicas), PGVector (StatefulSet), Redis
- Service and Ingress for external access
- ConfigMap for non-secret config, Secret for API keys
- Horizontal Pod Autoscaler (HPA) based on CPU

### Kubernetes Concepts (Hands-On, Not Theory)

**Core Objects You Must Know:**
- Pod: smallest deployable unit. Usually 1 container. Ephemeral.
- Deployment: manages ReplicaSets, handles rolling updates, rollbacks
- Service: stable network endpoint for a set of Pods (ClusterIP, NodePort, LoadBalancer)
- Ingress: HTTP routing from outside the cluster to Services
- ConfigMap: non-sensitive config (chunk sizes, model names)
- Secret: sensitive data (API keys, DB passwords) — base64 encoded, not encrypted by default
- StatefulSet: for stateful workloads like databases (stable network identity, ordered scaling)
- PersistentVolumeClaim (PVC): request storage that survives pod restarts

**Networking Fundamentals:**
- Every Pod gets its own IP address within the cluster
- Services use `kube-proxy` and `iptables` to route traffic
- DNS: `my-service.my-namespace.svc.cluster.local`
- Key insight: Kubernetes networking is flat — every pod can talk to every other pod. Services add a stable DNS name on top.

**What to write today:**
```
k8s/
├── namespace.yaml
├── configmap.yaml
├── secret.yaml
├── app-deployment.yaml       # FastAPI, 2 replicas
├── app-service.yaml          # ClusterIP
├── app-hpa.yaml              # Autoscale 2-10 pods
├── pgvector-statefulset.yaml # PGVector with PVC
├── pgvector-service.yaml
├── redis-deployment.yaml
├── redis-service.yaml
├── ingress.yaml              # External access
```

### Read Today
- **Kubernetes Up and Running, 3rd Edition** (O'Reilly, Burns et al.) — Ch 1-5: Core concepts, Pods, Services, Deployments. Focus on: understanding the object model, not memorizing YAML
- **Kubernetes Up and Running** — Ch 13: ConfigMaps and Secrets

### Python/Systems Concepts

**Process Management:**
- `SIGTERM` vs `SIGKILL` — K8s sends SIGTERM first, gives you 30s to clean up
- Graceful shutdown: FastAPI `on_event("shutdown")` or `lifespan` to close DB connections
- `preStop` hooks in K8s: run cleanup before pod termination
- Why: if your pod dies mid-request, you lose a user's query. Graceful shutdown finishes in-flight requests.

**Health Checks:**
- Liveness probe: "Is the process alive?" (restart if not)
- Readiness probe: "Can it serve traffic?" (remove from Service if not)
- Startup probe: "Has it finished booting?" (for slow-starting apps)
- Your app needs all three: liveness = `/health`, readiness = `/ready` (checks DB connection), startup = `/health` with longer timeout

### DSA
NeetCode: Kth Smallest Element in BST, Validate BST (trees)

---

## Day 16: Terraform + Infrastructure as Code

### What You Build
- Terraform config for GCP (your comfort zone):
  - Cloud Run service for the app
  - Cloud SQL for PostgreSQL (with pgvector)
  - Memorystore for Redis
  - Artifact Registry for Docker images
  - IAM service account with minimal permissions
- `terraform plan`, `terraform apply`, `terraform destroy`
- Separate environments: `dev/` and `prod/` with different configs

### Terraform Concepts

**Core Workflow:**
- `terraform init` — download providers
- `terraform plan` — preview changes (ALWAYS read this)
- `terraform apply` — make changes
- `terraform destroy` — tear down everything

**Key Concepts:**
- State file: Terraform's record of what exists. NEVER edit manually. Store in GCS bucket for team use.
- Modules: reusable infrastructure components
- Variables and outputs: parameterize your infra
- `count` and `for_each`: create multiple similar resources
- `depends_on`: explicit dependency ordering

**Infrastructure structure:**
```
infrastructure/
├── modules/
│   ├── cloud-run/
│   ├── cloud-sql/
│   ├── redis/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── terraform.tfvars
│   ├── prod/
│       ├── main.tf
│       ├── variables.tf
│       ├── terraform.tfvars
├── backend.tf            # Remote state in GCS
```

### Read Today
- **Terraform: Up & Running, 3rd Edition** (O'Reilly, Brikman) — Ch 1-3: Getting started, state, modules. Focus on: state management, module design
- **Terraform: Up & Running** — Ch 5: Tips and tricks for rapid deployment
- **Terraform: Up & Running** — Ch 8: Production-grade Terraform (skim — reference later)

### Python/Systems Concepts

**Environment Variables and Configuration:**
- `pydantic-settings` for type-safe config from env vars
- 12-factor app methodology: config in environment, not code
- Secret management: never commit secrets. Use GCP Secret Manager or env injection.
- `.env` files for local dev, env vars for production

### DSA
NeetCode: Word Search (backtracking), Combination Sum

---

## Day 17: Data Pipeline with Prefect + Automated Ingestion

### What You Build
- Prefect flow for automated document ingestion:
  1. Fetch new/changed documents from a source (API, S3, web scrape)
  2. Chunk incrementally (only new/changed docs)
  3. Generate embeddings
  4. Upsert to PGVector
  5. Run eval on a sample
  6. Log results to MLflow
- Schedule to run daily
- Failure handling: retry failed tasks, alert on repeated failures

### Data Engineering Concepts

**Pipeline Design Patterns:**
- Idempotency: running the pipeline twice produces the same result
- Incremental processing: track a "watermark" (last processed timestamp), only process new data
- Dead letter queue: failed documents go to a retry queue, not lost
- Backfill: ability to reprocess historical data

**Prefect Fundamentals:**
- `@flow` decorator for the main pipeline
- `@task` decorator for individual steps
- Task dependencies are automatic (based on data flow)
- Retries: `@task(retries=3, retry_delay_seconds=60)`
- Scheduling with Prefect Cloud (free tier) or self-hosted server

**Data Quality:**
- Schema validation on input documents
- Embedding quality check: cosine similarity of test queries before/after update
- Row counts and null checks at each pipeline stage
- Great Expectations or Pandera for structured data validation

### Read Today
- **Fundamentals of Data Engineering** (O'Reilly, Reis & Housley) — Ch 1-3: Data engineering lifecycle, architecture, good data architecture. Focus on: data lifecycle, batch vs stream
- **Fundamentals of Data Engineering** — Ch 7: Ingestion. Focus on: ETL patterns, incremental loading
- Prefect documentation: Tutorial + Deployments

### Paper
**BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
- Focus on: Section 3 (pre-training tasks), Figure 1
- Core insight: bidirectional pre-training + fine-tuning paradigm. Train once on massive data, adapt cheaply. This is the mental model behind every fine-tuning job.

### DSA
NeetCode: Implement Trie (prefix tree — common in ML feature engineering)

---

## Day 18-19: LinkedIn Visibility Sprint + Resume Rewrite (2 days)

### Day 18: Write and Post LinkedIn Content

**Write 4 LinkedIn posts (draft all, post 1 per day over the next week):**

Post 1: "How I reduced AI costs 85% in production"
- Open with the $0.02 → $0.003 number
- Explain 3 techniques: prompt optimization, caching, parallel processing
- Close with the lesson learned
- This is YOUR real Cloudaccel story — it's more compelling than any tutorial

Post 2: "I fine-tuned Mistral-7B for [domain] — what worked and what didn't"
- Show the eval results: base model vs fine-tuned
- Share the dataset creation process
- Mention what surprised you
- Link to your HuggingFace model

Post 3: "RAG evaluation is harder than RAG building"
- Share your Ragas scores
- Explain the failure modes you found
- Show how hybrid search beat pure vector
- Link to your GitHub repo

Post 4: "The LLMOps pipeline every AI engineer should build"
- Architecture diagram of your full pipeline
- MLflow + GitHub Actions + monitoring loop
- What breaks without each component
- Link to repo

**LinkedIn Profile Optimization:**
- Headline: "ML Engineer | RAG Systems, LLM Fine-Tuning, Production AI | Python, GCP"
- About: 3-4 lines max. What you build, what you've shipped, what you're looking for.
- Featured: link to your GitHub repos and HuggingFace model
- Skills: add "LangChain", "LangGraph", "RAG", "LLM Fine-Tuning", "MLOps", "MLflow"

### Day 19: Resume Rewrite

**New resume structure (AI-first, not backend-first):**

Header: Pavan Sai Bellapu — ML Engineer | RAG Systems, LLM Fine-Tuning, Production AI

**Section 1: AI/ML Projects** (your two new projects)
- Knowledge Assistant: RAG + agent + eval + MLOps pipeline
- Domain Fine-Tuning: QLoRA + dataset creation + deployment + eval
- Include metrics: faithfulness scores, latency, cost per query

**Section 2: Professional Experience** (rewritten with AI lens)
- Lead with RAG platform bullet (Vertex AI, FAISS, LangChain, 50K interactions)
- Cost optimization bullet ($0.02 → $0.003)
- Distributed systems bullet (Redis, 100+ servers, 1000+ concurrent)
- API performance bullet (90% latency reduction)
- DROP or minimize: WYSIWYG editor, forms/polls, AngularJS work. They dilute the AI positioning.

**Section 3: Skills** (reorganized)
- AI/ML: LLMs, RAG, Fine-Tuning (QLoRA/LoRA), LangChain, LangGraph, Prompt Engineering, Evaluation (Ragas), MLflow
- Infrastructure: Docker, Kubernetes, Terraform, GCP (Vertex AI, Cloud Run, BigQuery), GitHub Actions CI/CD
- Backend: Python, FastAPI, async/await, PostgreSQL, Redis, MongoDB
- Observability: OpenTelemetry, Arize Phoenix, structured logging

### Read Today
- No technical reading. Focus entirely on writing and positioning.
- Optional: **The Software Engineer's Guidebook** (Gergely Orosz, O'Reilly) — Ch on resumes and career positioning

### DSA
Day 18: NeetCode: Longest Consecutive Sequence (hashmaps)
Day 19: NeetCode: Meeting Rooms II (intervals + heaps)

---

## Day 20: AWS Basics + Cross-Cloud Knowledge

### What You Build
- Deploy your RAG system on AWS (parallel to GCP):
  - ECR for Docker images
  - ECS Fargate for the app (serverless containers)
  - RDS PostgreSQL with pgvector extension
  - ElastiCache for Redis
- Compare: GCP Cloud Run vs AWS ECS Fargate (write this up — it's interview gold)

### AWS Concepts (Targeted for ML Engineers)

**Compute:**
- EC2: VMs (know instance types — P3/P4 for GPU, T3/T4 for general)
- ECS Fargate: serverless containers (your Cloud Run equivalent)
- Lambda: serverless functions (for lightweight processing)
- SageMaker: managed ML platform (training, inference endpoints)

**Storage:**
- S3: object storage (training data, model artifacts, embeddings)
- RDS: managed databases (PostgreSQL with pgvector)
- ElastiCache: managed Redis
- EFS: shared filesystem (for mounting model weights)

**Networking:**
- VPC: virtual network (your private cloud)
- Security Groups: firewall rules (inbound/outbound)
- IAM: identity and access management (roles, policies)
- Key insight: IAM is the most important AWS service. Every interview asks about least-privilege access.

**ML-Specific:**
- SageMaker endpoints: deploy models for inference
- Bedrock: managed LLM API (like Vertex AI)
- S3 + SageMaker: standard pattern for training data

### Read Today
- AWS documentation: ECS Fargate getting started
- **Practical Cloud Security** (O'Reilly, Dotson) — Ch 1-3: Cloud security fundamentals, IAM, encryption (skim)

### DSA
NeetCode: Design Twitter (combines multiple data structures)

---

## Day 21: Application Sprint Launch + Week 3 Polish

### What You Build
- Create profiles on: Arc.dev, Turing, Toptal, Wellfound
- Polish both GitHub repos: add badges, clean READMEs, add screenshots
- Record a 5-minute Loom walkthrough of your RAG system architecture
- Send first 10 applications

### Application Platforms (India → Remote USD)

| Platform | Action | Notes |
|---|---|---|
| Arc.dev | Complete profile + assessment | Specifically matches international to US companies |
| Turing | Take vetting test | Pay USD, specifically hire from India |
| Toptal | Apply for screening | Higher rates ($60-100+/hr), harder entry |
| Wellfound | Apply to 5 startups/day | Startups most open to international |
| Remotive | Apply to all matching AI roles | Curated remote-only |
| LinkedIn | Apply + DM hiring managers | Include project link in message |
| HN Who's Hiring | Watch for March 1 thread | Direct to hiring managers |

### Week 3 Checkpoint

**GitHub shows:**
- RAG system deployed on BOTH GCP and AWS
- Kubernetes manifests
- Terraform infrastructure code
- Prefect data pipeline
- Fine-tuned model on HuggingFace

**Online presence:**
- 4 LinkedIn posts published (1/day)
- Rewritten resume
- Profiles on 4+ job platforms
- 10+ applications sent

**Papers read:** 12 (added BERT)
**NeetCode solved:** 34
**New skills:** Kubernetes, Terraform, Prefect, AWS basics, data pipelines

---

# WEEK 4: INTERVIEW PREP + SYSTEM DESIGN + AGGRESSIVE APPLICATIONS

## The Mission
This week is 50% interview preparation and 50% applications. You should be sending 5+ applications per day while preparing for the interviews that start coming in.

---

## Day 22-23: ML System Design Deep Dive (2 days)

### What You Practice

Do 1 system design problem per day, 45 minutes each, as if in an interview. Write your answer, draw the architecture, then compare to the reference.

**Problem Set (practice all 8 over these 2 days + evenings):**

| # | Problem | Key Concepts Tested |
|---|---|---|
| 1 | Design a RAG system for 10M legal documents | Chunking at scale, hybrid search, sharding, eval, cost |
| 2 | Design a content moderation pipeline using LLMs | Classification, confidence thresholds, human-in-the-loop |
| 3 | Design a domain-specific chatbot for healthcare | Fine-tuning vs RAG, compliance (HIPAA), evaluation |
| 4 | Design a real-time recommendation system | Feature store, embedding similarity, serving latency |
| 5 | Design an LLM evaluation pipeline for CI/CD | Golden datasets, quality gates, regression testing |
| 6 | Design an AI agent for customer support | Tool routing, escalation, safety, state management |
| 7 | Design a model serving platform for 100 models | Multi-model serving, A/B testing, canary deployment |
| 8 | Design a data pipeline for continuous fine-tuning | Data collection, quality filtering, retraining triggers |

**The Framework for Every ML System Design Answer:**

1. **Clarify requirements** (2 min): users, scale, latency, accuracy target
2. **High-level architecture** (5 min): draw boxes and arrows
3. **Data pipeline** (5 min): how does data flow in?
4. **Model/AI component** (10 min): which model, why, tradeoffs
5. **Serving** (5 min): how does it serve at scale?
6. **Evaluation** (5 min): how do you know it works?
7. **Monitoring** (3 min): how do you know when it breaks?
8. **Tradeoffs** (5 min): what would you do differently at 10x scale?

### Read Today
- **Designing Machine Learning Systems** (Chip Huyen) — Ch 1-2: Overview, ML Systems Design. Foundation for all system design thinking.
- **Designing Machine Learning Systems** — Ch 6: Model Development and Offline Evaluation
- **Designing Machine Learning Systems** — Ch 7: Model Deployment and Prediction Service
- **Designing Machine Learning Systems** — Ch 9: Continual Learning and Test in Production
- **ML System Design Interview** (Alex Xu & Ali Aminian) — any 2 chapters matching problems above

### Python/Systems Concepts

**Caching Strategies (you know Redis — go deeper):**
- Cache-aside pattern: check cache → miss → query DB → write to cache
- Write-through: write to cache AND DB simultaneously
- TTL strategy for AI: cache embeddings forever, cache LLM responses for 1 hour
- Cache invalidation: the "two hard problems in CS" — when to evict stale results
- Semantic caching: cache similar queries, not just exact matches

**Load Balancing:**
- Round-robin: simple, doesn't account for server load
- Least connections: send to least-busy server
- Consistent hashing: for distributing cache keys across multiple Redis instances
- Key insight: for LLM serving, you need "sticky" routing to reuse KV cache on the same GPU

### DSA
Day 22: NeetCode: Alien Dictionary (topological sort — hard)
Day 23: NeetCode: Median of Two Sorted Arrays (binary search — hard, skip if too time-consuming)

---

## Day 24: Behavioral Interview Prep + Your Story

### What You Prepare

**Your 3 Core Stories (practice telling each in 2 minutes):**

Story 1: "The RAG Platform" (Cloudaccel)
- Situation: 2000+ schools needed AI-powered content generation
- Action: architected RAG with Vertex AI, FAISS, LangChain, Dialogflow CX
- Result: 94% satisfaction, 50K monthly interactions, 60% workload reduction
- Lesson: the hardest part wasn't the AI — it was understanding teacher workflows

Story 2: "The Cost Optimization" (Cloudaccel)
- Situation: AI costs were $0.02 per interaction at scale → unsustainable
- Action: prompt optimization, data threshold validation, parallel processing
- Result: 85% reduction to $0.003, maintained quality
- Lesson: always measure cost per query from day 1, not after launch

Story 3: "The Fine-Tuning Project" (your new project)
- Situation: base model couldn't reliably extract structured data from [domain] documents
- Action: created 250-example dataset, fine-tuned with QLoRA, built eval pipeline
- Result: [your actual metrics] — X% improvement over base model
- Lesson: 200 high-quality examples beat 10K noisy ones

**Common Behavioral Questions for AI/FDE Roles:**
- "Tell me about a time you had to make a technical tradeoff"
- "Describe a project where the requirements were ambiguous"
- "How do you explain technical concepts to non-technical stakeholders?"
- "Tell me about a time something broke in production"
- "How do you prioritize when you have competing deadlines?"

Use STAR format (Situation, Task, Action, Result) for every answer. Keep it under 2 minutes.

### Read Today
- **The Trusted Advisor** (Maister, Green, Galford) — Ch 1-3: The Trust Equation (Credibility + Reliability + Intimacy / Self-Orientation). Focus on: how to build trust quickly with new teams
- **Crucial Conversations** (Patterson et al.) — Ch 1-3: Core principles (skim — internalize the concept of "creating safety" before difficult conversations)

### DSA
NeetCode: Climbing Stairs, House Robber (basic DP)

---

## Day 25: Coding Interview Patterns

### What You Practice

By now you have ~38 problems done. Today is pattern consolidation. Do 4-5 problems focusing on patterns you're weakest on.

**The 8 Patterns That Cover 80% of AI Company Coding Interviews:**

| Pattern | Key Problems | When to Recognize |
|---|---|---|
| **Hashmaps/Sets** | Two Sum, Group Anagrams, Top K Frequent | "Find", "count", "group", "duplicate" |
| **Sliding Window** | Longest Substring, Max Sum Subarray | "Contiguous subarray", "substring", "window" |
| **Two Pointers** | Container With Water, 3Sum | Sorted array, "pair", "meeting from both ends" |
| **BFS/DFS** | Number of Islands, Clone Graph, Course Schedule | Graphs, trees, "connected", "path", "level" |
| **Binary Search** | Search Rotated Array, Koko Eating Bananas | Sorted data, "minimum maximum", "search space" |
| **Trees** | Invert, Max Depth, Validate BST, LCA | Tree traversal, recursive structure |
| **Dynamic Programming** | Climbing Stairs, House Robber, Coin Change | "Count ways", "minimum cost", "maximum value" |
| **Heaps** | Top K, Merge K Sorted Lists, Meeting Rooms II | "K largest/smallest", "scheduling", "merge" |

### Python Tricks for Coding Interviews:
- `collections.defaultdict(list)` — auto-create empty lists
- `collections.Counter(arr)` — count frequencies instantly
- `heapq.nlargest(k, arr)` — top K without sorting
- `itertools.combinations(arr, 2)` — all pairs
- `bisect.bisect_left(arr, target)` — binary search in sorted list
- `functools.lru_cache(None)` — memoize recursive functions
- Slicing: `arr[::-1]` reverse, `arr[::2]` every other
- Tuple as dict key: `dict[(x, y)]` for coordinate problems

### Read Today
- NeetCode roadmap: review patterns you've missed
- Optional: **Cracking the Coding Interview** relevant sections (but NeetCode is better for patterns)

### DSA
Do 4-5 problems from your weakest patterns

---

## Day 26-27: Application Blitz (2 days)

### What You Do

These 2 days are 80% applications, 20% prep.

**Target: 40+ applications over 2 days**

**Morning (3h): Tailored Applications**
- 5 applications to high-match roles on Wellfound/Arc
- For each: read the job description, tailor 2-3 lines in your cover note
- Mention your specific project that matches their needs
- DM the hiring manager on LinkedIn with a 3-line message + project link

**Afternoon (3h): Volume Applications**
- 15 applications on Turing, Remotive, LinkedIn
- Use a template message, swap company name and 1 specific detail
- Apply to anything matching: AI Engineer, ML Engineer, LLM Engineer, Applied AI, MLOps Engineer

**Evening (2h): Outreach**
- Find 5 companies you'd love to work for that aren't posting jobs
- Find their CTO/VP Engineering on LinkedIn
- Send a message: "I built [this] and noticed you're working on [that]. Would love to chat about how I could help."
- This is how 30% of startup jobs are filled — before they're posted

**The Message Template:**
```
Hi [Name],

I'm an ML Engineer who's built RAG systems serving 50K+ monthly
interactions in production. I recently fine-tuned Mistral-7B for
[domain] and built a full LLMOps pipeline with eval gates and
monitoring.

Saw your [specific thing about company]. Would love to discuss
how I could help.

GitHub: [link]
```

### DSA
Day 26: NeetCode: Coin Change (DP)
Day 27: NeetCode: Merge K Sorted Lists (heaps)

---

## Day 28: Mock Interviews + Final Polish

### What You Do

**Morning: Self-Mock Interview (2 hours)**

Set a timer. Do each section exactly as you would in an interview:
- 45 min: ML System Design — pick a problem, whiteboard it, explain out loud
- 30 min: Coding — pick a medium NeetCode, solve with timer running
- 30 min: Behavioral — answer 5 questions out loud, record yourself
- 15 min: Review recordings, identify weak spots

**Afternoon: Project Polish**
- Final README cleanup
- Add a "Demo" section with screenshots or a Loom video link
- Ensure both projects deploy with a single `docker-compose up`
- Add a "Quick Start" that anyone can follow in 5 minutes

**Evening: Reflect and Plan**
- Count: applications sent, responses received, interviews scheduled
- Identify: which platforms got responses, which didn't
- Adjust: double down on what's working

### Week 4 Checkpoint

**Applications sent:** 50+
**Interviews scheduled:** aim for 2-5 by end of week 4
**GitHub projects:** 2, polished, deployed, with eval metrics
**HuggingFace:** fine-tuned model with model card
**LinkedIn:** 4 posts published, profile optimized
**NeetCode solved:** 42-45
**ML system design problems practiced:** 8
**Papers read:** 12

---

# REFERENCE SECTIONS

---

## Complete Paper Reading List (Priority Order)

| # | Paper | When to Read | Core Insight |
|---|---|---|---|
| 1 | Attention Is All You Need (2017) | Day 1 | Self-attention replaces recurrence |
| 2 | Dense Passage Retrieval (2020) | Day 2 | Dense beats BM25 for meaning, fails on exact |
| 3 | RAG for Knowledge-Intensive NLP (2020) | Day 3 | External memory decouples knowledge from parameters |
| 4 | GPT-3: Few-Shot Learners (2020) | Day 4 | Scale creates emergent capabilities |
| 5 | ReAct (2022) | Day 5 | Interleave reasoning + acting |
| 6 | MT-Bench / LLM-as-Judge (2023) | Day 6 | AI evaluation: 80% human agreement, known biases |
| 7 | LoRA (2021) | Day 8 | Low-rank adapters = cheap fine-tuning |
| 8 | vLLM / PagedAttention (2023) | Day 10 | KV cache paging = 2-4x more concurrent users |
| 9 | QLoRA (2023) | Day 11 | 4-bit base + LoRA = full quality at 1/4 memory |
| 10 | DPO (2023) | Day 12 | Skip reward model, optimize preferences directly |
| 11 | FlashAttention (2022) | Day 13 | Tiled attention = O(N) memory instead of O(N²) |
| 12 | BERT (2018) | Day 17 | Pre-training + fine-tuning paradigm |

---

## Complete O'Reilly Reading Map

### AI Engineering — Chip Huyen (2025)
| Chapter | Day | Focus |
|---|---|---|
| Ch 1: Building AI Apps | Day 1 | AI stack, planning framework |
| Ch 2: Foundation Models | Day 4 | Sampling, structured outputs |
| Ch 3: Evaluation Methodology | Day 6 | Entropy, perplexity, eval methods (FULL READ) |
| Ch 4: Evaluate AI Systems | Day 6 | Eval pipelines, automated eval |
| Ch 5: Prompt Engineering | Day 5 | System prompts, defensive prompting |
| Ch 6: RAG and Agents | Day 2-3 | Chunking, retrieval, reranking |
| Ch 7: Finetuning | Day 11 | When to fine-tune vs prompt |
| Ch 9: Inference Optimization | Day 13 | KV cache, quantization, batching |
| Ch 10: AI Engineering Architecture | Day 22 | Full system design patterns |

### LLM Engineer's Handbook — Iusztin & Labonne (2024)
| Chapter | Day | Focus |
|---|---|---|
| Ch 2: Tooling | Day 7 | Architecture patterns (skim) |
| Ch 4: RAG Feature Pipeline | Day 2 | Ingestion, chunking, embedding |
| Ch 5: Supervised Fine-Tuning | Day 11-12 | SFT, LoRA, QLoRA, chat templates (FULL READ) |
| Ch 6: Preference Alignment | Day 12 | DPO concept (skim) |
| Ch 7: Evaluating LLMs | Day 8 | Evaluation frameworks |
| Ch 8: Inference Optimization | Day 13 | Quantization, flash attention |
| Ch 9: Deploying Inference | Day 13 | Deployment patterns |
| Ch 10: Monitoring | Day 14 | Prompt tracking, drift |
| Ch 11: MLOps/LLMOps | Day 14 | Full MLOps pipeline (FULL READ) |

### Designing Machine Learning Systems — Chip Huyen (2022)
| Chapter | Day | Focus |
|---|---|---|
| Ch 1-2: Overview + Design | Day 22 | ML system thinking |
| Ch 6: Model Development | Day 22 | Offline evaluation |
| Ch 7: Deployment | Day 22 | Serving patterns |
| Ch 9: Continual Learning | Day 9, 22 | Testing ML, production eval |

### Observability Engineering — Majors, Fong-Jones, Miranda
| Chapter | Day | Focus |
|---|---|---|
| Ch 1-2: What Is Observability | Day 10 | Core concepts |
| Ch 5-7: Events + Instrumentation | Day 10 | Structured events (skim) |
| Ch 10: SLOs and Alerting | Day 14 | Error budgets, burn rates |

### Kubernetes Up and Running, 3rd Edition — Burns et al.
| Chapter | Day | Focus |
|---|---|---|
| Ch 1-5: Core concepts | Day 15 | Pods, Services, Deployments |
| Ch 13: ConfigMaps, Secrets | Day 15 | Config management |

### Terraform: Up & Running, 3rd Edition — Brikman
| Chapter | Day | Focus |
|---|---|---|
| Ch 1-3: Getting started | Day 16 | State, modules |
| Ch 5: Tips and tricks | Day 16 | Rapid deployment |
| Ch 8: Production-grade | Day 16 | Reference later |

### Fundamentals of Data Engineering — Reis & Housley
| Chapter | Day | Focus |
|---|---|---|
| Ch 1-3: Lifecycle, architecture | Day 17 | Data maturity, pipeline design |
| Ch 7: Ingestion | Day 17 | ETL patterns |

### Supplementary (Read as Needed)
- **Fundamentals of Software Architecture** (Richards & Ford) — Ch 19-23 for architecture decisions, diagramming, negotiation
- **The Trusted Advisor** (Maister et al.) — Ch 1-3 for client trust
- **Crucial Conversations** — Ch 1-3 for difficult conversations
- **ML System Design Interview** (Xu & Aminian) — system design practice problems
- **Practical Cloud Security** (Dotson) — Ch 1-3 for cloud security fundamentals

---

## Complete Tools Reference

### Core Stack (Must Know)
| Tool | Category | Why |
|---|---|---|
| **Python 3.11+** | Language | Everything is Python |
| **FastAPI** | Web framework | Standard for ML APIs |
| **Pydantic** | Validation | Request/response models, config |
| **Pytest** | Testing | Unit + eval tests |
| **Docker** | Containers | Packaging and deployment |
| **Git/GitHub** | Version control | Collaboration + CI/CD |

### AI/ML Stack
| Tool | Category | Why |
|---|---|---|
| **LangChain** | Orchestration | Most common in job listings |
| **LangGraph** | Agent framework | Stateful, controllable agents |
| **OpenAI / Anthropic SDKs** | LLM APIs | Direct API access |
| **Hugging Face Transformers** | ML library | Model loading, tokenization |
| **Unsloth** | Fine-tuning | Fastest QLoRA implementation |
| **TRL (HF)** | Training | SFTTrainer, DPO training |
| **vLLM** | Inference | Production LLM serving |
| **tiktoken** | Tokenization | Token counting |

### Data & Storage
| Tool | Category | Why |
|---|---|---|
| **PostgreSQL + pgvector** | Vector DB | Production-grade, hybrid search |
| **Redis** | Cache | Query caching, session state |
| **Prefect** | Pipeline orchestration | Data pipeline scheduling |
| **SQLAlchemy + asyncpg** | Database ORM | Async DB access |

### MLOps & Monitoring
| Tool | Category | Why |
|---|---|---|
| **MLflow** | Experiment tracking | Industry standard, open source |
| **Ragas** | AI evaluation | RAG-specific eval metrics |
| **GitHub Actions** | CI/CD | Free, universal |
| **OpenTelemetry** | Observability | Distributed tracing standard |
| **Arize Phoenix** | AI observability | LLM-specific traces |
| **Ruff** | Linting | Fastest Python linter |
| **mypy** | Type checking | Catch errors early |

### Infrastructure
| Tool | Category | Why |
|---|---|---|
| **Docker Compose** | Local multi-container | Development environment |
| **Kubernetes (minikube)** | Container orchestration | Enterprise deployment |
| **Terraform** | IaC | Reproducible infrastructure |
| **GCP Cloud Run** | Serverless containers | Your comfort zone |
| **AWS ECS Fargate** | Serverless containers | Cross-cloud skill |

---

## Python Concepts Master List

### Must-Know (Build Intuition Through Projects)

**Async Programming:**
- `asyncio` event loop, `async/await`, `asyncio.gather()`, `asyncio.Semaphore`
- `uvloop` (C-based event loop used by uvicorn)
- When to use async vs threading vs multiprocessing

**HTTP & Networking:**
- TCP 3-way handshake, TLS/SSL handshake
- HTTP/1.1 keep-alive vs HTTP/2 multiplexing
- DNS resolution flow
- Sockets: `socket.socket(AF_INET, SOCK_STREAM)` — understand what lies beneath `httpx`
- WebSockets vs SSE vs long polling — tradeoffs for real-time communication

**Data Handling:**
- Generators (`yield`), generator expressions, `itertools`
- `json` serialization/deserialization, `orjson` for speed
- `struct` and `Protocol Buffers` for binary serialization (you already know protobuf)
- File handling: context managers, `pathlib.Path`, `aiofiles` for async file IO

**Type System:**
- Type hints: `list[str]`, `Optional[str]`, `Union[str, int]`
- Generics: `TypeVar`, `Generic[T]`
- `Protocol` for structural subtyping
- Pydantic `BaseModel` vs `dataclass` — when to use which

**Concurrency:**
- GIL (Global Interpreter Lock): what it is, why it exists, how to work around it
- `threading.Thread` for IO-bound parallelism
- `multiprocessing.Pool` for CPU-bound parallelism
- `concurrent.futures.ThreadPoolExecutor` and `ProcessPoolExecutor`
- Queue patterns: `asyncio.Queue`, `queue.Queue`, `multiprocessing.Queue`

**Memory Management:**
- Reference counting + cyclic garbage collector
- `__slots__` for memory-efficient classes
- `sys.getsizeof()` for measuring object sizes
- Weak references: `weakref` module for caches that don't prevent GC
- Why Python lists use 1.125x memory growth factor

**Testing:**
- `pytest`: fixtures, parametrize, conftest.py, markers
- `pytest-asyncio` for async test functions
- `unittest.mock`: `MagicMock`, `patch`, `AsyncMock`
- Test-Driven Development (TDD): write the test first, then the code
- Property-based testing: `hypothesis` library

**Error Handling:**
- Exception hierarchy: `BaseException → Exception → specific`
- Custom exceptions for domain logic
- Context managers for resource cleanup
- `tenacity` for retry logic with exponential backoff
- Circuit breaker pattern for external service calls

**Performance:**
- `cProfile` and `line_profiler` for finding bottlenecks
- `functools.lru_cache` for memoization
- `orjson` vs `json` (10x faster serialization)
- Connection pooling for databases and HTTP clients
- Batch operations: always batch API calls and DB writes

---

## Best Practices and Anti-Patterns

### AI System Anti-Patterns (Memorize These)

| Anti-Pattern | What Goes Wrong | What to Do Instead |
|---|---|---|
| **No evaluation** | You can't prove it works | Build eval pipeline from day 1 |
| **Demo-driven development** | Works in demo, fails on real data | Test on production-like data before shipping |
| **Ignoring retrieval quality** | Bad chunks → bad answers → blame the model | Measure retrieval precision/recall separately from generation |
| **Prompt spaghetti** | 20 nested if-else prompt templates | Version prompts in MLflow, A/B test changes |
| **No cost tracking** | $10K/month surprise bill | Log token count per query, set budget alerts |
| **Hallucination tolerance** | "The AI said so" without verification | Add citation extraction, confidence scores, "I don't know" paths |
| **Fine-tuning too early** | Spent 2 weeks when prompt engineering would work | Always try prompt engineering first, fine-tune only when it provably can't work |
| **Single-model dependency** | OpenAI goes down, your system dies | Abstract the LLM layer, support multiple providers |
| **No graceful degradation** | API timeout → user sees error page | Return cached result or "I need a moment" instead of crashing |

### Python Anti-Patterns

| Anti-Pattern | What Goes Wrong | What to Do Instead |
|---|---|---|
| **`import *`** | Namespace pollution, unclear dependencies | Explicit imports: `from module import specific_thing` |
| **Mutable default args** | `def f(items=[])` — shared across calls | Use `None`: `def f(items=None): items = items or []` |
| **Bare except** | `except:` catches `SystemExit`, `KeyboardInterrupt` | Catch specific: `except ValueError:` or at minimum `except Exception:` |
| **God class** | One class does everything | Single Responsibility: each class/module does one thing |
| **Not using context managers** | File handles, DB connections leak | Always `with open()`, `async with pool.acquire()` |
| **Synchronous in async code** | `requests.get()` in an async function blocks the event loop | Use `httpx.AsyncClient` or run sync in `asyncio.to_thread()` |
| **No type hints** | Runtime errors from wrong types | Type everything, run `mypy --strict` |
| **Print debugging** | `print("here")` scattered everywhere | Use `structlog` with proper levels: debug, info, warning, error |

### Infrastructure Anti-Patterns

| Anti-Pattern | What Goes Wrong | What to Do Instead |
|---|---|---|
| **Secrets in code/config** | Leaked API keys in GitHub | Environment variables, Secret Manager, never commit `.env` |
| **No health checks** | Dead containers receive traffic | Liveness + readiness probes on every service |
| **Latest tag in Docker** | Can't reproduce builds | Tag with git SHA: `image:abc123` |
| **No resource limits** | One pod eats all cluster memory | Set CPU/memory requests AND limits |
| **Manual deployment** | "It works on my machine" | Docker + CI/CD + IaC. Every deployment is automated. |
| **No rollback plan** | Bad deploy → stuck in broken state | Blue-green or canary deployment, automated rollback |
| **Logging to stdout only** | Can't search or aggregate | Structured JSON logs → centralized logging (ELK, CloudWatch) |

---

## Best Practice Sources and References

### Architecture & Design
- **Martin Fowler's Blog** (martinfowler.com) — patterns, refactoring, architecture. Read: "Microservices", "StranglerFig"
- **The Twelve-Factor App** (12factor.net) — 12 principles for building SaaS. Essential for production systems.
- **Google SRE Book** (free online: sre.google/sre-book) — Ch 1-4 on reliability, SLOs, error budgets
- **DDIA (Designing Data-Intensive Applications)** — the bible for data systems understanding

### Python Best Practices
- **Fluent Python, 2nd Edition** (O'Reilly, Luciano Ramalho) — the deepest Python book. Read Ch 17-21 on concurrency when you need it.
- **Architecture Patterns with Python** (O'Reilly, Percival & Gregory) — domain-driven design in Python. Read Ch 1-4 for repository pattern, unit of work.
- **Effective Python, 2nd Edition** (O'Reilly, Brett Slatkin) — 90 specific ways to write better Python. Read items on generators, decorators, concurrency.
- **Cosmic Python** (cosmicpython.com — free online) — DDD, ports and adapters, event-driven architecture in Python

### MLOps & Production ML
- **Google MLOps Whitepaper** — "MLOps: Continuous delivery for ML on Google Cloud". Levels 0, 1, 2 of MLOps maturity.
- **Made With ML** (madewithml.com) — Goku Mohandas's free MLOps course. Excellent practical content.
- **Evidently AI Blog** — monitoring, drift detection, production ML patterns
- **DataTalks Club MLOps Zoomcamp** (free) — structured MLOps course, run as background learning

### Docker & Kubernetes
- **Docker Deep Dive** (Nigel Poulton) — short, practical, constantly updated
- **Kubernetes Patterns** (O'Reilly, Ibryam & Huß) — design patterns for K8s. Read Ch on sidecar, ambassador, adapter.
- **The Kubernetes Book** (Nigel Poulton) — more accessible than K8s Up and Running
- **Kelsey Hightower's "Kubernetes the Hard Way"** (GitHub) — build K8s from scratch. Do this only if you want deep infra understanding.

### Security
- **OWASP Top 10 for LLM Applications** (owasp.org) — memorize this
- **Practical Cloud Security** (O'Reilly, Dotson) — cloud security fundamentals
- Simon Willison's blog on prompt injection — the definitive practitioner resource

---

## Standout Skills That Make You an Immediate Hire

### The "Nobody Else Has This" Differentiators

**1. Evaluation as a First-Class Skill**
- Golden dataset creation and curation process
- Ragas metrics with actual numbers in your README
- CI pipeline that blocks deployment when quality drops
- Comparison table showing experiment history with decisions

**2. Structured Extraction at Scale**
- Upload PDF → extract typed Pydantic fields → validate → output
- Accuracy metrics on a test set
- This is where companies make money from AI (not chatbots)

**3. Reliable Agents (not demo agents)**
- Router accuracy: 95%+ on a labeled test set
- Fallback paths, cost controls, human escalation
- Evaluation proving reliability, not just "it works in the demo"

**4. Full LLMOps Pipeline**
- Automated ingestion → experiment tracking → CI/CD eval gates → monitoring → auto-rollback
- This complete loop in one GitHub repo puts you in top 5% of applicants

**5. Cost Awareness**
- Cost per query calculated and displayed
- Token usage optimization
- Your Cloudaccel story: $0.02 → $0.003 (85% reduction)
- Interviewers love: "this system costs $X at Y scale, here's how I'd reduce it"

### The GitHub README That Gets You Hired

```
## Architecture
[diagram]

## LLMOps Pipeline
- Automated ingestion: Prefect flow, daily schedule, incremental processing
- Experiment tracking: MLflow for all prompt/retrieval/model configs
- CI/CD: GitHub Actions — Ragas eval on every PR, blocks merge if quality drops
- Monitoring: nightly eval on 50 sampled queries, auto-rollback on regression

## Evaluation Results (v2.3)
| Metric              | Score  |
|---------------------|--------|
| Faithfulness        | 0.91   |
| Answer Relevance    | 0.88   |
| Context Precision   | 0.85   |
| Avg Latency (p95)   | 1.2s   |
| Cost per Query      | $0.003 |

## Experiment History
| Version | Change                  | Faithfulness | Decision |
|---------|-------------------------|-------------|----------|
| v2.3    | Added reranker (Cohere) | 0.91        | Shipped  |
| v2.2    | Chunk size 256→512      | 0.82        | Reverted |
| v2.1    | Hybrid search           | 0.87        | Shipped  |
| v2.0    | Baseline                | 0.84        | Baseline |

## Quick Start
docker-compose up  # that's it
```

---

## Daily DSA Tracker (42 Problems in 28 Days)

### Week 1
| Day | Problems | Pattern |
|---|---|---|
| 1 | Two Sum, Valid Anagram | Arrays, Hashmaps |
| 2 | Group Anagrams, Top K Frequent Elements | Hashmaps |
| 3 | Product of Array Except Self, Encode/Decode Strings | Arrays |
| 4 | Valid Parentheses, Min Stack | Stacks |
| 5 | Binary Search, Search Rotated Sorted Array | Binary Search |
| 6 | Invert Binary Tree, Max Depth | Trees |
| 7 | (Docker day — skip or do 1 easy) | — |

### Week 2
| Day | Problems | Pattern |
|---|---|---|
| 8 | LRU Cache | Hashmaps + Linked Lists |
| 9 | Clone Graph, Number of Islands | BFS/DFS |
| 10 | Course Schedule | Topological Sort |
| 11 | Longest Substring Without Repeating | Sliding Window |
| 12 | Container With Most Water | Two Pointers |
| 13 | Merge Intervals, Insert Interval | Intervals |
| 14 | (Monitoring day — do 1 easy) | — |

### Week 3
| Day | Problems | Pattern |
|---|---|---|
| 15 | Kth Smallest in BST, Validate BST | Trees |
| 16 | Word Search, Combination Sum | Backtracking |
| 17 | Implement Trie | Trie |
| 18 | Longest Consecutive Sequence | Hashmaps |
| 19 | Meeting Rooms II | Intervals + Heaps |
| 20 | Design Twitter | Design (multi-structure) |
| 21 | (Application day — do 1 easy) | — |

### Week 4
| Day | Problems | Pattern |
|---|---|---|
| 22 | Alien Dictionary | Topological Sort (hard) |
| 23 | Median of Two Sorted Arrays (optional) | Binary Search (hard) |
| 24 | Climbing Stairs, House Robber | DP |
| 25 | 4-5 problems from weakest patterns | Mixed |
| 26 | Coin Change | DP |
| 27 | Merge K Sorted Lists | Heaps |
| 28 | (Mock interview day) | — |

---

## The Mindset

This plan works because it's built on one principle: **proof over preparation.**

Every day you create something that didn't exist yesterday — a deployed endpoint, an evaluation score, a LinkedIn post, an application sent. That accumulation of proof is what gets you hired.

Not certificates. Not courses completed. Not books finished. Proof that you can build, evaluate, deploy, and explain AI systems.

You have 28 days. The clock starts tomorrow morning. Open your terminal and type:

```
mkdir knowledge-assistant && cd knowledge-assistant && git init
```

Go.
