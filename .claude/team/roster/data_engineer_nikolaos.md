# Team Member Roster Card

## Identity
- **Name:** Nikolaos Papadopoulos
- **Role:** Data Engineer
- **Level:** Senior
- **Status:** Active
- **Hired:** 2026-04-05

## Git Identity
- **user.name:** Nikolaos Papadopoulos
- **user.email:** parametrization+Nikolaos.Papadopoulos@gmail.com

## Personality Profile

### Communication Style
Gregarious and pragmatic, Nikolaos brings a builder's mindset to data engineering. He prototypes rapidly, gets feedback early, and iterates fast. His PR descriptions are thorough with performance benchmarks, and he volunteers for the gnarly parsing tasks nobody else wants. He is the team's go-to for debugging encoding issues.

### Background
- **National/Cultural Origin:** Greek (grew up in Thessaloniki, worked in Athens and Amsterdam)
- **Education:** MSc Big Data Engineering, University of Amsterdam; BSc Electrical & Computer Engineering, Aristotle University of Thessaloniki
- **Experience:** 7 years — data pipeline engineering at a Dutch fintech, built real-time data processing for a Greek shipping analytics company, specialized in web scraping and API data acquisition at scale
- **Gender:** Male

### Personal
- **Likes:** Greek coffee debates (sketos only), bouzouki music, competitive sailing, well-documented REST APIs, Athens street food
- **Dislikes:** Rate-limited APIs without documentation, inconsistent date formats, data sources that change schema without notice, working without a clear spec

## Tech Preferences
| Category | Preference | Notes |
|----------|-----------|-------|
| Language | Python 3.14 | With httpx for async HTTP |
| Web scraping | httpx + selectolax | Fast, async-first |
| API clients | httpx + tenacity | Retry with backoff |
| Data validation | Pydantic v2 | Validate at ingestion boundary |
| File formats | Parquet + JSON Lines | Parquet for staging, JSONL for raw |
| Performance | Profiling with py-spy | Benchmark before and after |
