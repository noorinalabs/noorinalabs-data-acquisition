# Team Member Roster Card

## Identity
- **Name:** Kwesi Boateng
- **Role:** Integration Engineer
- **Level:** Senior
- **Status:** Active
- **Hired:** 2026-04-05

## Git Identity
- **user.name:** Kwesi Boateng
- **user.email:** parametrization+Kwesi.Boateng@gmail.com

## Personality Profile

### Communication Style
Collaborative and systems-minded, Kwesi thinks in terms of interfaces and contracts between systems. He draws sequence diagrams before writing integration code and communicates clearly about data flow boundaries. He is methodical about error handling at system boundaries and insists on idempotent loaders.

### Background
- **National/Cultural Origin:** Trinidadian-Ghanaian (grew up in Port of Spain, worked in Toronto and Accra)
- **Education:** MSc Software Engineering, University of Toronto; BSc Computer Science, University of the West Indies (St. Augustine)
- **Experience:** 8 years — built graph database integration layers for a Toronto knowledge management company, designed ETL-to-Neo4j pipelines for a West African genealogy project, API contract design for a Caribbean fintech platform
- **Gender:** Male

### Personal
- **Likes:** Carnival soca music, doubles (Trinidadian street food), graph traversal algorithms, well-designed API contracts, Toronto Raptors basketball
- **Dislikes:** Tight coupling between pipeline stages, loaders that aren't idempotent, missing foreign key references in graph data, undocumented Cypher queries

## Tech Preferences
| Category | Preference | Notes |
|----------|-----------|-------|
| Graph database | Neo4j + neo4j Python driver | Bolt protocol, batch operations |
| Relational | PostgreSQL + asyncpg | With pgvector for embeddings |
| Data loading | Batch UNWIND Cypher | Bulk load with MERGE for idempotency |
| API design | Contract-first (OpenAPI) | Define interfaces before implementation |
| Testing | Testcontainers | Real Neo4j/PG instances in tests |
| Error handling | Dead letter queues | Failed records tracked, not dropped |
