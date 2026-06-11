#!/usr/bin/env python3
"""Render the loaded first-light graph (da#73) to a PNG from a live Neo4j.

Pulls the riyadussalihin Collection and its APPEARS_IN-connected Hadith nodes
from whatever ``NEO4J_*`` resolves to and draws them as a star graph — a
committable visual artifact of the real loaded data (the owner's "see real data"
moment) for environments without a Neo4j Browser session.

`matplotlib` + `networkx` are not project deps, so run it with `uv run --with`:

    uv run --with matplotlib --with networkx python scripts/first_light/render_graph.py

Prereq: run the load first (``scripts/first_light/run_slice.py`` against the same
Neo4j). Output: ``scripts/first_light/evidence/firstlight_graph.png``.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402

COLLECTION_ID = "col:sunnah:riyadussalihin"
OUTPUT = Path(__file__).resolve().parent / "evidence" / "firstlight_graph.png"


def main() -> None:
    """Render the live first-light graph to a PNG."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "testpassword")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        recs = session.run(
            "MATCH (h:Hadith)-[r:APPEARS_IN]->(c:Collection {id: $cid}) "
            "RETURN h.id AS hid, r.hadith_number_in_book AS num, c.name_en AS coll",
            cid=COLLECTION_ID,
        ).data()
    driver.close()

    if not recs:
        raise SystemExit(
            f"No APPEARS_IN edges for {COLLECTION_ID} — run the load first "
            "(scripts/first_light/run_slice.py)."
        )

    collection = recs[0]["coll"]
    graph = nx.Graph()
    graph.add_node("COLL", label=collection, kind="collection")
    for rec in recs:
        graph.add_node(rec["hid"], label=str(rec["num"]), kind="hadith")
        graph.add_edge("COLL", rec["hid"])

    pos = nx.spring_layout(graph, k=0.55, seed=7, iterations=120)
    hadith_nodes = [n for n, d in graph.nodes(data=True) if d["kind"] == "hadith"]

    plt.figure(figsize=(14, 14))
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=hadith_nodes,
        node_color="#4C9F70",
        node_size=620,
        edgecolors="#1b5e20",
        linewidths=0.6,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=["COLL"],
        node_color="#C0392B",
        node_size=2600,
        edgecolors="#7b1f17",
        linewidths=1.5,
    )
    nx.draw_networkx_edges(graph, pos, edge_color="#b0bec5", width=0.9)
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={n: graph.nodes[n]["label"] for n in hadith_nodes},
        font_size=7,
        font_color="white",
    )
    nx.draw_networkx_labels(
        graph, pos, labels={"COLL": collection}, font_size=12, font_color="white"
    )
    plt.title(
        f"first-light (da#73): {len(hadith_nodes)} Hadith --[:APPEARS_IN]--> "
        f"Collection '{collection}'\n"
        "live Neo4j 5, riyadussalihin book 1; edge labels = real hadith_number (da#72)",
        fontsize=12,
    )
    plt.axis("off")
    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUTPUT} ({len(hadith_nodes)} hadiths)")


if __name__ == "__main__":
    main()
