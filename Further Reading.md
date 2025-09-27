# Anomaly Investigation Workbench: An AI-Powered Tool for Graph Root Cause Analysis

## 1. Project Mission & Scope

**Mission:** To develop an intelligent tool that moves beyond simple anomaly *detection* and empowers Swisscom network engineers to perform rapid **root cause analysis** on a live, evolving knowledge graph.

**Project Scope:**
The goal of this hackathon project was to build a complete, end-to-end proof-of-concept that covers:
1.  **Unsupervised Anomaly Scoring:** Training a state-of-the-art Temporal Graph Network (TGN) to understand "normal" graph behavior and assign meaningful anomaly scores to every event without relying on pre-existing labels.
2.  **Data Processing Pipeline:** Creating a robust pipeline to process the model's output into aggregated, actionable datasets.
3.  **Interactive Investigation UI:** Building a user-centric workbench that guides an engineer through a logical investigation funnel, from high-level triage to deep-dive analysis.
4.  **AI Integration:** Demonstrating how Large Language Models (LLMs) can be integrated to both control the UI via natural language and provide expert-level analysis, transforming the workbench into an AIOps platform.

## 2. Core Objectives

This project was designed to meet the key objectives of the challenge:

*   ✅ **Spot Anomalies in a Dynamic Graph:** Our TGN model successfully assigns an anomaly score to every `add` and `remove` event over time.
*   ✅ **Analyze Node-Level Anomalies:** The workbench isolates "Top Offender" nodes, allowing engineers to immediately identify potentially faulty devices or misconfigured components.
*   ✅ **Analyze Edge-Level Anomalies:** The core of our model is identifying wrong or missing relationships. The UI allows for deep dives into these specific events.
*   ✅ **Analyze Graph Structure Changes:** The interactive timeline and dynamic filtering allow engineers to spot and investigate unusual patterns and sudden bursts of anomalous activity.
*   ✅ **LLM Interface:** We have implemented two powerful LLM integrations to showcase a next-generation user experience.
*   ✅ **Unsupervised Setting:** The entire system is built on an unsupervised model trained on a "clean" dataset, fulfilling the core requirement.

## 3. System Architecture: Real-Time Ready

The current system is built around a powerful concept: the separation of the **Scoring Engine** from the **Investigation UI**.

*   **Current Implementation (Hackathon Scope):**
    For this challenge, we trained our TGN model and used it to score the provided dataset, generating a set of static files (`anomalies.jsonl`, `nodes.json`). The workbench UI is then served these static files. This is a robust and performant approach for analyzing a fixed dataset.

*   **Real-Time Ready Architecture:**
    This architecture was intentionally designed to be **real-time ready**. The static files are simply a snapshot of what a live data stream would produce. In a production environment, the system would work as follows:
    1.  A **real-time data ingestion pipeline** would feed new graph events (adds/removes) directly to our trained TGN model.
    2.  The model would **score each event in real-time**, appending the new, labeled data points to a live database or stream.
    3.  The Investigation Workbench would connect to this live data source instead of the static files.

    The core logic of the model and the UI would remain unchanged. This part of the implementation was deemed out of scope due to the time constraints of the hackathon, but the system is fundamentally built for this evolution.

## 4. AI Integration: The Co-Pilot and the Controller

We have integrated AI in two distinct and powerful ways to demonstrate the future of AIOps.

#### a) The AI Controller: Natural Language Queries

The "Ask AI" search bar demonstrates how an LLM can function as a **natural language interface to a complex tool**.

*   **Current Implementation:** When a user types a query like "show me urgent problems," the UI makes a **real API call** to the Swiss AI platform. The LLM is prompted to understand the user's intent and respond with a structured JSON command, which we call a "Tool Call." The UI then parses this JSON and executes the corresponding action (e.g., applying the P99 filter).
*   **Future Vision:** This is a mockup of a powerful paradigm. In a production system, the LLM would have a rich set of tools it could call, allowing engineers to perform complex filtering and analysis with simple sentences, dramatically lowering the barrier to entry for deep data exploration.

#### b) The AI Co-Pilot: Root Cause Analysis

The "Ask AI Co-Pilot" button inside the investigation storyboard demonstrates how an LLM can act as an **expert analyst**.

*   **Current Implementation:** When activated, the UI sends the **context-specific, automated diagnosis** to the Swiss AI API. We use an advanced system prompt that instructs the AI to act as a network expert and provide a confident root cause hypothesis and actionable next steps, based *only* on the data provided.
*   **Future Vision & Limitations:** This feature is incredibly powerful, but its accuracy is directly tied to the quality of its training data. For a production-grade Co-Pilot, this model would need to be **fine-tuned on domain-specific data** from Swisscom, such as internal incident reports, network topology documentation, and historical outage analyses. With access to such data, the Co-Pilot could move from providing educated hypotheses to delivering highly accurate, context-aware diagnostic insights that could resolve incidents in a fraction of the time.

---