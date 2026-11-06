---
name: langgraph-report
description: Read a langgraph.json and report the capability names it would become. Does not register them.
license: Apache-2.0
---

# langgraph.report

Read the text of a `langgraph.json` plus a namespace. The result lists the capability names `namespace.<graph>` that `axi langgraph port` would register.

This file is the instruction document. The invocation contract is `axiom_ext_langgraph.skills.impl:report`. The skill does not write a registry and does not compile a graph.
