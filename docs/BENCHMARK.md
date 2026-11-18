# PReMind-Bench

The benchmark, its fields, licenses and loading instructions are documented in the dataset card
(`Yeongtak/PReMind-Bench` on Hugging Face). Summary:

- **Public split:** `test`, 1,132 rows (1,020 main requests and 112 test-only rows); `conflict` is a 152-row subset
  (140 interdependency cases, 63 with seen and 77 with unseen hazard types, and 12 safe controls).
- **Memory:** each row shows the four retrieved memory entries of the online top-4 retrieval.
- **Requests** differ in the evidence they need (personal fact, safety knowledge, both, or the scene only) and in
  situational risk (safe or unsafe), which vary independently.
- **Splits:** training and test use disjoint capture scenes and target persons; objects and pets may recur. Personal
  facts are fictitious. Train and dev are not public (available on request).
