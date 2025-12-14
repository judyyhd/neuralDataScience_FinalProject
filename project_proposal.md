# Cross-Region Information Flow During Visual Stimuli
## The Core Question
When a visual stimulus appears, how does neural activity cascade through the brain? Does V1 "talk to" V2, which then talks to higher areas? Or do multiple regions get activated in parallel?
## Conceptual Framework
Think of it like dropping a stone in a connected system of pools. The ripples don't just spread randomly - there are specific pathways. In the brain:
- V1 (primary visual cortex) receives direct input from thalamus
- Higher visual areas (V2, V4, etc.) receive input from V1 and each other
- Thalamus (LGN, LP) both sends and receives visual information
- Hippocampus receives highly processed information
## Specific Analyses
Phase 1: Temporal Dynamics
- For each brain region, compute average PSTH (peristimulus time histogram) after stimulus onset
- Measure response latency: when does each region first respond?
- Expected pattern: V1 → higher visual areas → hippocampus (with ~10-50ms delays)
Phase 2: Pairwise Relationships
- Cross-correlation between regions: does V1 activity at time t predict V2 activity at time t+delay?
- Granger causality: does V1 activity help predict future V2 activity, beyond what V2's own history tells you?
- Spike-triggered averages: when V2 fires, what was V1 doing 20ms earlier?
Phase 3: Information Transfer
- Mutual information: how much does knowing V1 activity reduce uncertainty about V2 activity?
- Transfer entropy: how much does V1's past help predict V2's future?
- Time-lagged decoding: train decoder on V1 activity to predict stimulus, then test if V2 activity at t+50ms contains the same information
Phase 4: Stimulus-Dependent Flow
- Does information flow differently for different stimuli (natural scenes vs. gratings)?
- Are some pathways stronger for certain visual features?
What Makes This Project Strong
- Uses unique dataset strength: Simultaneous multi-region recording
- Clear narrative arc: "We tracked how visual information flows through the brain"
- Builds on your skills: Population coding, spike analysis, curve fitting concepts
- Multiple difficulty levels: Start simple (latencies, correlations), add complexity as time permits
Potential Challenges
- Time-lagged analysis can be computationally intensive
- Need to handle different numbers of neurons per region
- Granger causality and transfer entropy have assumptions to verify