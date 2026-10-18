# Architecture

This is a routing map for implemented contracts, not a required full reference read.
Read the section relevant to the task in [detailed contracts](docs/reference/ARCHITECTURE_DETAILS.md).
The [roadmap](BOARDGAME_AI_PROJECT_PLAN.md) describes planned capabilities; it does
not imply they already exist. [Information boundaries](INFORMATION_BOUNDARIES.md)
remain normative for observations, hidden state and learning/evaluation separation.

The engine core is independent of PyTorch. Engines own rules, chance and filtered
observations; agents receive visible observations and supplied legal actions.
Independent and shared learners currently coexist with distinct checkpoint formats.
Generic plugin-driven GPU learning remains unfinished.

## Package and dependencies

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#package-and-dependencies).

## Engine, records, and lifecycle

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#engine-records-and-lifecycle).

## Connect Four

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#connect-four).

## Guildworks rules and transitions

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#guildworks-rules-and-transitions).

## Guildworks recurrent learning

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#guildworks-recurrent-learning).

## Tic-Tac-Toe and baseline agents

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#tic-tac-toe-and-baseline-agents).

## Simulation, metrics, human play, and replays

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#simulation-metrics-human-play-and-replays).

## Observation, actions, and network

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#observation-actions-and-network).

## Collection, credit assignment, and PPO

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#collection-credit-assignment-and-ppo).

## Seeds, checkpoints, and resume

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#seeds-checkpoints-and-resume).

## Evaluation and analysis

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#evaluation-and-analysis).

## Development monitoring and performance

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#development-monitoring-and-performance).

## Experiment protocol identity

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#experiment-protocol-identity).

## Public search-target learning

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#public-search-target-learning).

## Frozen policy diagnosis

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#frozen-policy-diagnosis).

## Extension boundaries

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#extension-boundaries).

## Verification

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#verification).

## Public search evaluation

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#public-search-evaluation).

## Shared multi-game learning

[Read this contract](docs/reference/ARCHITECTURE_DETAILS.md#shared-multi-game-learning).

### Phase 7 paired evaluation and calibration preparation

[Reference](docs/reference/ARCHITECTURE_DETAILS.md#phase-7-paired-evaluation-and-calibration-preparation).
