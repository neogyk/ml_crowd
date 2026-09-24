# Crowd ML

My first thoughts about this project:[url]( https://medium.com/@fusion-ai/massive-crowd-based-federated-pretraining-of-the-llm-on-the-edge-devices-c95ef8d01743)

Cross-device inference and finetuning over gRPC. A **master** node partitions a
gguf model into contiguous blocks of layers and allocates them to **slave**
devices (phones, edge nodes); activations then flow through the devices as a
pipeline.

```
                 ┌────────────┐
                 │   master   │  registry · partitioning · balancing
                 └─────┬──────┘
        ASSIGN + shard  │  gRPC
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ slave A │ ──► │ slave B │ ──► │ slave C │
   │ [0:12)  │     │ [12:20) │     │ [20:32) │
   └─────────┘     └─────────┘     └─────────┘
        └──────── activations ──────────┘
```

## Layout

| Path | What it holds |
| --- | --- |
| [proto/crowd.proto](ml_crowd/proto/crowd.proto) | The service contract: register, heartbeat, shard load, forward, backward |
| [master/server.py](ml_crowd/master/server.py) | The master gRPC server and its planning loop |
| [master/registry.py](ml_crowd/master/registry.py) | Which devices exist, and what they last reported |
| [master/pipeline.py](ml_crowd/master/pipeline.py) | Walks one request through the chain of stages |
| [slave/client.py](ml_crowd/slave/client.py) | The edge device: client to the master, server for its block |
| [slave/executor.py](ml_crowd/slave/executor.py) | Where a real runtime (llama.cpp, executorch) plugs in |
| [common/partition.py](ml_crowd/common/partition.py) | Splits layers into blocks weighted by device capability |
| [common/balance.py](ml_crowd/common/balance.py) | Decides when to re-partition; admits requests per device |
| [common/gguf.py](ml_crowd/common/gguf.py) | Reads gguf metadata and per-layer byte ranges |

## Running it

```bash
pip install -r ml_crowd/requirements.txt
make proto                                    # generate the gRPC stubs

make master MODEL=/path/to/model.gguf         # terminal 1
make slave PORT=50052                         # terminal 2
make slave PORT=50053                         # terminal 3
```

Each slave registers, reports its capabilities, and starts heartbeating. The
master's planner loop notices the membership change, re-partitions, and streams
each device the layers it now owns.

## Partitioning

Blocks are kept **contiguous** — the pipeline runs layers in order, so a device
owning a scattered set of layers would force extra round trips. Each device's
share is proportional to a capacity weight built from its FLOPs estimate,
discounted by current load and by low battery. Every registered device gets at
least one layer; if there are more devices than layers, the slowest are left
out of the plan.

## Balancing

The balancer re-partitions when membership changes (always) or when the slowest
stage exceeds the mean stage time by more than `imbalance_tolerance` (default
35%), subject to a cooldown so shards are not shuffled constantly. It also acts
as an admission gate, so one device is never asked to run two forward passes at
once.

## Plugging in a real runtime

`EchoExecutor` passes activations through unchanged, which is enough to exercise
the transport. To run actual compute, implement the four methods of
`BlockExecutor` — `load`, `forward`, `backward`, `release` — and pass an
instance to `Slave(...)`.

## Tests

```bash
make test     # 27 tests, no gRPC install needed
```

Partitioning, balancing, the registry and the gguf reader are covered without
the generated stubs, so they can be run on any machine.

## Not yet implemented

- **MQTT transport.** The design note names MQTT alongside gRPC; only gRPC is
  wired up here. MQTT would suit discovery and heartbeats on flaky mobile links,
  with gRPC kept for the activation path.
- **Gradient aggregation.** `PushGradients` accepts and counts chunks; the
  master does not yet average them or apply an optimizer step.
- **Transport security.** All channels are insecure. Any deployment off a
  trusted LAN needs TLS and device authentication.
