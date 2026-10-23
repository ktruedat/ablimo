# Sandbox image for the ablimo agent demo.
#
# Build stage has network access (to fetch crates); the RUN is launched with
# `--network none` (see run.sh), so the agent and its tools have no route off
# the machine even if a tool misbehaved. The inert canary only writes a marker.
#
# Build context is the repo root:  docker build -f docker/Dockerfile -t ablimo-agent .

# Build on bookworm so the binary's glibc matches the bookworm-slim runtime.
FROM rust:1-slim-bookworm AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release -p ablimo-agent

FROM debian:bookworm-slim
# Marker is written here; run.sh mounts a host dir over /out to observe it.
ENV ABLIMO_OUT=/out
RUN mkdir -p /out
COPY --from=build /app/target/release/ablimo-agent /usr/local/bin/ablimo-agent
ENTRYPOINT ["ablimo-agent"]
