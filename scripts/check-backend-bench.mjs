#!/usr/bin/env node

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

function usage() {
  console.log(`Usage: node scripts/check-backend-bench.mjs --file <path> [checks...]

Checks:
  --require <scenarioId:backendId>
  --min-samples <scenarioId:backendId:count>
  --min-speedup <scenarioId:backendId:factor>
`);
}

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(1);
}

function parsePair(value, flag) {
  const parts = value.split(":");
  if (parts.length < 2) {
    fail(`${flag} requires scenarioId:backendId`);
  }
  return {
    scenarioId: parts[0],
    backendId: parts.slice(1).join(":"),
  };
}

function parseSamples(value) {
  const parts = value.split(":");
  if (parts.length < 3) {
    fail("--min-samples requires scenarioId:backendId:count");
  }
  const count = Number.parseInt(parts.pop(), 10);
  if (!Number.isFinite(count) || count <= 0) {
    fail(`invalid --min-samples count: ${value}`);
  }
  return {
    scenarioId: parts[0],
    backendId: parts.slice(1).join(":"),
    count,
  };
}

function parseSpeedup(value) {
  const parts = value.split(":");
  if (parts.length < 3) {
    fail("--min-speedup requires scenarioId:backendId:factor");
  }
  const factor = Number.parseFloat(parts.pop());
  if (!Number.isFinite(factor) || factor <= 0) {
    fail(`invalid --min-speedup factor: ${value}`);
  }
  return {
    scenarioId: parts[0],
    backendId: parts.slice(1).join(":"),
    factor,
  };
}

function parseArgs(argv) {
  const options = {
    file: null,
    require: [],
    minSamples: [],
    minSpeedup: [],
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case "--file":
        i++;
        if (i >= argv.length) fail("--file requires a path");
        options.file = resolve(process.cwd(), argv[i]);
        break;
      case "--require":
        i++;
        if (i >= argv.length) fail("--require requires a scenarioId:backendId pair");
        options.require.push(parsePair(argv[i], "--require"));
        break;
      case "--min-samples":
        i++;
        if (i >= argv.length) fail("--min-samples requires scenarioId:backendId:count");
        options.minSamples.push(parseSamples(argv[i]));
        break;
      case "--min-speedup":
        i++;
        if (i >= argv.length) fail("--min-speedup requires scenarioId:backendId:factor");
        options.minSpeedup.push(parseSpeedup(argv[i]));
        break;
      case "--help":
      case "-h":
        usage();
        process.exit(0);
      default:
        fail(`unknown option: ${arg}`);
    }
  }

  if (options.file === null) {
    fail("--file is required");
  }

  return options;
}

function resultKey(result) {
  return `${result.scenarioId}\0${result.backendId}`;
}

function getResult(resultMap, scenarioId, backendId) {
  return resultMap.get(`${scenarioId}\0${backendId}`) ?? null;
}

function assertOkResult(result, description) {
  if (result === null) {
    fail(`missing benchmark result for ${description}`);
  }
  if (result.status !== "ok") {
    fail(`${description} did not succeed: ${result.reason ?? result.status}`);
  }
}

function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  const payload = JSON.parse(readFileSync(options.file, "utf8"));
  const results = Array.isArray(payload.results) ? payload.results : [];
  const resultMap = new Map(results.map(result => [resultKey(result), result]));

  for (const requirement of options.require) {
    const result = getResult(
      resultMap,
      requirement.scenarioId,
      requirement.backendId
    );
    assertOkResult(
      result,
      `${requirement.scenarioId}:${requirement.backendId}`
    );
  }

  for (const minimum of options.minSamples) {
    const result = getResult(resultMap, minimum.scenarioId, minimum.backendId);
    assertOkResult(result, `${minimum.scenarioId}:${minimum.backendId}`);
    if ((result.iterations ?? 0) < minimum.count) {
      fail(
        `${minimum.scenarioId}:${minimum.backendId} produced ${result.iterations ?? 0} samples; expected at least ${minimum.count}`
      );
    }
  }

  for (const minimum of options.minSpeedup) {
    const result = getResult(resultMap, minimum.scenarioId, minimum.backendId);
    assertOkResult(result, `${minimum.scenarioId}:${minimum.backendId}`);
    if ((result.speedupVsTs ?? 0) < minimum.factor) {
      fail(
        `${minimum.scenarioId}:${minimum.backendId} speedup ${result.speedupVsTs ?? 0}x is below ${minimum.factor}x`
      );
    }
  }

  console.log(`Benchmark checks passed for ${options.file}`);
}

main();
