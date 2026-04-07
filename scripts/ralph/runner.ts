#!/usr/bin/env bun
/**
 * Ralph Agent Loop Runner
 * Runs Claude Code in a loop to implement user stories
 * Adapted for Python ML research projects (no web server required)
 */

import { spawnSync } from "child_process";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { join } from "path";

const PROJECT_ROOT = join(import.meta.dir, "../..");
const PROMPT_FILE = join(import.meta.dir, "prompt.md");
const LOG_FILE = join(import.meta.dir, "log.md");
const MAX_ITERATIONS = 20;

// Parse optional --prompt argument
const extraPrompt = process.argv.includes("--prompt")
  ? process.argv[process.argv.indexOf("--prompt") + 1]
  : "";

// Initialize log file if it doesn't exist
if (!existsSync(LOG_FILE)) {
  writeFileSync(
    LOG_FILE,
    `# Ralph Loop Log\n\n_No iterations completed yet._\n`,
    "utf-8"
  );
}

const basePrompt = readFileSync(PROMPT_FILE, "utf-8");
const fullPrompt = extraPrompt
  ? `${basePrompt}\n\n## Extra Context\n${extraPrompt}`
  : basePrompt;

console.log("🤖 Starting Ralph Agent Loop");
console.log(`📁 Project: ${PROJECT_ROOT}`);
console.log(`🔄 Max iterations: ${MAX_ITERATIONS}`);
console.log("─".repeat(60));

for (let i = 1; i <= MAX_ITERATIONS; i++) {
  console.log(`\n🔁 Iteration ${i}/${MAX_ITERATIONS}`);

  // Check if all stories pass before each iteration
  const verifyResult = spawnSync(
    "bun",
    ["run", "scripts/verify-user-stories.ts"],
    {
      cwd: PROJECT_ROOT,
      stdio: "pipe",
      encoding: "utf-8",
    }
  );

  if (verifyResult.status === 0) {
    console.log("\n🎉 All user stories passing! Loop complete.");
    process.exit(0);
  }

  console.log("⏳ Stories remaining, invoking Claude Code agent...\n");

  // Run Claude Code CLI (non-interactive, single pass)
  const result = spawnSync(
    "claude",
    ["--print", "--max-turns", "30", fullPrompt],
    {
      cwd: PROJECT_ROOT,
      stdio: "inherit",
      encoding: "utf-8",
      timeout: 600_000, // 10 min per iteration
    }
  );

  if (result.status !== 0 && result.status !== null) {
    console.error(`\n❌ Agent exited with code ${result.status}`);
    if (i < MAX_ITERATIONS) {
      console.log("Retrying next iteration...");
    }
  }

  console.log(`\n✓ Iteration ${i} complete`);
}

console.log(`\n⚠️  Reached max iterations (${MAX_ITERATIONS}). Check stories.`);
process.exit(1);
