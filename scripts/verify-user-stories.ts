#!/usr/bin/env bun
/**
 * User Stories Verification Script
 * Validates all JSON story files in docs/user-stories/
 */

import { z } from "zod";
import { readdirSync, readFileSync } from "fs";
import { join } from "path";

const StorySchema = z.object({
  description: z.string().min(1),
  steps: z.array(z.string()).min(1),
  passes: z.boolean(),
});

const StoriesFileSchema = z.array(StorySchema);

const storiesDir = join(import.meta.dir, "../docs/user-stories");

let totalStories = 0;
let passingStories = 0;
let failingStories = 0;
let validationErrors = 0;

try {
  const files = readdirSync(storiesDir).filter((f) => f.endsWith(".json"));
  console.log(`\n📋 Found ${files.length} story files\n`);

  for (const file of files.sort()) {
    const filePath = join(storiesDir, file);
    const content = JSON.parse(readFileSync(filePath, "utf-8"));

    const result = StoriesFileSchema.safeParse(content);
    if (!result.success) {
      console.error(`❌ Schema error in ${file}:`);
      console.error(result.error.format());
      validationErrors++;
      continue;
    }

    console.log(`📄 ${file}`);
    for (const story of result.data) {
      totalStories++;
      const status = story.passes ? "✅ PASS" : "⏳ TODO";
      console.log(`  ${status} ${story.description}`);
      if (!story.passes) {
        failingStories++;
        console.log(`       Steps (${story.steps.length}):`);
        for (const step of story.steps.slice(0, 3)) {
          console.log(`         - ${step}`);
        }
        if (story.steps.length > 3) {
          console.log(`         ... +${story.steps.length - 3} more`);
        }
      } else {
        passingStories++;
      }
    }
    console.log();
  }

  console.log("─".repeat(60));
  console.log(`📊 Summary:`);
  console.log(`   Total stories:  ${totalStories}`);
  console.log(`   ✅ Passing:      ${passingStories}`);
  console.log(`   ⏳ Todo:         ${failingStories}`);
  if (validationErrors > 0) {
    console.log(`   ❌ Errors:       ${validationErrors}`);
  }
  console.log(
    `   Progress:       ${Math.round((passingStories / totalStories) * 100)}%`
  );

  if (failingStories === 0 && validationErrors === 0) {
    console.log("\n🎉 All stories passing!");
    process.exit(0);
  } else {
    console.log(`\n⏳ ${failingStories} stories remaining`);
    process.exit(failingStories > 0 ? 1 : 0);
  }
} catch (err) {
  console.error("Error reading stories:", err);
  process.exit(1);
}
