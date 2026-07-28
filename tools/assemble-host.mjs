// Assemble the browser host from the packages that own its pieces.
//
// The toolchain resolves `.almd` modules from dependencies but has no
// equivalent for web host assets — `native/` is native-only and hardcoded — so
// the JS and WGSL a page serves have to be copied. Copying by hand is how a
// downstream host once diverged from the package that owns it and a bug got
// fixed in the copy instead of the source.
//
// This makes the copy mechanical, and records where every file came from so the
// next reader can tell. `--check` re-verifies without writing, which is what CI
// should run: it fails if a served file no longer matches its source, in either
// direction.
//
// Usage:
//   node tools/assemble-host.mjs <dest> <pkg-host-dir>...
//   node tools/assemble-host.mjs --check <dest> <pkg-host-dir>...

import { readFile, writeFile, mkdir, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { join, resolve, basename, dirname } from "node:path";

const run = promisify(execFile);

const args = process.argv.slice(2);
const check = args[0] === "--check";
const [dest, ...pkgs] = check ? args.slice(1) : args;

if (!dest || pkgs.length === 0) {
  console.error("usage: node tools/assemble-host.mjs [--check] <dest> <pkg-host-dir>...");
  process.exit(2);
}

const sha = (buf) => createHash("sha256").update(buf).digest("hex").slice(0, 16);

async function commitOf(dir) {
  try {
    const { stdout } = await run("git", ["-C", dir, "rev-parse", "--short", "HEAD"]);
    return stdout.trim();
  } catch {
    return "unknown";
  }
}

async function manifestOf(hostDir) {
  const path = join(hostDir, "MANIFEST");
  if (!existsSync(path)) throw new Error(`${hostDir} has no MANIFEST`);
  const text = await readFile(path, "utf8");
  return text.split("\n").map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));
}

const problems = [];
const provenance = [];

await mkdir(dest, { recursive: true });

for (const hostDir of pkgs) {
  if (!existsSync(hostDir)) {
    problems.push(`${hostDir} not found — is the package checked out?`);
    continue;
  }
  const pkg = basename(resolve(hostDir, ".."));
  const commit = await commitOf(hostDir);
  for (const name of await manifestOf(hostDir)) {
    const from = join(hostDir, name);
    const to = join(dest, name);
    if (!existsSync(from)) { problems.push(`${pkg}: ${name} listed in MANIFEST but missing`); continue; }
    const src = await readFile(from);
    if (check) {
      if (!existsSync(to)) { problems.push(`${name} missing from ${dest}`); continue; }
      const have = await readFile(to);
      if (!have.equals(src)) {
        problems.push(`${name} differs from ${pkg} (${commit}) — the served copy has drifted`);
        continue;
      }
    } else {
      await mkdir(dirname(to), { recursive: true });
      await writeFile(to, src);
    }
    provenance.push(`${name.padEnd(16)} ${pkg}@${commit} ${sha(src)}`);
  }
}

// Anything served that neither a package nor the consumer claims is a fork
// waiting to happen. The consumer declares its OWN files in `<dest>/OWNED`;
// without that the app's own code reads as unclaimed.
if (check && problems.length === 0) {
  const claimed = new Set(provenance.map((l) => l.split(/\s+/)[0]));
  const ownedPath = join(dest, "OWNED");
  if (existsSync(ownedPath)) {
    const text = await readFile(ownedPath, "utf8");
    for (const l of text.split("\n").map((x) => x.trim())) {
      if (l && !l.startsWith("#")) claimed.add(l);
    }
  }
  const served = await readdir(dest).catch(() => []);
  for (const f of served) {
    if (f.startsWith(".") || f === "OWNED") continue;
    if (/\.(js|wgsl|ttf)$/.test(f) && !claimed.has(f)) {
      problems.push(`${f} is served but claimed by neither a package MANIFEST nor ${dest}/OWNED`);
    }
  }
}

const header = `# Assembled by tools/assemble-host.mjs — do not edit these files here.\n` +
  `# Fix them in the package that owns them, then re-run the assembler.\n`;

if (check) {
  if (problems.length) {
    console.error("host assembly is out of date:\n");
    for (const p of problems) console.error(`  · ${p}`);
    console.error("\nrun: node tools/assemble-host.mjs " + [dest, ...pkgs].join(" "));
    process.exit(1);
  }
  console.log(`host matches its sources — ${provenance.length} files`);
} else {
  if (problems.length) {
    console.error("assembly failed:\n");
    for (const p of problems) console.error(`  · ${p}`);
    process.exit(1);
  }
  await writeFile(join(dest, ".provenance"), header + provenance.join("\n") + "\n");
  console.log(`assembled ${provenance.length} files into ${dest}`);
  for (const l of provenance) console.log("  " + l);
}
