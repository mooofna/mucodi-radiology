// Hard-links the built demo assets (demo/.local/assets) into static/assets for dev and build.
// Hard links cost no extra disk space; nothing here is tracked by git.
import { existsSync, linkSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const source = process.env.MUCODI_ASSETS ?? join(web, '../demo/.local/assets');
const target = join(web, 'static/assets');
if (!existsSync(source)) {
  console.error(`no assets at ${source}; run the demo pipeline first or set MUCODI_ASSETS`);
  process.exit(1);
}
rmSync(target, { recursive: true, force: true });
let count = 0;
/** @param {string} dir @param {string} rel */
function walk(dir, rel) {
  for (const name of readdirSync(dir)) {
    const src = join(dir, name);
    const dst = join(target, rel, name);
    if (statSync(src).isDirectory()) walk(src, join(rel, name));
    else {
      mkdirSync(dirname(dst), { recursive: true });
      linkSync(src, dst);
      count += 1;
    }
  }
}
walk(source, '');
console.log(`linked ${count} asset files into static/assets`);
