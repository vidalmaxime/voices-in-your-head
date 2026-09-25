import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { loadingLayout } from './loading-layout.ts';

// Use the dimensions of the actual cast, including the tallest and widest portraits.
const cast = JSON.parse(readFileSync(new URL('../../assets/chorus/new-cast/characters.json', import.meta.url)));
const aspects = cast.map(({width, height}) => width / height);

test('all loading portraits fit without overlap, including reflections and entrance motion', () => {
  const slots = loadingLayout(aspects);
  assert.equal(slots.length, 18);
  const bounds = slots.map(({x, y, width, character}) => {
    const height = width * (16 / 9) / aspects[character];
    return {left:x-width/2, right:x+width/2, top:y-height, bottom:y+height*0.64};
  });
  for (const [i, a] of bounds.entries()) {
    assert.ok(a.left >= 0 && a.right <= 100 && a.top >= 0 && a.bottom <= 100);
    for (const b of bounds.slice(i+1)) {
      assert.ok(a.right < b.left || b.right < a.left || a.bottom < b.top || b.bottom < a.top,
        `Character ${i+1} overlaps another portrait or reflection`);
    }
  }
});
