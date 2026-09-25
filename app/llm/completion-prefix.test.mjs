import assert from "node:assert/strict";
import test from "node:test";
import { buildCompletionMessages } from "./completion-logic.ts";
import { getCompletionCachePrefix } from "./completion-prefix.ts";

for (const includeExamples of [true, false]) {
  test(`cached prefix survives adding, changing, and clearing history (examples=${includeExamples})`, () => {
    // Exercise real prompt construction without a model download. Character
    // tokens make the text boundary observable independently of a vocabulary.
    const tokenize = (fragment, context) => Array.from(
      JSON.stringify(buildCompletionMessages(fragment, { context, includeExamples })),
      character => BigInt(character.codePointAt(0)),
    );
    const prefix = getCompletionCachePrefix(tokenize);
    assert.ok(prefix.length > 8);
    const histories = [
      undefined,
      [{ transcript: "I am moving to Paris", completions: ["next spring"] }],
      [
        { transcript: "Yesterday I felt differently", completions: [] },
        { transcript: "Now I think", completions: ["I need a break", "I can keep going"] },
      ],
      [],
    ];
    for (const context of histories) {
      for (const fragment of ["I should start looking for", "", "étonnamment, I", "\"hello\""]) {
        const full = tokenize(fragment, context);
        assert.deepEqual(full.slice(0, prefix.length), prefix);
        assert.deepEqual([...prefix, ...full.slice(prefix.length)], full);
      }
    }
  });
}

test("leaves two tokens before the first variable token, including unequal probe lengths", () => {
  const shared = Array.from({ length: 12 }, (_, index) => BigInt(index));
  const prefix = getCompletionCachePrefix((fragment, context) =>
    context ? shared : [...shared, fragment.startsWith("alpha") ? 30n : 31n],
  );
  assert.deepEqual(prefix, shared.slice(0, 10));
});

test("refuses a missing or too short common prefix", () => {
  assert.throws(() => getCompletionCachePrefix(() => []), /no safely cacheable prefix/);
  assert.throws(() => getCompletionCachePrefix((fragment, context) =>
    [context ? 2n : 1n, ...Array(20).fill(3n)],
  ), /no safely cacheable prefix/);
});
