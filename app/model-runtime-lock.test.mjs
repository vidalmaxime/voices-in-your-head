import assert from "node:assert/strict";
import test from "node:test";

import { acquireModelRuntimeLease, MODEL_RUNTIME_LOCK_NAME } from "./model-runtime-lock.ts";

class FakeLockManager {
  held = false;
  requests = [];

  async request(name, options, callback) {
    this.requests.push({ name, options });
    if (this.held) {
      await callback(null);
      return;
    }

    this.held = true;
    try {
      await callback({ name });
    } finally {
      this.held = false;
    }
  }
}

test("holds one exclusive model lease for the page lifetime", async () => {
  const manager = new FakeLockManager();
  const first = await acquireModelRuntimeLease(manager);
  assert.ok(first);
  assert.equal(manager.held, true);
  assert.deepEqual(manager.requests[0], {
    name: MODEL_RUNTIME_LOCK_NAME,
    options: { mode: "exclusive", ifAvailable: true },
  });

  const second = await acquireModelRuntimeLease(manager);
  assert.equal(second, null);

  first.release();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(manager.held, false);

  const third = await acquireModelRuntimeLease(manager);
  assert.ok(third);
  third.release();
});

test("fails closed when Web Locks are unavailable", async () => {
  assert.equal(await acquireModelRuntimeLease(undefined), null);
});
