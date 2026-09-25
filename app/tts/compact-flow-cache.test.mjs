import assert from 'node:assert/strict';
import test from 'node:test';
import { compactFlowModel, requiredFlowCapacity } from '../../public/tts-onnx/compact-flow-cache.js';

test('supported voice and text leave room for the entire generation', () => {
  assert.equal(requiredFlowCapacity(189, 50, 240), 512);
  assert.equal(requiredFlowCapacity(128, 144, 240), 512);
});

test('restores original capacity instead of shortening context or output', () => {
  assert.equal(requiredFlowCapacity(129, 144, 240), 1000);
  assert.equal(requiredFlowCapacity(700, 50, 240), 1000);
  // Preserve the original model behavior even if reaching the token/frame cap
  // would overflow: it can still finish earlier through EOS.
  assert.equal(requiredFlowCapacity(900, 50, 240), 1000);
});

test('invalid cache lengths fail rather than accidentally selecting a small cache', () => {
  for (const value of [NaN, Infinity, -1, 1.5]) {
    assert.throws(() => requiredFlowCapacity(value, 50, 240), /Invalid/);
  }
});

test('unknown models are left untouched', async () => {
  const small = new Uint8Array([0xe8, 7, 0xe8, 7]);
  assert.equal(await compactFlowModel(small.buffer), false);
  assert.deepEqual([...small], [0xe8, 7, 0xe8, 7]);
  // Even a file of the expected size must pass the full cryptographic digest.
  const sameSize = new Uint8Array(76341079);
  sameSize[76178383] = 0xe8;
  sameSize[76178384] = 7;
  assert.equal(await compactFlowModel(sameSize.buffer), false);
  assert.equal(sameSize[76178383], 0xe8);
  assert.equal(sameSize[76178384], 7);
});
