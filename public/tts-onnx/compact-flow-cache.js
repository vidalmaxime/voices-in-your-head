// Metadata-only patch for the exact english_2026-04 INT8 export. These 42
// protobuf Dimension.dim_value fields encode 1000 as e8 07; 512 is 80 04.
// Both varints have the same size. Operators, weights, and active KV entries
// are unchanged. Unknown exports always retain their original capacity.
export const FLOW_MODEL_SHA256 = 'f9bd8106b79a0192c1c43399ab938fb24900a95c1c599870d75a884e99000116';
const MODEL_BYTES = 76341079;
const OFFSETS = [
    76178383, 76178469, 76178555, 76178641, 76178730, 76178819,
    76178973, 76179071, 76179169, 76179267, 76179368, 76179469,
    76191875, 76192320, 76195506, 76195835, 76196285,
    76218442, 76218895, 76222115, 76222444, 76222902,
    76245085, 76245539, 76248761, 76249090, 76249549,
    76271741, 76272195, 76275417, 76275746, 76276205,
    76298397, 76298851, 76302073, 76302402, 76302861,
    76325053, 76325507, 76328729, 76329058, 76329517,
];

export async function compactFlowModel(buffer) {
    if (buffer.byteLength !== MODEL_BYTES || !globalThis.crypto?.subtle) return false;
    const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', buffer));
    const hex = Array.from(hash, byte => byte.toString(16).padStart(2, '0')).join('');
    if (hex !== FLOW_MODEL_SHA256) return false;
    const bytes = new Uint8Array(buffer);
    if (!OFFSETS.every(offset => bytes[offset] === 0xe8 && bytes[offset + 1] === 0x07)) return false;
    for (const offset of OFFSETS) {
        bytes[offset] = 0x80;
        bytes[offset + 1] = 0x04;
    }
    return true;
}

/** Never trim live history to fit the optimization. Grow back to the original. */
export function requiredFlowCapacity(voiceFrames, textFrames, maxFrames) {
    const needed = voiceFrames + textFrames + maxFrames;
    if (![voiceFrames, textFrames, maxFrames].every(n => Number.isSafeInteger(n) && n >= 0)) {
        throw new Error('Invalid flow cache length');
    }
    return needed <= 512 ? 512 : 1000;
}
