import { compactFlowModel, requiredFlowCapacity } from './compact-flow-cache.js';

// Pocket TTS ONNX Web Worker
console.log('Pocket TTS Worker Starting...');
self.postMessage({ type: 'status', status: 'Worker Thread Started', state: 'idle' });

// Load ONNX Runtime (will be loaded dynamically in loadModels for module worker)
let ort = null;
let streamChunkFrames = 4;
let compactFlowEnabled = true;
let flowSessionOptions = null;

// Configuration
let MODELS = {
    mimi_encoder: './onnx/mimi_encoder.onnx',
    text_conditioner: './onnx/text_conditioner.onnx',
    flow_lm_main: './onnx/flow_lm_main_int8.onnx',
    flow_lm_flow: './onnx/flow_lm_flow_int8.onnx',
    mimi_decoder: './onnx/mimi_decoder_int8.onnx',
    tokenizer: './tokenizer.model',
    bundle: './bundle.json',
    bos_before_voice: './bos_before_voice.npy',
    voices: './voices.bin'
};

const SAMPLE_RATE = 24000;
const SAMPLES_PER_FRAME = 1920;
const MAX_FRAMES = 240;
let STATE_CACHE_FRAMES = 1000;
const DEBUG_LOGS = false;
// Text chunking target; lower if long passages hit generation limits.
const CHUNK_TARGET_TOKENS = 50;
// If true, re-run voice conditioning per chunk to avoid stale AR state.
const RESET_FLOW_STATE_EACH_CHUNK = true;
// If true, reset decoder state per chunk to avoid carry-over artifacts.
const RESET_MIMI_STATE_EACH_CHUNK = true;

// Model downloads go through the Cache API, so page loads after the first
// read the roughly 125 MB of Pocket files from disk instead of the network.
// ONNX Runtime fetches URLs itself, so sessions are created from the cached
// bytes rather than from the URL.
const MODEL_CACHE_NAME = 'pocket-tts-models-v1';
let modelCachePromise = null;
let loadProfile = null;

function openModelCache() {
    if (typeof caches === 'undefined') return Promise.resolve(null);
    modelCachePromise ??= caches.open(MODEL_CACHE_NAME).catch((err) => {
        console.warn('Model cache unavailable; downloading every load:', err);
        return null;
    });
    return modelCachePromise;
}

/**
 * Fetches a URL through the model cache and returns its bytes. `key` names
 * the cache entry when the request is not a plain GET of the URL (a byte
 * range of the voice bundle); `accept` decides which responses count.
 */
async function fetchModelBytes(url, options) {
    const startedAt = performance.now();
    try { return await readModelBytes(url, options); }
    finally { if (loadProfile) loadProfile.fileReadMs += performance.now() - startedAt; }
}

async function readModelBytes(url, { key = url, init, accept = (response) => response.ok } = {}) {
    const cache = await openModelCache();
    if (cache) {
        try {
            const hit = await cache.match(key);
            if (hit) {
                if (loadProfile) loadProfile.cachedFiles++;
                return await hit.arrayBuffer();
            }
        } catch (err) {
            console.warn('Model cache read failed:', err);
        }
    }
    if (loadProfile) loadProfile.downloadedFiles++;
    const response = await fetch(url, init);
    if (!accept(response)) {
        throw new Error(`Fetch failed for ${url}: ${response.status} ${response.statusText}`);
    }
    const bytes = await response.arrayBuffer();
    if (cache) {
        // Stored as a complete response under its own key, so a partial
        // (206) download is cacheable too.
        try {
            await cache.put(key, new Response(bytes, {
                headers: {
                    'Content-Type': 'application/octet-stream',
                    'Content-Length': String(bytes.byteLength),
                },
            }));
        } catch (err) {
            console.warn('Model cache write failed:', err);
        }
    }
    return bytes;
}

async function createSessionFromUrl(url, options) {
    const bytes = await fetchModelBytes(url);
    return createProfiledSession(bytes, options);
}

async function createProfiledSession(bytes, options) {
    const startedAt = performance.now();
    try { return await ort.InferenceSession.create(new Uint8Array(bytes), options); }
    finally { if (loadProfile) loadProfile.sessionCreationMs += performance.now() - startedAt; }
}

// State
let mimiEncoderSession = null;
let textConditionerSession = null;
let flowLmMainSession = null;
let flowLmFlowSession = null;
let mimiDecoderSession = null;
let ttsSessionOptions = null;
let ttsUsesWebGpu = false;
let mimiEncoderProvider = null;
let tokenizerProcessor = null;
let tokenizerModelB64 = null;
let bundleMetadata = null;
let bosBeforeVoice = null;
let predefinedVoices = {};
let predefinedVoiceNames = ['alba', 'azelma', 'cosette', 'eponine', 'fantine', 'javert', 'jean', 'marius'];
let predefinedVoiceLoadPromises = {};
let stTensors = []; // Optimization: Pre-allocated s/t tensors for max LSD
let isGenerating = false;
let isReady = false;

// Dynamic LSD (Latent Solver/Diffusion steps)
const MAX_LSD = 10;  // Max quality setting
let currentLSD = 1;  // Default to 1 for faster generation (can be increased via set_lsd)

// Current voice embedding (cached)
let currentVoiceEmbedding = null;
// Conditioned flow-LM state for the active custom voice, kept on the CPU so
// every chunk after the first rebuilds it with a copy instead of a prefill.
// Used when the resident state below is disabled.
let customVoiceFlowState = null;
let currentVoiceName = null;

// The voice-conditioned flow-LM state and the decoder's initial state, kept as
// resident tensors that every utterance starts from. ONNX Runtime never writes
// to its inputs and returns fresh state outputs, so the same tensors can seed
// run after run. Without this each utterance expanded and uploaded about
// 65 MiB of state before its first frame.
let useResidentState = true;
let residentVoiceState = null; // { source, state, tensors }
let residentMimiState = null;  // { state, tensors }
const NO_PROTECTED_TENSORS = new Set();

const PREDEFINED_VOICE_RANGES = {
    alba: { start: 9, end: 6193895 },
    azelma: { start: 6193895, end: 14157255 },
    cosette: { start: 14157255, end: 20351144 },
    eponine: { start: 20351144, end: 27282313 },
    fantine: { start: 27282313, end: 33820266 },
    javert: { start: 33820266, end: 40014154 },
    jean: { start: 40014154, end: 46208040 },
    marius: { start: 46208040, end: 52401928 },
};

// Text preprocessing utilities
const ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const ORDINAL_ONES = ['', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth'];
const ORDINAL_TENS = ['', '', 'twentieth', 'thirtieth', 'fortieth', 'fiftieth', 'sixtieth', 'seventieth', 'eightieth', 'ninetieth'];

function numberToWords(num, options = {}) {
    const { andword = '', zero = 'zero', group = 0 } = options;
    if (num === 0) return zero;
    const convert = (n) => {
        if (n < 20) return ONES[n];
        if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? ' ' + ONES[n % 10] : '');
        if (n < 1000) {
            const remainder = n % 100;
            return ONES[Math.floor(n / 100)] + ' hundred' + (remainder ? (andword ? ' ' + andword + ' ' : ' ') + convert(remainder) : '');
        }
        if (n < 1000000) {
            const thousands = Math.floor(n / 1000);
            const remainder = n % 1000;
            return convert(thousands) + ' thousand' + (remainder ? ' ' + convert(remainder) : '');
        }
        if (n < 1000000000) {
            const millions = Math.floor(n / 1000000);
            const remainder = n % 1000000;
            return convert(millions) + ' million' + (remainder ? ' ' + convert(remainder) : '');
        }
        const billions = Math.floor(n / 1000000000);
        const remainder = n % 1000000000;
        return convert(billions) + ' billion' + (remainder ? ' ' + convert(remainder) : '');
    };
    if (group === 2 && num > 1000 && num < 10000) {
        const high = Math.floor(num / 100);
        const low = num % 100;
        if (low === 0) return convert(high) + ' hundred';
        else if (low < 10) return convert(high) + ' ' + (zero === 'oh' ? 'oh' : zero) + ' ' + ONES[low];
        else return convert(high) + ' ' + convert(low);
    }
    return convert(num);
}

function ordinalToWords(num) {
    if (num < 20) return ORDINAL_ONES[num] || numberToWords(num) + 'th';
    if (num < 100) {
        const tens = Math.floor(num / 10);
        const ones = num % 10;
        if (ones === 0) return ORDINAL_TENS[tens];
        return TENS[tens] + ' ' + ORDINAL_ONES[ones];
    }
    const cardinal = numberToWords(num);
    if (cardinal.endsWith('y')) return cardinal.slice(0, -1) + 'ieth';
    if (cardinal.endsWith('one')) return cardinal.slice(0, -3) + 'first';
    if (cardinal.endsWith('two')) return cardinal.slice(0, -3) + 'second';
    if (cardinal.endsWith('three')) return cardinal.slice(0, -5) + 'third';
    if (cardinal.endsWith('ve')) return cardinal.slice(0, -2) + 'fth';
    if (cardinal.endsWith('e')) return cardinal.slice(0, -1) + 'th';
    if (cardinal.endsWith('t')) return cardinal + 'h';
    return cardinal + 'th';
}

const UNICODE_MAP = {
    'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'æ': 'ae', 'ç': 'c', 'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e', 'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i', 'ñ': 'n', 'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o', 'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u', 'ý': 'y', 'ÿ': 'y', 'ß': 'ss', 'œ': 'oe', 'ð': 'd', 'þ': 'th', 'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A', 'Å': 'A', 'Æ': 'AE', 'Ç': 'C', 'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E', 'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I', 'Ñ': 'N', 'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O', 'Ø': 'O', 'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U', 'Ý': 'Y', '\u201C': '"', '\u201D': '"', '\u2018': "'", '\u2019': "'", '\u2026': '...', '\u2013': '-', '\u2014': '-'
};

function convertToAscii(text) {
    return text.split('').map(c => UNICODE_MAP[c] || c).join('').normalize('NFD').replace(/[\u0300-\u036f]/g, '');
}

const ABBREVIATIONS = [
    [/\bmrs\./gi, 'misuss'], [/\bms\./gi, 'miss'], [/\bmr\./gi, 'mister'], [/\bdr\./gi, 'doctor'], [/\bst\./gi, 'saint'], [/\bco\./gi, 'company'], [/\bjr\./gi, 'junior'], [/\bmaj\./gi, 'major'], [/\bgen\./gi, 'general'], [/\bdrs\./gi, 'doctors'], [/\brev\./gi, 'reverend'], [/\blt\./gi, 'lieutenant'], [/\bhon\./gi, 'honorable'], [/\bsgt\./gi, 'sergeant'], [/\bcapt\./gi, 'captain'], [/\besq\./gi, 'esquire'], [/\bltd\./gi, 'limited'], [/\bcol\./gi, 'colonel'], [/\bft\./gi, 'fort']
];
const CASED_ABBREVIATIONS = [
    [/\bTTS\b/g, 'text to speech'], [/\bHz\b/g, 'hertz'], [/\bkHz\b/g, 'kilohertz'], [/\bKBs\b/g, 'kilobytes'], [/\bKB\b/g, 'kilobyte'], [/\bMBs\b/g, 'megabytes'], [/\bMB\b/g, 'megabyte'], [/\bGBs\b/g, 'gigabytes'], [/\bGB\b/g, 'gigabyte'], [/\bTBs\b/g, 'terabytes'], [/\bTB\b/g, 'terabyte'], [/\bAPIs\b/g, "a p i's"], [/\bAPI\b/g, 'a p i'], [/\bCLIs\b/g, "c l i's"], [/\bCLI\b/g, 'c l i'], [/\bCPUs\b/g, "c p u's"], [/\bCPU\b/g, 'c p u'], [/\bGPUs\b/g, "g p u's"], [/\bGPU\b/g, 'g p u'], [/\bAve\b/g, 'avenue'], [/\betc\b/g, 'etcetera']
];

function expandAbbreviations(text) {
    for (const [regex, replacement] of [...ABBREVIATIONS, ...CASED_ABBREVIATIONS]) text = text.replace(regex, replacement);
    return text;
}

const NUM_PREFIX_RE = /#(\d)/g;
const NUM_SUFFIX_RE = /(\d)([KMBT])/gi;
const NUM_LETTER_SPLIT_RE = /(\d)([a-z])|([a-z])(\d)/gi;
const COMMA_NUMBER_RE = /(\d[\d,]+\d)/g;
const DATE_RE = /(^|[^/])(\d\d?[/-]\d\d?[/-]\d\d(?:\d\d)?)($|[^/])/g;
const PHONE_NUMBER_RE = /\(?\d{3}\)?[-.\s]\d{3}[-.\s]?\d{4}/g;
const TIME_RE = /(\d\d?):(\d\d)(?::(\d\d))?/g;
const POUNDS_RE = /£([\d,]*\d+)/g;
const DOLLARS_RE = /\$([\d.,]*\d+)/g;
const DECIMAL_NUMBER_RE = /(\d+(?:\.\d+)+)/g;
const MULTIPLY_RE = /(\d)\s?\*\s?(\d)/g;
const DIVIDE_RE = /(\d)\s?\/\s?(\d)/g;
const ADD_RE = /(\d)\s?\+\s?(\d)/g;
const SUBTRACT_RE = /(\d)?\s?-\s?(\d)/g;
const FRACTION_RE = /(\d+)\/(\d+)/g;
const ORDINAL_RE = /(\d+)(st|nd|rd|th)/gi;
const NUMBER_RE = /\d+/g;

function normalizeNumbers(text) {
    text = text.replace(NUM_PREFIX_RE, (_, d) => `number ${d}`);
    text = text.replace(NUM_SUFFIX_RE, (_, num, suffix) => {
        const map = { k: 'thousand', m: 'million', b: 'billion', t: 'trillion' };
        return `${num} ${map[suffix.toLowerCase()]}`;
    });
    for (let i = 0; i < 2; i++) {
        text = text.replace(NUM_LETTER_SPLIT_RE, (m, d1, l1, l2, d2) => {
            if (d1 && l1) return `${d1} ${l1}`;
            if (l2 && d2) return `${l2} ${d2}`;
            return m;
        });
    }
    text = text.replace(COMMA_NUMBER_RE, m => m.replace(/,/g, ''));
    text = text.replace(DATE_RE, (_, pre, date, post) => pre + date.split(/[./-]/).join(' dash ') + post);
    text = text.replace(PHONE_NUMBER_RE, m => {
        const digits = m.replace(/\D/g, '');
        return digits.length === 10 ? `${digits.slice(0, 3).split('').join(' ')}, ${digits.slice(3, 6).split('').join(' ')}, ${digits.slice(6).split('').join(' ')}` : m;
    });
    text = text.replace(TIME_RE, (_, hours, minutes, seconds) => {
        const h = parseInt(hours), m = parseInt(minutes), s = seconds ? parseInt(seconds) : 0;
        if (!seconds) return m === 0 ? (h === 0 ? '0' : h > 12 ? `${hours} minutes` : `${hours} o'clock`) : minutes.startsWith('0') ? `${hours} oh ${minutes[1]}` : `${hours} ${minutes}`;
        let res = '';
        if (h !== 0) res = hours + ' ' + (m === 0 ? 'oh oh' : minutes.startsWith('0') ? `oh ${minutes[1]}` : minutes);
        else if (m !== 0) res = minutes + ' ' + (s === 0 ? 'oh oh' : seconds.startsWith('0') ? `oh ${seconds[1]}` : seconds);
        else res = seconds;
        return res + ' ' + (s === 0 ? '' : seconds.startsWith('0') ? `oh ${seconds[1]}` : seconds);
    });
    text = text.replace(POUNDS_RE, (_, amount) => `${amount.replace(/,/g, '')} pounds`);
    text = text.replace(DOLLARS_RE, (_, amount) => {
        const parts = amount.replace(/,/g, '').split('.');
        const dollars = parseInt(parts[0]) || 0;
        const cents = parts[1] ? parseInt(parts[1]) : 0;
        if (dollars && cents) return `${dollars} ${dollars === 1 ? 'dollar' : 'dollars'}, ${cents} ${cents === 1 ? 'cent' : 'cents'}`;
        if (dollars) return `${dollars} ${dollars === 1 ? 'dollar' : 'dollars'}`;
        if (cents) return `${cents} ${cents === 1 ? 'cent' : 'cents'}`;
        return 'zero dollars';
    });
    text = text.replace(DECIMAL_NUMBER_RE, m => m.split('.').join(' point ').split('').join(' '));
    text = text.replace(MULTIPLY_RE, '$1 times $2');
    text = text.replace(DIVIDE_RE, '$1 over $2');
    text = text.replace(ADD_RE, '$1 plus $2');
    text = text.replace(SUBTRACT_RE, (_, a, b) => (a ? a : '') + ' minus ' + b);
    text = text.replace(FRACTION_RE, '$1 over $2');
    text = text.replace(ORDINAL_RE, (_, num) => ordinalToWords(parseInt(num)));
    text = text.replace(NUMBER_RE, m => {
        const num = parseInt(m);
        if (num > 1000 && num < 3000) {
            if (num === 2000) return 'two thousand';
            if (num > 2000 && num < 2010) return 'two thousand ' + numberToWords(num % 100);
            if (num % 100 === 0) return numberToWords(Math.floor(num / 100)) + ' hundred';
            return numberToWords(num, { zero: 'oh', group: 2 });
        }
        return numberToWords(num);
    });
    return text;
}

const SPECIAL_CHARACTERS = [
    [/@/g, ' at '], [/&/g, ' and '], [/%/g, ' percent '], [/:/g, '.'], [/;/g, ','], [/\+/g, ' plus '], [/\\/g, ' backslash '], [/~/g, ' about '], [/(^| )<3/g, ' heart '], [/<=/g, ' less than or equal to '], [/>=/g, ' greater than or equal to '], [/</g, ' less than '], [/>/g, ' greater than '], [/=/g, ' equals '], [/\//g, ' slash '], [/_/g, ' '],
];
const LINK_HEADER_RE = /https?:\/\//gi;
const DASH_RE = /(.) - (.)/g;
const DOT_RE = /([A-Z])\.([A-Z])/gi;
const PARENTHESES_RE = /[\(\[\{][^\)\]\}]*[\)\]\}](.)?/g;

function normalizeSpecial(text) {
    text = text.replace(LINK_HEADER_RE, 'h t t p s colon slash slash ');
    text = text.replace(DASH_RE, '$1, $2');
    text = text.replace(DOT_RE, '$1 dot $2');
    text = text.replace(PARENTHESES_RE, (m, after) => {
        let result = m.replace(/[\(\[\{]/g, ', ').replace(/[\)\]\}]/g, ', ');
        if (after && /[$.!?,]/.test(after)) result = result.slice(0, -2) + after;
        return result;
    });
    return text;
}

function expandSpecialCharacters(text) {
    for (const [regex, replacement] of SPECIAL_CHARACTERS) text = text.replace(regex, replacement);
    return text;
}

function collapseWhitespace(text) {
    return text.replace(/\s+/g, ' ').replace(/ ([.\?!,])/g, '$1');
}

function dedupPunctuation(text) {
    return text.replace(/\.\.\.+/g, '[ELLIPSIS]').replace(/,+/g, ',').replace(/[.,]*\.[.,]*/g, '.').replace(/[.,!]*![.,!]*/g, '!').replace(/[.,!?]*\?[.,!?]*/g, '?').replace(/\[ELLIPSIS\]/g, '...');
}

const SENTENCE_SPLIT_RE = /[^.!?]+[.!?]+|[^.!?]+$/g;

function splitTextIntoSentences(text) {
    const matches = text.match(SENTENCE_SPLIT_RE);
    if (!matches) return [];
    return matches.map(sentence => sentence.trim()).filter(Boolean);
}

function splitTokenIdsIntoChunks(tokenIds, maxTokens) {
    const chunks = [];
    for (let i = 0; i < tokenIds.length; i += maxTokens) {
        const chunkText = tokenizerProcessor.decodeIds(tokenIds.slice(i, i + maxTokens)).trim();
        if (chunkText) chunks.push(chunkText);
    }
    return chunks;
}

// Split text into sentence chunks (target <= CHUNK_TARGET_TOKENS tokens)
function splitIntoBestSentences(text) {
    const preparedText = prepareText(text);
    if (!preparedText) return [];

    const sentences = splitTextIntoSentences(preparedText);
    if (sentences.length === 0) return [];

    // Merge sentences into chunks that stay within the token target
    const chunks = [];
    let currentChunk = '';
    for (const sentenceText of sentences) {
        const sentenceTokenIds = tokenizerProcessor.encodeIds(sentenceText);
        const sentenceTokens = sentenceTokenIds.length;

        if (sentenceTokens > CHUNK_TARGET_TOKENS) {
            if (currentChunk !== '') {
                chunks.push(currentChunk.trim());
                currentChunk = '';
            }
            const splitChunks = splitTokenIdsIntoChunks(sentenceTokenIds, CHUNK_TARGET_TOKENS);
            for (const splitChunk of splitChunks) {
                if (splitChunk) chunks.push(splitChunk.trim());
            }
            continue;
        }

        if (currentChunk === '') {
            currentChunk = sentenceText;
            continue;
        }

        const combined = `${currentChunk} ${sentenceText}`;
        const combinedTokens = tokenizerProcessor.encodeIds(combined).length;
        if (combinedTokens > CHUNK_TARGET_TOKENS) {
            chunks.push(currentChunk.trim());
            currentChunk = sentenceText;
        } else {
            currentChunk = combined;
        }
    }

    if (currentChunk !== '') {
        chunks.push(currentChunk.trim());
    }

    return chunks;
}

// Pocket TTS specific text preprocessing
function prepareText(text) {
    text = text.trim();
    if (!text) return '';

    // Convert to ASCII
    text = convertToAscii(text);

    // Normalize numbers first
    text = normalizeNumbers(text);

    // Normalize special characters
    text = normalizeSpecial(text);

    // Expand abbreviations
    text = expandAbbreviations(text);

    // Expand special characters
    text = expandSpecialCharacters(text);

    // Collapse whitespace
    text = collapseWhitespace(text);

    // Deduplicate punctuation
    text = dedupPunctuation(text);

    // Final cleanup
    text = text.trim();

    // Ensure proper punctuation at end
    if (text && text[text.length - 1].match(/[a-zA-Z0-9]/)) {
        text = text + '.';
    }

    // Capitalize first letter
    if (text && !text[0].match(/[A-Z]/)) {
        text = text[0].toUpperCase() + text.slice(1);
    }

    return text;
}

// ----------------------------------------------------------------------------
// Worker Logic
// ----------------------------------------------------------------------------

self.onmessage = async (e) => {
    const { type, data } = e.data;
    console.log('Worker received message:', type);

    if (type === 'load') {
        try {
            if (data && data.urls) {
                MODELS = { ...MODELS, ...data.urls };
            }
            if (data && typeof data.residentState === 'boolean') {
                useResidentState = data.residentState;
            }
            if ([2, 4].includes(data?.streamChunkFrames)) streamChunkFrames = data.streamChunkFrames;
            compactFlowEnabled = data?.flowCacheFrames !== 1000;
            await loadModels();
            postMessage({ type: 'loaded', flowCacheFrames: STATE_CACHE_FRAMES });
        } catch (err) {
            postMessage({ type: 'error', error: err.toString() });
        }
    } else if (type === 'generate') {
        if (!isReady) {
            postMessage({ type: 'error', error: 'Models are not loaded yet.' });
            return;
        }
        if (isGenerating) return;
        try {
            await startGeneration(data.text, data.voiceName, data.requestId);
        } catch (err) {
            console.error('Generation Error:', err);
            postMessage({ type: 'error', error: err.toString() });
        }
    } else if (type === 'encode_voice') {
        if (!isReady) {
            postMessage({ type: 'error', error: 'Models are not loaded yet.' });
            return;
        }
        try {
            const embedding = await encodeVoiceAudio(data.audio);
            customVoiceFlowState = null;
            releaseResidentVoiceState();
            currentVoiceEmbedding = embedding;
            currentVoiceName = 'custom';
            postMessage({ type: 'voice_encoded', voiceName: 'custom' });
        } catch (err) {
            console.error('Voice encoding error:', err);
            postMessage({ type: 'error', error: 'Failed to encode voice: ' + err.toString() });
        }
    } else if (type === 'set_voice') {
        if (!isReady) {
            postMessage({ type: 'error', error: 'Models are not loaded yet.' });
            return;
        }
        if (data.voiceName === 'custom') {
            // Custom voice already set via encode_voice
            postMessage({ type: 'voice_set', voiceName: 'custom' });
        } else if (predefinedVoiceNames.includes(data.voiceName)) {
            try {
                currentVoiceEmbedding = await loadPredefinedVoice(data.voiceName);
                currentVoiceName = data.voiceName;
                postMessage({ type: 'voice_set', voiceName: data.voiceName });
            } catch (err) {
                postMessage({ type: 'error', error: `Failed to load voice ${data.voiceName}: ${err}` });
            }
        } else {
            postMessage({ type: 'error', error: `Unknown voice: ${data.voiceName}` });
        }
    } else if (type === 'set_lsd') {
        // Dynamic LSD adjustment for edge devices
        const lsdValue = data?.lsd ?? e.data?.lsd;
        const newLSD = Math.max(1, Math.min(MAX_LSD, lsdValue ?? currentLSD));
        if (newLSD !== currentLSD) {
            console.log(`LSD adjusted: ${currentLSD} → ${newLSD}`);
            currentLSD = newLSD;
        }
    } else if (type === 'stop') {
        isGenerating = false;
        postMessage({ type: 'status', status: 'Stopped', state: 'idle' });
    }
};

async function loadModels() {
    if (textConditionerSession && flowLmMainSession && flowLmFlowSession && mimiDecoderSession) return;
    const loadStartedAt = performance.now();
    loadProfile = { fileReadMs: 0, sessionCreationMs: 0, cachedFiles: 0, downloadedFiles: 0 };

    postMessage({ type: 'status', status: 'Loading ONNX Runtime...', state: 'loading' });

    // Load ONNX Runtime dynamically
    const version = '1.20.0';
    const cdnBase = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/`;

    try {
        const ortModule = await import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/ort.min.mjs`);
        ort = ortModule.default || ortModule;
    } catch (e) {
        console.error('Failed to load ONNX Runtime:', e);
        throw new Error('Failed to load ONNX Runtime: ' + e.message);
    }

    if (!ort) {
        throw new Error('ONNX Runtime failed to load');
    }

    postMessage({ type: 'status', status: 'Loading models...', state: 'loading' });

    // Configure WASM Paths
    ort.env.wasm.wasmPaths = cdnBase;

    // Enable SIMD for significant performance boost (2-4x faster)
    ort.env.wasm.simd = true;

    // Configure multi-threading
    if (!self.crossOriginIsolated) {
        console.warn('Environment is not cross-origin isolated. Disabling WASM multi-threading.');
        console.warn('To enable multi-threading, serve with headers:');
        console.warn('  Cross-Origin-Opener-Policy: same-origin');
        console.warn('  Cross-Origin-Embedder-Policy: require-corp');
        ort.env.wasm.numThreads = 1;
    } else {
        const threads = Math.min(navigator.hardwareConcurrency || 4, 4);
        ort.env.wasm.numThreads = threads;
        if (DEBUG_LOGS) {
            console.log(`Multi-threading enabled with ${threads} threads`);
        }
    }

    // Check for WebGPU support
    let hasWebGPU = false;
    try {
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            const adapter = await navigator.gpu.requestAdapter();
            hasWebGPU = !!adapter;
        }
    } catch (e) {
        console.log('WebGPU not available:', e.message);
    }

    console.log(`ORT: crossOriginIsolated=${self.crossOriginIsolated}, simd=${ort.env.wasm.simd}, threads=${ort.env.wasm.numThreads}, webgpu=${hasWebGPU}`);

    try {
        // Use WebGPU if available (2-5x faster), fall back to WASM
        const executionProviders = hasWebGPU ? ['webgpu', 'wasm'] : ['wasm'];
        ttsUsesWebGpu = hasWebGPU;
        console.log(`Using execution providers: ${executionProviders.join(', ')}`);

        const sessionOptions = {
            executionProviders,
            graphOptimizationLevel: 'all'
        };
        ttsSessionOptions = sessionOptions;

        // Recurrent caches are consumed by the next inference call and never
        // inspected by JavaScript. Keeping them on the GPU avoids downloading
        // and uploading tens of megabytes of state for every generated frame.
        const flowMainSessionOptions = hasWebGPU ? {
            ...sessionOptions,
            preferredOutputLocation: {
                conditioning: 'gpu-buffer',
                ...Object.fromEntries(
                    Object.keys(FLOW_LM_STATE_SHAPES).map(name => [`out_${name}`, 'gpu-buffer'])
                )
            }
        } : sessionOptions;
        const mimiDecoderSessionOptions = hasWebGPU ? {
            ...sessionOptions,
            preferredOutputLocation: Object.fromEntries(
                Object.keys(MIMI_DECODER_STATE_SHAPES).map(name => [`out_${name}`, 'gpu-buffer'])
            )
        } : sessionOptions;

        postMessage({ type: 'status', status: 'Loading text conditioner...', state: 'loading' });
        textConditionerSession = await createSessionFromUrl(MODELS.text_conditioner, sessionOptions);

        postMessage({ type: 'status', status: 'Loading language generator...', state: 'loading' });
        flowSessionOptions = flowMainSessionOptions;
        const flowBytes = await fetchModelBytes(MODELS.flow_lm_main);
        let compacted = false;
        if (compactFlowEnabled) {
            try { compacted = await compactFlowModel(flowBytes); }
            catch (err) { console.warn('Cache compaction unavailable; retaining original model:', err); }
        }
        setFlowCacheFrames(compacted ? 512 : 1000);
        flowLmMainSession = await createProfiledSession(flowBytes, flowSessionOptions);

        postMessage({ type: 'status', status: 'Loading flow decoder...', state: 'loading' });
        flowLmFlowSession = await createSessionFromUrl(MODELS.flow_lm_flow, sessionOptions);

        postMessage({ type: 'status', status: 'Loading audio decoder...', state: 'loading' });
        mimiDecoderSession = await createSessionFromUrl(MODELS.mimi_decoder, mimiDecoderSessionOptions);

        if (DEBUG_LOGS) {
            console.log('Core TTS models loaded successfully');
            console.log('Flow LM Main inputs:', flowLmMainSession.inputNames);
            console.log('Flow LM Main outputs:', flowLmMainSession.outputNames);
            console.log('MIMI decoder inputs:', mimiDecoderSession.inputNames);
            console.log('MIMI decoder outputs:', mimiDecoderSession.outputNames);
        }

        // Load tokenizer
        postMessage({ type: 'status', status: 'Loading tokenizer...', state: 'loading' });
        if (DEBUG_LOGS) {
            console.log('Loading tokenizer...');
        }

        const tokenizerBuffer = await fetchModelBytes(MODELS.tokenizer);
        tokenizerModelB64 = btoa(String.fromCharCode(...new Uint8Array(tokenizerBuffer)));

        // Import and initialize sentencepiece processor
        const spModule = await import('./sentencepiece.js?v=2');
        const SentencePieceProcessor = spModule.SentencePieceProcessor;
        if (!SentencePieceProcessor) {
            throw new Error('SentencePieceProcessor not found in sentencepiece.js');
        }
        tokenizerProcessor = new SentencePieceProcessor();
        await tokenizerProcessor.loadFromB64StringModel(tokenizerModelB64);
        if (DEBUG_LOGS) {
            console.log('Tokenizer loaded');
        }

        // Load only the tiny manifest for preset voice names. The heavy voice
        // state bundle is fetched lazily for the selected voice during speech.
        postMessage({ type: 'status', status: 'Loading voice manifest...', state: 'loading' });
        predefinedVoiceNames = await loadPredefinedVoiceNames();

        // Send list of available voices
        postMessage({
            type: 'voices_loaded',
            voices: predefinedVoiceNames,
            defaultVoice: currentVoiceName
        });

        // Pre-allocate s/t tensors for Flow Matching Loop (Optimization)
        // Pre-allocate for MAX_LSD to support dynamic switching
        if (DEBUG_LOGS) {
            console.log(`Pre-allocating Flow Matching tensors for LSD 1-${MAX_LSD}...`);
        }
        stTensors = {};

        for (let lsd = 1; lsd <= MAX_LSD; lsd++) {
            stTensors[lsd] = [];
            const dt = 1.0 / lsd;
            for (let j = 0; j < lsd; j++) {
                const s = j / lsd;
                const t = s + dt;
                stTensors[lsd].push({
                    s: new ort.Tensor('float32', new Float32Array([s]), [1, 1]),
                    t: new ort.Tensor('float32', new Float32Array([t]), [1, 1])
                });
            }
        }

        isReady = true;
        postMessage({ type: 'status', status: 'Ready', state: 'idle' });
        postMessage({ type: 'model_status', status: 'ready', text: 'Ready' });
        postMessage({ type: 'load_profile', profile: { ...loadProfile, totalMs: performance.now() - loadStartedAt } });
        loadProfile = null;

    } catch (err) {
        loadProfile = null;
        console.error('Model load failed:', err);
        throw err;
    }
}

async function ensureMimiEncoderSession(forceWasm = false) {
    if (mimiEncoderSession && (!forceWasm || mimiEncoderProvider === 'wasm')) {
        return mimiEncoderSession;
    }
    if (mimiEncoderSession) {
        await releaseMimiEncoderSession();
    }
    if (!ort || !ttsSessionOptions) {
        await loadModels();
    }
    if (!ort || !ttsSessionOptions) {
        throw new Error('ONNX Runtime is not initialized');
    }

    postMessage({ type: 'status', status: 'Loading voice encoder...', state: 'loading' });
    const sessionOptions = forceWasm ? {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    } : ttsSessionOptions;
    mimiEncoderSession = await createSessionFromUrl(MODELS.mimi_encoder, sessionOptions);
    mimiEncoderProvider = forceWasm || !ttsUsesWebGpu ? 'wasm' : 'webgpu';
    postMessage({ type: 'status', status: 'Ready', state: 'idle' });
    return mimiEncoderSession;
}

async function releaseMimiEncoderSession() {
    const session = mimiEncoderSession;
    mimiEncoderSession = null;
    mimiEncoderProvider = null;
    if (!session || typeof session.release !== 'function') return;

    try {
        await session.release();
    } catch (err) {
        console.warn('Could not release voice encoder session:', err);
    }
}

async function loadPredefinedVoiceNames() {
    try {
        bundleMetadata = JSON.parse(new TextDecoder().decode(await fetchModelBytes(MODELS.bundle)));
        await loadBosBeforeVoice();
        if (Array.isArray(bundleMetadata.predefined_voices) && bundleMetadata.predefined_voices.length > 0) {
            return bundleMetadata.predefined_voices.filter((voice) => typeof voice === 'string');
        }
    } catch (err) {
        console.warn('Could not load voice manifest, using bundled defaults:', err);
    }
    return Object.keys(PREDEFINED_VOICE_RANGES);
}

async function loadBosBeforeVoice() {
    bosBeforeVoice = null;
    if (!bundleMetadata?.insert_bos_before_voice) return;

    try {
        bosBeforeVoice = parseNpyFloat32(await fetchModelBytes(MODELS.bos_before_voice));
    } catch (err) {
        console.warn('Could not load BOS voice token for custom voices:', err);
    }
}

async function loadPredefinedVoice(voiceName) {
    if (predefinedVoices[voiceName]) {
        return predefinedVoices[voiceName];
    }
    if (predefinedVoiceLoadPromises[voiceName]) {
        return predefinedVoiceLoadPromises[voiceName];
    }

    predefinedVoiceLoadPromises[voiceName] = (async () => {
        postMessage({ type: 'status', status: `Loading ${voiceName} voice...`, state: 'loading' });

        try {
            const range = PREDEFINED_VOICE_RANGES[voiceName];
            if (range) {
                try {
                    // One voice's slice of the bundle, cached under its own key.
                    const slice = await fetchModelBytes(MODELS.voices, {
                        key: `${MODELS.voices}?voice=${encodeURIComponent(voiceName)}`,
                        init: { headers: { Range: `bytes=${range.start}-${range.end - 1}` } },
                        accept: (response) => response.status === 206,
                    });
                    const voice = parsePtvbVoiceSlice(slice);
                    cacheSinglePredefinedVoice(voiceName, voice);
                    return voice;
                } catch (err) {
                    console.warn(`Range fetch for ${voiceName} failed, falling back to full voice bundle:`, err);
                }
            }

            predefinedVoices = parseVoicesBin(await fetchModelBytes(MODELS.voices));
            if (!predefinedVoices[voiceName]) {
                throw new Error(`Voice bundle did not contain ${voiceName}`);
            }
            return predefinedVoices[voiceName];
        } finally {
            delete predefinedVoiceLoadPromises[voiceName];
        }
    })();

    return predefinedVoiceLoadPromises[voiceName];
}

function cacheSinglePredefinedVoice(voiceName, voice) {
    // Keep one preset voice state bundle resident at a time. Custom voice state
    // stays in currentVoiceEmbedding and is not stored in predefinedVoices.
    predefinedVoices = { [voiceName]: voice };
}

function parseVoicesBin(buffer) {
    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(5, buffer.byteLength)));
    if (magic === 'PTVB1') {
        return parsePtvbVoices(buffer);
    }

    // Simple binary format:
    // Header: 4 bytes (uint32) = number of voices
    // For each voice:
    //   - 32 bytes: voice name (null-terminated string)
    //   - 4 bytes (uint32): number of frames
    //   - 4 bytes (uint32): embedding dim (1024)
    //   - frames * dim * 4 bytes: float32 embeddings

    const voices = {};
    const view = new DataView(buffer);
    let offset = 0;

    const numVoices = view.getUint32(offset, true);
    offset += 4;

    for (let i = 0; i < numVoices; i++) {
        // Read voice name
        const nameBytes = new Uint8Array(buffer, offset, 32);
        const nameEnd = nameBytes.indexOf(0);
        const name = new TextDecoder().decode(nameBytes.subarray(0, nameEnd > 0 ? nameEnd : 32)).trim();
        offset += 32;

        // Read dimensions
        const numFrames = view.getUint32(offset, true);
        offset += 4;
        const embDim = view.getUint32(offset, true);
        offset += 4;

        // Read embeddings
        const embSize = numFrames * embDim;
        const embeddings = new Float32Array(buffer, offset, embSize);
        offset += embSize * 4;

        // Store as [1, numFrames, embDim] shaped array info
        voices[name] = {
            data: new Float32Array(embeddings),
            shape: [1, numFrames, embDim]
        };

        console.log(`Loaded voice '${name}': ${numFrames} frames, ${embDim} dim`);
    }

    return voices;
}

function parsePtvbVoiceSlice(buffer) {
    const voices = {};
    const view = new DataView(buffer);
    const decoder = new TextDecoder();
    let offset = 0;

    const readUint16 = () => {
        const value = view.getUint16(offset, true);
        offset += 2;
        return value;
    };
    const readUint32 = () => {
        const value = view.getUint32(offset, true);
        offset += 4;
        return value;
    };
    const readString = (length) => {
        const value = decoder.decode(new Uint8Array(buffer, offset, length));
        offset += length;
        return value;
    };

    const name = readString(readUint16());
    const tensorCount = readUint16();
    const tensors = {};

    for (let tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
        const path = readString(readUint16());
        const dtypeCode = view.getUint8(offset++);
        const rank = view.getUint8(offset++);
        const shape = [];
        for (let i = 0; i < rank; i++) {
            shape.push(readUint32());
        }
        const byteLength = readUint32();
        tensors[path] = {
            dtype: dtypeCode === 0 ? 'float32' : dtypeCode === 1 ? 'int64' : `unknown:${dtypeCode}`,
            shape,
            dataOffset: offset,
            byteLength,
        };
        offset += byteLength;
    }

    voices[name] = {
        type: 'flow_state_bundle',
        buffer,
        tensors,
    };

    console.log(`Loaded preset voice '${name}' with ${tensorCount} tensors`);
    return voices[name];
}

function parseNpyFloat32(buffer) {
    const view = new DataView(buffer);
    const magic = new Uint8Array(buffer, 0, 6);
    const expectedMagic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
    for (let i = 0; i < expectedMagic.length; i++) {
        if (magic[i] !== expectedMagic[i]) {
            throw new Error('Invalid NPY header');
        }
    }

    const major = view.getUint8(6);
    const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
    const headerOffset = major === 1 ? 10 : 12;
    const headerText = new TextDecoder().decode(new Uint8Array(buffer, headerOffset, headerLength));
    const shapeMatch = headerText.match(/\(\s*([0-9,\s]+)\)/);
    if (!shapeMatch) {
        throw new Error('Could not parse NPY shape');
    }

    const shape = shapeMatch[1]
        .split(',')
        .map((part) => part.trim())
        .filter(Boolean)
        .map(Number);
    const dataOffset = headerOffset + headerLength;
    const byteLength = buffer.byteLength - dataOffset;
    const data = new Float32Array(byteLength / 4);
    new Uint8Array(data.buffer).set(new Uint8Array(buffer, dataOffset, byteLength));
    return { data, shape };
}

function parsePtvbVoices(buffer) {
    const voices = {};
    const view = new DataView(buffer);
    const decoder = new TextDecoder();
    let offset = 5; // "PTVB1"

    const readUint16 = () => {
        const value = view.getUint16(offset, true);
        offset += 2;
        return value;
    };
    const readUint32 = () => {
        const value = view.getUint32(offset, true);
        offset += 4;
        return value;
    };
    const readString = (length) => {
        const value = decoder.decode(new Uint8Array(buffer, offset, length));
        offset += length;
        return value;
    };

    const voiceCount = readUint32();
    for (let voiceIdx = 0; voiceIdx < voiceCount; voiceIdx++) {
        const name = readString(readUint16());
        const tensorCount = readUint16();
        const tensors = {};

        for (let tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
            const path = readString(readUint16());
            const dtypeCode = view.getUint8(offset++);
            const rank = view.getUint8(offset++);
            const shape = [];
            for (let i = 0; i < rank; i++) {
                shape.push(readUint32());
            }
            const byteLength = readUint32();
            tensors[path] = {
                dtype: dtypeCode === 0 ? 'float32' : dtypeCode === 1 ? 'int64' : `unknown:${dtypeCode}`,
                shape,
                dataOffset: offset,
                byteLength,
            };
            offset += byteLength;
        }

        voices[name] = {
            type: 'flow_state_bundle',
            buffer,
            tensors,
        };

        console.log(`Loaded precomputed voice '${name}' with ${tensorCount} tensors`);
    }

    return voices;
}

function copyTensorDataFromBundle(voiceBundle, record) {
    const bytes = new Uint8Array(voiceBundle.buffer, record.dataOffset, record.byteLength);
    if (record.dtype === 'float32') {
        const data = new Float32Array(record.byteLength / 4);
        new Uint8Array(data.buffer).set(bytes);
        return data;
    }
    if (record.dtype === 'int64') {
        const data = new BigInt64Array(record.byteLength / 8);
        new Uint8Array(data.buffer).set(bytes);
        return data;
    }
    throw new Error(`Unsupported PTVB tensor dtype: ${record.dtype}`);
}

function copyPaddedFlowCacheFromBundle(voiceBundle, record, targetShape) {
    const source = copyTensorDataFromBundle(voiceBundle, record);
    const target = new Float32Array(targetShape.reduce((a, b) => a * b, 1));

    if (record.shape.length !== targetShape.length || record.shape.length !== 5) {
        throw new Error(`Unsupported flow cache shape: ${JSON.stringify(record.shape)}`);
    }

    const sourceSeq = record.shape[2];
    const targetSeq = targetShape[2];
    if (sourceSeq > targetSeq) {
        throw new Error(`Voice cache has ${sourceSeq} frames, but model only accepts ${targetSeq}`);
    }

    const outer = record.shape[0] * record.shape[1];
    const sourceTail = record.shape[3] * record.shape[4];
    const targetTail = targetShape[3] * targetShape[4];
    if (sourceTail !== targetTail) {
        throw new Error(`Voice cache tail shape mismatch: ${JSON.stringify(record.shape)} vs ${JSON.stringify(targetShape)}`);
    }

    for (let outerIdx = 0; outerIdx < outer; outerIdx++) {
        const sourceOffset = outerIdx * sourceSeq * sourceTail;
        const targetOffset = outerIdx * targetSeq * targetTail;
        target.set(
            source.subarray(sourceOffset, sourceOffset + sourceSeq * sourceTail),
            targetOffset,
        );
    }

    return target;
}

function createFlowStateFromPrecomputedVoice(voiceBundle) {
    const flowLmState = initState(flowLmMainSession, FLOW_LM_STATE_SHAPES);

    for (let layer = 0; layer < FLOW_LM_CACHE_LAYERS; layer++) {
        const prefix = `transformer.layers.${layer}.self_attn`;
        const cacheRecord = voiceBundle.tensors[`${prefix}/cache`];
        const offsetRecord = voiceBundle.tensors[`${prefix}/offset`];
        if (!cacheRecord || !offsetRecord) {
            throw new Error(`Voice bundle is missing flow state for layer ${layer}`);
        }

        const cacheStateName = `state_${layer * 3}`;
        const cacheShape = FLOW_LM_STATE_SHAPES[cacheStateName].shape;
        flowLmState[`state_${layer * 3}`] = new ort.Tensor(
            'float32',
            copyPaddedFlowCacheFromBundle(voiceBundle, cacheRecord, cacheShape),
            cacheShape,
        );
        flowLmState[`state_${layer * 3 + 2}`] = new ort.Tensor(
            'int64',
            copyTensorDataFromBundle(voiceBundle, offsetRecord),
            offsetRecord.shape,
        );
    }

    return flowLmState;
}

function cloneTypedArray(data) {
    return new data.constructor(data);
}

function compactFlowCache(data, dims, keepFrames) {
    const outer = dims[0] * dims[1];
    const seq = dims[2];
    const tail = dims[3] * dims[4];
    const compact = new Float32Array(outer * keepFrames * tail);

    for (let outerIdx = 0; outerIdx < outer; outerIdx++) {
        const sourceOffset = outerIdx * seq * tail;
        compact.set(
            data.subarray(sourceOffset, sourceOffset + keepFrames * tail),
            outerIdx * keepFrames * tail,
        );
    }

    return compact;
}

function expandFlowCache(compact, dims, keepFrames) {
    const outer = dims[0] * dims[1];
    const seq = dims[2];
    const tail = dims[3] * dims[4];
    const data = new Float32Array(outer * seq * tail);

    for (let outerIdx = 0; outerIdx < outer; outerIdx++) {
        const sourceOffset = outerIdx * keepFrames * tail;
        data.set(
            compact.subarray(sourceOffset, sourceOffset + keepFrames * tail),
            outerIdx * seq * tail,
        );
    }

    return data;
}

async function readTensorData(tensor) {
    if (tensor.location === 'cpu' || tensor.location === 'cpu-pinned') {
        return tensor.data;
    }
    // WebGPU keeps the recurrent caches in GPU buffers; download without
    // releasing so the caller can keep generating from the same state.
    return tensor.getData();
}

/**
 * Copies a voice-conditioned flow-LM state to the CPU. Only the frames the
 * cache actually holds are kept, which is the same compact layout preset
 * voices ship inside voices.bin.
 */
async function snapshotFlowLmState(state) {
    const cacheFrames = {};
    for (let layer = 0; layer < FLOW_LM_CACHE_LAYERS; layer++) {
        const offsetTensor = state[`state_${layer * 3 + 2}`];
        if (!offsetTensor) continue;
        const offsetData = await readTensorData(offsetTensor);
        cacheFrames[`state_${layer * 3}`] = Number(offsetData[0]);
    }

    const snapshot = {};
    for (const [name, tensor] of Object.entries(state)) {
        const data = await readTensorData(tensor);
        const dims = [...tensor.dims];
        const frames = cacheFrames[name];

        if (frames === undefined || dims.length !== 5) {
            snapshot[name] = { type: tensor.type, dims, data: cloneTypedArray(data) };
            continue;
        }

        const keepFrames = Math.max(0, Math.min(frames, dims[2]));
        snapshot[name] = {
            type: tensor.type,
            dims,
            keepFrames,
            data: compactFlowCache(data, dims, keepFrames),
        };
    }

    return snapshot;
}

function createFlowStateFromSnapshot(snapshot) {
    const state = {};

    for (const [name, record] of Object.entries(snapshot)) {
        const data = record.keepFrames === undefined
            ? cloneTypedArray(record.data)
            : expandFlowCache(record.data, record.dims, record.keepFrames);
        state[name] = new ort.Tensor(record.type, data, record.dims);
    }

    return state;
}

function createVoiceConditioningTensor(voiceEmb) {
    let data = voiceEmb.data;
    let shape = voiceEmb.shape.slice();

    if (bundleMetadata?.insert_bos_before_voice && bosBeforeVoice) {
        if (shape.length !== 3 || bosBeforeVoice.shape.length !== 3) {
            throw new Error(`Invalid voice conditioning shape: ${JSON.stringify(shape)}`);
        }
        if (shape[0] !== 1 || bosBeforeVoice.shape[0] !== 1 || shape[2] !== bosBeforeVoice.shape[2]) {
            throw new Error(`Voice BOS shape mismatch: ${JSON.stringify(shape)} vs ${JSON.stringify(bosBeforeVoice.shape)}`);
        }

        const combined = new Float32Array(bosBeforeVoice.data.length + data.length);
        combined.set(bosBeforeVoice.data, 0);
        combined.set(data, bosBeforeVoice.data.length);
        data = combined;
        shape = [1, shape[1] + bosBeforeVoice.shape[1], shape[2]];
    }

    return new ort.Tensor('float32', data, shape);
}

function describeRuntimeError(error) {
    if (error instanceof Error && error.message) return error.message;
    if (typeof error === 'number') return `ONNX Runtime error code ${error}`;
    return String(error);
}

async function runVoiceEncoder(encoderSession, audioData) {
    // audioData should be Float32Array at 24kHz, mono
    // Reshape to [1, 1, samples]
    const input = new ort.Tensor('float32', audioData, [1, 1, audioData.length]);
    let outputs = null;

    try {
        outputs = await encoderSession.run({ audio: input });
        const embeddings = outputs[encoderSession.outputNames[0]];
        return {
            data: new Float32Array(embeddings.data),
            shape: [...embeddings.dims]
        };
    } finally {
        disposeTensor(input);
        if (outputs) {
            for (const tensor of Object.values(outputs)) disposeTensor(tensor);
        }
    }
}

async function encodeVoiceAudio(audioData) {
    const durationSeconds = audioData.length / SAMPLE_RATE;
    let webGpuError = null;

    try {
        const encoderSession = await ensureMimiEncoderSession();
        return await runVoiceEncoder(encoderSession, audioData);
    } catch (error) {
        webGpuError = error;
        if (!ttsUsesWebGpu) {
            throw new Error(
                `Voice encoder failed for ${durationSeconds.toFixed(1)}s of audio: ${describeRuntimeError(error)}`
            );
        }
    } finally {
        // The copied embedding is all generation needs. Release the enrollment
        // model so the full pipeline does not retain an idle encoder.
        await releaseMimiEncoderSession();
    }

    postMessage({
        type: 'status',
        status: 'Voice encoder retrying on CPU...',
        state: 'loading'
    });

    try {
        const fallbackSession = await ensureMimiEncoderSession(true);
        return await runVoiceEncoder(fallbackSession, audioData);
    } catch (fallbackError) {
        throw new Error(
            `Voice encoder failed for ${durationSeconds.toFixed(1)}s of audio. ` +
            `WebGPU: ${describeRuntimeError(webGpuError)}. ` +
            `CPU fallback: ${describeRuntimeError(fallbackError)}.`
        );
    } finally {
        await releaseMimiEncoderSession();
        postMessage({ type: 'status', status: 'Ready', state: 'idle' });
    }
}

// Hardcoded state shapes extracted from ONNX model metadata
// These are the initial shapes - dynamic dimensions start at 0
const FLOW_LM_CACHE_LAYERS = 6;

const FLOW_LM_STATE_SHAPES = {
    // KV cache layers: [kv=2, batch=1, max_seq, heads=16, head_dim=64]
    state_0: { shape: [2, 1, STATE_CACHE_FRAMES, 16, 64], dtype: 'float32' },
    state_1: { shape: [0], dtype: 'float32' },  // dynamic
    state_2: { shape: [1], dtype: 'int64' },    // step counter
    state_3: { shape: [2, 1, STATE_CACHE_FRAMES, 16, 64], dtype: 'float32' },
    state_4: { shape: [0], dtype: 'float32' },
    state_5: { shape: [1], dtype: 'int64' },
    state_6: { shape: [2, 1, STATE_CACHE_FRAMES, 16, 64], dtype: 'float32' },
    state_7: { shape: [0], dtype: 'float32' },
    state_8: { shape: [1], dtype: 'int64' },
    state_9: { shape: [2, 1, STATE_CACHE_FRAMES, 16, 64], dtype: 'float32' },
    state_10: { shape: [0], dtype: 'float32' },
    state_11: { shape: [1], dtype: 'int64' },
    state_12: { shape: [2, 1, STATE_CACHE_FRAMES, 16, 64], dtype: 'float32' },
    state_13: { shape: [0], dtype: 'float32' },
    state_14: { shape: [1], dtype: 'int64' },
    state_15: { shape: [2, 1, STATE_CACHE_FRAMES, 16, 64], dtype: 'float32' },
    state_16: { shape: [0], dtype: 'float32' },
    state_17: { shape: [1], dtype: 'int64' },
};

const MIMI_DECODER_STATE_SHAPES = {
    state_0: { shape: [1], dtype: 'bool' },
    state_1: { shape: [1, 512, 6], dtype: 'float32' },
    state_2: { shape: [1], dtype: 'bool' },
    state_3: { shape: [1, 64, 2], dtype: 'float32' },
    state_4: { shape: [1, 256, 6], dtype: 'float32' },
    state_5: { shape: [1], dtype: 'bool' },
    state_6: { shape: [1, 256, 2], dtype: 'float32' },
    state_7: { shape: [1], dtype: 'bool' },
    state_8: { shape: [1, 128, 0], dtype: 'float32' },  // dynamic
    state_9: { shape: [1, 128, 5], dtype: 'float32' },
    state_10: { shape: [1], dtype: 'bool' },
    state_11: { shape: [1, 128, 2], dtype: 'float32' },
    state_12: { shape: [1], dtype: 'bool' },
    state_13: { shape: [1, 64, 0], dtype: 'float32' },  // dynamic
    state_14: { shape: [1, 64, 4], dtype: 'float32' },
    state_15: { shape: [1], dtype: 'bool' },
    state_16: { shape: [1, 64, 2], dtype: 'float32' },
    state_17: { shape: [1], dtype: 'bool' },
    state_18: { shape: [1, 32, 0], dtype: 'float32' },  // dynamic
    state_19: { shape: [2, 1, 8, STATE_CACHE_FRAMES, 64], dtype: 'float32' },
    state_20: { shape: [1], dtype: 'int64' },
    state_21: { shape: [1], dtype: 'int64' },
    state_22: { shape: [2, 1, 8, STATE_CACHE_FRAMES, 64], dtype: 'float32' },
    state_23: { shape: [1], dtype: 'int64' },
    state_24: { shape: [1], dtype: 'int64' },
    state_25: { shape: [1], dtype: 'bool' },
    state_26: { shape: [1, 512, 16], dtype: 'float32' },
    state_27: { shape: [1], dtype: 'bool' },
    state_28: { shape: [1, 1, 6], dtype: 'float32' },
    state_29: { shape: [1], dtype: 'bool' },
    state_30: { shape: [1, 64, 2], dtype: 'float32' },
    state_31: { shape: [1], dtype: 'bool' },
    state_32: { shape: [1, 32, 0], dtype: 'float32' },  // dynamic
    state_33: { shape: [1], dtype: 'bool' },
    state_34: { shape: [1, 512, 2], dtype: 'float32' },
    state_35: { shape: [1], dtype: 'bool' },
    state_36: { shape: [1, 64, 4], dtype: 'float32' },
    state_37: { shape: [1], dtype: 'bool' },
    state_38: { shape: [1, 128, 2], dtype: 'float32' },
    state_39: { shape: [1], dtype: 'bool' },
    state_40: { shape: [1, 64, 0], dtype: 'float32' },  // dynamic
    state_41: { shape: [1], dtype: 'bool' },
    state_42: { shape: [1, 128, 5], dtype: 'float32' },
    state_43: { shape: [1], dtype: 'bool' },
    state_44: { shape: [1, 256, 2], dtype: 'float32' },
    state_45: { shape: [1], dtype: 'bool' },
    state_46: { shape: [1, 128, 0], dtype: 'float32' },  // dynamic
    state_47: { shape: [1], dtype: 'bool' },
    state_48: { shape: [1, 256, 6], dtype: 'float32' },
    state_49: { shape: [2, 1, 8, STATE_CACHE_FRAMES, 64], dtype: 'float32' },
    state_50: { shape: [1], dtype: 'int64' },
    state_51: { shape: [1], dtype: 'int64' },
    state_52: { shape: [2, 1, 8, STATE_CACHE_FRAMES, 64], dtype: 'float32' },
    state_53: { shape: [1], dtype: 'int64' },
    state_54: { shape: [1], dtype: 'int64' },
    state_55: { shape: [1, 512, 16], dtype: 'float32' },
};

function initState(session, stateShapes) {
    /**
     * Initialize state tensors for a stateful ONNX model using hardcoded shapes.
     */
    const state = {};

    for (const inputName of session.inputNames) {
        if (inputName.startsWith('state_')) {
            const stateInfo = stateShapes[inputName];
            if (!stateInfo) {
                console.warn(`Unknown state input: ${inputName}, skipping`);
                continue;
            }

            const { shape, dtype } = stateInfo;
            const size = shape.reduce((a, b) => a * b, 1);

            let data;
            if (dtype === 'int64') {
                data = new BigInt64Array(size);
            } else if (dtype === 'bool') {
                data = new Uint8Array(size);
            } else {
                data = new Float32Array(size);
            }

            state[inputName] = new ort.Tensor(dtype, data, shape);
            if (DEBUG_LOGS) {
                console.log(`Init state ${inputName}: shape=${JSON.stringify(shape)}, dtype=${dtype}`);
            }
        }
    }

    return state;
}

// setTimeout is clamped to at least a millisecond per call, which the AR loop
// pays dozens of times per utterance. A message-channel ping drains the same
// pending worker messages for a fraction of that.
const yieldChannel = new MessageChannel();
let pendingYield = null;
yieldChannel.port1.onmessage = () => {
    const resolve = pendingYield;
    pendingYield = null;
    if (resolve) resolve();
};

function yieldToMessageQueue() {
    return new Promise((resolve) => {
        pendingYield = resolve;
        yieldChannel.port2.postMessage(0);
    });
}

function disposeTensor(tensor) {
    if (!tensor || typeof tensor.dispose !== 'function') return;
    try {
        tensor.dispose();
    } catch (err) {
        if (DEBUG_LOGS) console.warn('Tensor disposal failed:', err);
    }
}

function disposeState(state, protectedTensors = NO_PROTECTED_TENSORS) {
    if (!state) return;
    for (const tensor of Object.values(state)) {
        if (!protectedTensors.has(tensor)) disposeTensor(tensor);
    }
}

function replaceNamedStateOutputs(state, result, outputNames, protectedTensors = NO_PROTECTED_TENSORS) {
    for (const outputName of outputNames) {
        if (!outputName.startsWith('out_state_')) continue;
        const stateName = outputName.slice(4);
        const nextTensor = result[outputName];
        if (!nextTensor) continue;
        const previousTensor = state[stateName];
        state[stateName] = nextTensor;
        if (previousTensor !== nextTensor && !protectedTensors.has(previousTensor)) {
            disposeTensor(previousTensor);
        }
    }
}

// Moves the large float32 tensors of a state onto the GPU once, so runs that
// start from it do not upload them again. Small and integer tensors stay on
// the CPU; the runtime uploads those in microseconds.
const RESIDENT_UPLOAD_MIN_BYTES = 1 << 20;

function uploadStateToGpu(state) {
    const device = ttsUsesWebGpu ? ort?.env?.webgpu?.device : null;
    if (!device || typeof ort.Tensor.fromGpuBuffer !== 'function') return state;

    const uploaded = {};
    for (const [name, tensor] of Object.entries(state)) {
        const onCpu = tensor.location === 'cpu' || tensor.location === 'cpu-pinned';
        if (!onCpu || tensor.type !== 'float32' || tensor.data.byteLength < RESIDENT_UPLOAD_MIN_BYTES) {
            uploaded[name] = tensor;
            continue;
        }
        try {
            const size = Math.ceil(tensor.data.byteLength / 16) * 16;
            const buffer = device.createBuffer({
                size,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(buffer, 0, tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength);
            uploaded[name] = ort.Tensor.fromGpuBuffer(buffer, {
                dataType: 'float32',
                dims: [...tensor.dims],
                dispose: () => buffer.destroy(),
            });
        } catch (err) {
            console.warn(`Could not keep ${name} on the GPU; using the CPU copy:`, err);
            uploaded[name] = tensor;
        }
    }
    return uploaded;
}

function releaseResidentVoiceState() {
    if (!residentVoiceState) return;
    const { state } = residentVoiceState;
    residentVoiceState = null;
    disposeState(state);
}

function makeResident(state) {
    return { state, tensors: new Set(Object.values(state)) };
}

function disposeResultExcept(result, keptOutputNames) {
    for (const [name, tensor] of Object.entries(result)) {
        if (!keptOutputNames.has(name)) disposeTensor(tensor);
    }
}

// Timing marks for the current generation, relative to the generate message.
// The page merges them into its per-utterance profile.
let generationTimings = null;

function markTiming(name) {
    if (generationTimings && generationTimings[name] === undefined) {
        generationTimings[name] = Math.round(performance.now() - generationTimings.startedAt);
    }
}

async function startGeneration(text, voiceName, requestId) {
    let generationError = null;
    isGenerating = true;
    generationTimings = { startedAt: performance.now() };
    // Note: LSD is now controlled by the main thread via set_lsd messages
    // Previously reset to MAX_LSD here, but that overrode user settings
    postMessage({ type: 'status', status: 'Generating...', state: 'running' });
    postMessage({ type: 'generation_started', data: { time: performance.now() } });

    try {
        // Split text into sentence chunks (target <= CHUNK_TARGET_TOKENS tokens)
        const chunks = splitIntoBestSentences(text);
        console.log(`Split into ${chunks.length} chunks:`, chunks);

        if (chunks.length === 0) {
            throw new Error('No text to generate');
        }

        // Get voice embedding
        let voiceEmb = currentVoiceEmbedding;
        if (voiceName && voiceName !== currentVoiceName) {
            if (predefinedVoiceNames.includes(voiceName)) {
                voiceEmb = await loadPredefinedVoice(voiceName);
                currentVoiceEmbedding = voiceEmb;
                currentVoiceName = voiceName;
            }
        }

        if (!voiceEmb) {
            throw new Error('No voice embedding available. Please select a voice or upload custom audio.');
        }

        markTiming('textPreparedMs');
        // Run generation pipeline with chunks
        await runGenerationPipeline(voiceEmb, chunks);

    } catch (err) {
        console.error('Generation error:', err);
        generationError = err.toString();
    } finally {
        if (isGenerating) {
            postMessage({ type: 'stream_ended' });
            postMessage({ type: 'status', status: 'Finished', state: 'idle' });
        }
        isGenerating = false;
        // One terminal acknowledgement, after synthesis and cleanup, including
        // cancellation. Consumers may safely begin the next track after this.
        postMessage({ type: 'generation_complete', requestId, error: generationError });
    }
}

function setFlowCacheFrames(frames) {
    STATE_CACHE_FRAMES = frames;
    for (const state of Object.values(FLOW_LM_STATE_SHAPES)) {
        if (state.shape.length === 5) state.shape[2] = frames;
    }
}

async function ensureFlowCacheCapacity(voiceEmb, chunks) {
    let voiceFrames;
    if (voiceEmb.type === 'flow_state_bundle') {
        voiceFrames = 0;
        for (let layer = 0; layer < FLOW_LM_CACHE_LAYERS; layer++) {
            const record = voiceEmb.tensors[`transformer.layers.${layer}.self_attn/offset`];
            const cache = voiceEmb.tensors[`transformer.layers.${layer}.self_attn/cache`];
            voiceFrames = Math.max(voiceFrames, Number(copyTensorDataFromBundle(voiceEmb, record)[0]), cache.shape[2]);
        }
    } else {
        voiceFrames = voiceEmb.shape[1] + (bundleMetadata?.insert_bos_before_voice && bosBeforeVoice ? bosBeforeVoice.shape[1] : 0);
    }
    const textFrames = Math.max(...chunks.map(text => tokenizerProcessor.encodeIds(text).length));
    const required = requiredFlowCapacity(voiceFrames, textFrames, MAX_FRAMES);
    if (required <= STATE_CACHE_FRAMES && flowLmMainSession) return;
    // Rare oversized input: release the compact session before restoring the
    // original. Preserve all voice/text context and the generation limit.
    releaseResidentVoiceState();
    customVoiceFlowState = null;
    await flowLmMainSession?.release();
    flowLmMainSession = null;
    flowLmMainSession = await createSessionFromUrl(MODELS.flow_lm_main, flowSessionOptions);
    setFlowCacheFrames(1000);
}

async function runGenerationPipeline(voiceEmb, chunks) {
    await ensureFlowCacheCapacity(voiceEmb, chunks);
    // The decoder always starts from the same zeroed state; keep one resident
    // copy and seed every utterance from it.
    function freshMimiState() {
        if (!useResidentState) {
            return { state: initState(mimiDecoderSession, MIMI_DECODER_STATE_SHAPES), protectedTensors: NO_PROTECTED_TENSORS };
        }
        if (!residentMimiState) {
            residentMimiState = makeResident(
                uploadStateToGpu(initState(mimiDecoderSession, MIMI_DECODER_STATE_SHAPES))
            );
        }
        return { state: { ...residentMimiState.state }, protectedTensors: residentMimiState.tensors };
    }

    // Initialize state - may be reset per chunk
    let { state: mimiState, protectedTensors: protectedMimiTensors } = freshMimiState();
    const emptySeq = new ort.Tensor('float32', new Float32Array(0), [1, 0, 32]);
    const emptyTextEmb = new ort.Tensor('float32', new Float32Array(0), [1, 0, 1024]);

    async function buildVoiceConditionedState() {
        if (useResidentState) {
            if (!residentVoiceState || residentVoiceState.source !== voiceEmb) {
                releaseResidentVoiceState();
                const built = await conditionVoiceState();
                residentVoiceState = { source: voiceEmb, ...makeResident(uploadStateToGpu(built)) };
            }
            return { state: { ...residentVoiceState.state }, protectedTensors: residentVoiceState.tensors };
        }
        return { state: await conditionVoiceState(), protectedTensors: NO_PROTECTED_TENSORS };
    }

    async function conditionVoiceState() {
        if (voiceEmb.type === 'flow_state_bundle') {
            console.log('Using precomputed voice state');
            postMessage({ type: 'status', status: 'Using precomputed voice...', state: 'running' });
            return createFlowStateFromPrecomputedVoice(voiceEmb);
        }

        if (customVoiceFlowState && customVoiceFlowState.source === voiceEmb) {
            console.log('Using cached custom voice state');
            return createFlowStateFromSnapshot(customVoiceFlowState.snapshot);
        }

        let flowLmState = initState(flowLmMainSession, FLOW_LM_STATE_SHAPES);
        const voiceTensor = createVoiceConditioningTensor(voiceEmb);
        console.log('Voice embeddings shape:', voiceEmb.shape);
        console.log('Running voice conditioning...');
        postMessage({ type: 'status', status: 'Conditioning voice...', state: 'running' });
        const voiceCondInputs = {
            sequence: emptySeq,
            text_embeddings: voiceTensor,
            ...flowLmState
        };

        const condResult = await flowLmMainSession.run(voiceCondInputs);
        disposeTensor(voiceTensor);

        // Update state from voice conditioning
        replaceNamedStateOutputs(flowLmState, condResult, flowLmMainSession.outputNames);
        disposeResultExcept(
            condResult,
            new Set(flowLmMainSession.outputNames.filter(name => name.startsWith('out_state_')))
        );

        // Every later chunk rebuilds this state from the snapshot instead of
        // running the prefill again.
        customVoiceFlowState = {
            source: voiceEmb,
            snapshot: await snapshotFlowLmState(flowLmState),
        };
        return flowLmState;
    }

    let flowLmState = null;
    let protectedFlowTensors = NO_PROTECTED_TENSORS;

    try {
    ({ state: flowLmState, protectedTensors: protectedFlowTensors } = await buildVoiceConditionedState());
    markTiming('voiceStateMs');

    // Streaming parameters
    // Four frames produce 320 ms of audio. Emitting the same small batch
    // throughout generation avoids the long post-start wait that occurred when
    // a 160 ms first chunk was followed by a 960 ms batch.
    const STREAM_CHUNK_FRAMES = streamChunkFrames;

    // Tracking across all chunks
    let generatedLatentCount = 0;
    let isFirstAudioChunk = true;
    let totalDecodedFrames = 0;
    let totalFlowLmTime = 0;
    let totalDecodeTime = 0;
    const arStartTime = performance.now();

    // Process each text chunk
    for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {
        if (!isGenerating) break;

        if (RESET_FLOW_STATE_EACH_CHUNK && chunkIdx > 0) {
            disposeState(flowLmState, protectedFlowTensors);
            ({ state: flowLmState, protectedTensors: protectedFlowTensors } = await buildVoiceConditionedState());
        }
        if (RESET_MIMI_STATE_EACH_CHUNK && chunkIdx > 0) {
            disposeState(mimiState, protectedMimiTensors);
            ({ state: mimiState, protectedTensors: protectedMimiTensors } = freshMimiState());
        }

        const chunkText = chunks[chunkIdx];
        console.log(`Processing chunk ${chunkIdx + 1}/${chunks.length}: "${chunkText}"`);
        postMessage({ type: 'status', status: `Preparing text chunk ${chunkIdx + 1}/${chunks.length}...`, state: 'running' });

        let isFirstAudioChunkOfTextChunk = true;

        // Tokenize this chunk
        const tokenIds = tokenizerProcessor.encodeIds(chunkText);
        console.log(`Chunk ${chunkIdx + 1} tokens:`, tokenIds.length);

        // Text conditioning for this chunk
        const textInput = new ort.Tensor('int64', BigInt64Array.from(tokenIds.map(x => BigInt(x))), [1, tokenIds.length]);
        const textCondResult = await textConditionerSession.run({ token_ids: textInput });
        if (chunkIdx === 0) markTiming('textConditionerMs');
        let textEmb = textCondResult[textConditionerSession.outputNames[0]];

        if (textEmb.dims.length === 2) {
            textEmb = new ort.Tensor('float32', textEmb.data, [1, textEmb.dims[0], textEmb.dims[1]]);
        }

        const textCondInputs = {
            sequence: emptySeq,
            text_embeddings: textEmb,
            ...flowLmState
        };

        postMessage({ type: 'status', status: `Conditioning text chunk ${chunkIdx + 1}/${chunks.length}...`, state: 'running' });
        const condResult = await flowLmMainSession.run(textCondInputs);
        if (chunkIdx === 0) markTiming('flowConditionedMs');
        disposeResultExcept(textCondResult, new Set());
        disposeTensor(textInput);
        disposeTensor(textEmb);

        // Update state from text conditioning
        replaceNamedStateOutputs(flowLmState, condResult, flowLmMainSession.outputNames, protectedFlowTensors);
        disposeResultExcept(
            condResult,
            new Set(flowLmMainSession.outputNames.filter(name => name.startsWith('out_state_')))
        );

        // AR generation for this chunk
        const chunkLatents = [];
        let currentLatent = new ort.Tensor('float32', new Float32Array(32).fill(NaN), [1, 1, 32]);
        let chunkDecodedFrames = 0;
        const FRAMES_AFTER_EOS = 3;  // Match PyTorch behavior - generate extra frames after EOS
        let eosStep = null;

        let chunkEnded = false;
        let chunkGenTimeMs = 0;
        postMessage({ type: 'status', status: `Generating audio frames...`, state: 'running' });

        async function decodePendingFrames(isLastChunk) {
            const pending = chunkLatents.length - chunkDecodedFrames;
            if (pending <= 0) return;

            if (DEBUG_LOGS) {
                postMessage({
                    type: 'status',
                    status: `Decoding ${pending} audio frame${pending === 1 ? '' : 's'}...`,
                    state: 'running'
                });
            }

            const decodeLatents = new Float32Array(pending * 32);
            for (let i = 0; i < pending; i++) {
                decodeLatents.set(chunkLatents[chunkDecodedFrames + i], i * 32);
            }

            const latentTensor = new ort.Tensor('float32', decodeLatents, [1, pending, 32]);
            const decodeInputs = { latent: latentTensor, ...mimiState };
            const decStart = performance.now();
            const decodeResult = await mimiDecoderSession.run(decodeInputs);
            disposeTensor(latentTensor);
            const decElapsed = performance.now() - decStart;
            totalDecodeTime += decElapsed;
            chunkGenTimeMs += decElapsed;
            const audioChunk = decodeResult[mimiDecoderSession.outputNames[0]].data;

            const keptDecoderOutputs = new Set();
            for (let i = 1; i < mimiDecoderSession.outputNames.length; i++) {
                const outputName = mimiDecoderSession.outputNames[i];
                const stateName = `state_${i - 1}`;
                const nextTensor = decodeResult[outputName];
                if (!nextTensor) continue;
                keptDecoderOutputs.add(outputName);
                const previousTensor = mimiState[stateName];
                mimiState[stateName] = nextTensor;
                if (previousTensor !== nextTensor && !protectedMimiTensors.has(previousTensor)) {
                    disposeTensor(previousTensor);
                }
            }
            disposeResultExcept(decodeResult, keptDecoderOutputs);

            chunkDecodedFrames += pending;
            totalDecodedFrames += pending;

            const audioFloat32 = new Float32Array(audioChunk);
            postMessage({
                type: 'audio_chunk',
                data: audioFloat32,
                metrics: {
                    bbTime: 0,
                    decTime: 0,
                    chunkDuration: audioFloat32.length / SAMPLE_RATE,
                    genTimeSec: chunkGenTimeMs / 1000,
                    isFirst: isFirstAudioChunk,
                    isLast: isLastChunk,
                    chunkStart: isFirstAudioChunkOfTextChunk,
                    flushed: true
                }
            }, [audioFloat32.buffer]);

            isFirstAudioChunk = false;
            isFirstAudioChunkOfTextChunk = false;
            chunkGenTimeMs = 0;
        }

        for (let step = 0; step < MAX_FRAMES; step++) {
            if (!isGenerating) break;

            // Yield every 4 steps to allow message processing (e.g., set_lsd)
            if (step > 0 && step % 4 === 0) {
                await yieldToMessageQueue();
            }

            const arInputs = {
                sequence: currentLatent,
                text_embeddings: emptyTextEmb,
                ...flowLmState
            };

            const stepStart = performance.now();
            const previousLatent = currentLatent;
            const arResult = await flowLmMainSession.run(arInputs);
            disposeTensor(previousLatent);
            const stepElapsed = performance.now() - stepStart;
            chunkGenTimeMs += stepElapsed;
            if (chunkIdx === 0 && step === 0) {
                markTiming('firstFrameMs');
                if (generationTimings) generationTimings.firstStepMs = Math.round(stepElapsed);
            }
            if (DEBUG_LOGS && (step === 0 || (step + 1) % 5 === 0)) {
                postMessage({
                    type: 'status',
                    status: `Generating frame ${step + 1} (${stepElapsed.toFixed(0)}ms)...`,
                    state: 'running'
                });
            }

            const conditioning = arResult['conditioning'];
            const eosLogit = arResult['eos_logit'].data[0];
            const isEos = eosLogit > -4.0;

            // Track when EOS is first detected
            if (isEos && eosStep === null) {
                eosStep = step;
            }

            // Only stop after FRAMES_AFTER_EOS additional frames
            const shouldStop = eosStep !== null && step >= eosStep + FRAMES_AFTER_EOS;

            // Flow matching (LSD loop) - uses currentLSD which can be adjusted dynamically
            const TEMP = 0.7;
            const STD = Math.sqrt(TEMP);
            let xData = new Float32Array(32);
            for (let i = 0; i < 32; i++) {
                let u = 0, v = 0;
                while (u === 0) u = Math.random();
                while (v === 0) v = Math.random();
                xData[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * STD;
            }

            const lsdSteps = currentLSD;
            const dt = 1.0 / lsdSteps;

            for (let j = 0; j < lsdSteps; j++) {
                const xTensor = new ort.Tensor('float32', xData, [1, 32]);
                const flowInputs = {
                    c: conditioning,
                    s: stTensors[lsdSteps][j].s,
                    t: stTensors[lsdSteps][j].t,
                    x: xTensor
                };

                const flowResult = await flowLmFlowSession.run(flowInputs);
                const v = flowResult['flow_dir'].data;

                for (let k = 0; k < 32; k++) {
                    xData[k] += v[k] * dt;
                }
                disposeTensor(xTensor);
                disposeResultExcept(flowResult, new Set());
            }

            totalFlowLmTime += stepElapsed;

            const latentData = xData;
            chunkLatents.push(new Float32Array(latentData));
            generatedLatentCount++;

            // Update state
            currentLatent = new ort.Tensor('float32', latentData, [1, 1, 32]);
            replaceNamedStateOutputs(flowLmState, arResult, flowLmMainSession.outputNames, protectedFlowTensors);
            disposeResultExcept(
                arResult,
                new Set(flowLmMainSession.outputNames.filter(name => name.startsWith('out_state_')))
            );

            // Decode audio chunks
            const pending = chunkLatents.length - chunkDecodedFrames;
            let decodeSize = 0;

            if (shouldStop) {
                decodeSize = pending;
            } else if (pending >= STREAM_CHUNK_FRAMES) {
                decodeSize = STREAM_CHUNK_FRAMES;
            }

            if (decodeSize > 0) {
                if (DEBUG_LOGS) {
                    postMessage({
                        type: 'status',
                        status: `Decoding ${decodeSize} audio frame${decodeSize === 1 ? '' : 's'}...`,
                        state: 'running'
                    });
                }
                const decodeLatents = new Float32Array(decodeSize * 32);
                for (let i = 0; i < decodeSize; i++) {
                    decodeLatents.set(chunkLatents[chunkDecodedFrames + i], i * 32);
                }

                const latentTensor = new ort.Tensor('float32', decodeLatents, [1, decodeSize, 32]);
                const decodeInputs = { latent: latentTensor, ...mimiState };

                const decStart = performance.now();
                const decodeResult = await mimiDecoderSession.run(decodeInputs);
                disposeTensor(latentTensor);
                const decElapsed = performance.now() - decStart;
                totalDecodeTime += decElapsed;
                chunkGenTimeMs += decElapsed;
                const audioChunk = decodeResult[mimiDecoderSession.outputNames[0]].data;

                // Update MIMI state
                const keptDecoderOutputs = new Set();
                for (let i = 1; i < mimiDecoderSession.outputNames.length; i++) {
                    const outputName = mimiDecoderSession.outputNames[i];
                    const stateName = `state_${i - 1}`;
                    const nextTensor = decodeResult[outputName];
                    if (!nextTensor) continue;
                    keptDecoderOutputs.add(outputName);
                    const previousTensor = mimiState[stateName];
                    mimiState[stateName] = nextTensor;
                    if (previousTensor !== nextTensor && !protectedMimiTensors.has(previousTensor)) {
                        disposeTensor(previousTensor);
                    }
                }
                disposeResultExcept(decodeResult, keptDecoderOutputs);

                chunkDecodedFrames += decodeSize;
                totalDecodedFrames += decodeSize;

                const audioFloat32 = new Float32Array(audioChunk);
                const isLastChunk = shouldStop && chunkIdx === chunks.length - 1;
                if (isFirstAudioChunk) {
                    markTiming('firstAudioMs');
                    if (generationTimings) {
                        generationTimings.firstDecodeMs = Math.round(decElapsed);
                        generationTimings.framesInFirstAudio = decodeSize;
                    }
                }
                postMessage({
                    type: 'audio_chunk',
                    data: audioFloat32,
                    metrics: {
                        bbTime: 0,
                        decTime: 0,
                        chunkDuration: audioFloat32.length / SAMPLE_RATE,
                        genTimeSec: chunkGenTimeMs / 1000,
                        isFirst: isFirstAudioChunk,
                        isLast: isLastChunk,
                        chunkStart: isFirstAudioChunkOfTextChunk,
                        timings: isFirstAudioChunk ? { ...generationTimings } : undefined
                    }
                }, [audioFloat32.buffer]);

                isFirstAudioChunk = false;
                isFirstAudioChunkOfTextChunk = false;
                chunkGenTimeMs = 0;
            }

            if (shouldStop) {
                console.log(`Chunk ${chunkIdx + 1} EOS at step ${eosStep}, stopped at step ${step}, ${chunkLatents.length} frames`);
                chunkEnded = true;
                break;
            }
        }

        if (isGenerating && chunkDecodedFrames < chunkLatents.length) {
            await decodePendingFrames(chunkIdx === chunks.length - 1);
        }

        disposeTensor(currentLatent);
    }

    const totalTime = (performance.now() - arStartTime) / 1000;
    const audioSeconds = generatedLatentCount * SAMPLES_PER_FRAME / SAMPLE_RATE;

    // RTFx based on actual generation time (flow LM + decoder), not including conditioning
    const genTime = (totalFlowLmTime + totalDecodeTime) / 1000;
    const rtfx = audioSeconds / genTime;

    console.log(`Generation complete: ${generatedLatentCount} frames (${audioSeconds.toFixed(2)}s audio)`);
    console.log(`  Total time: ${totalTime.toFixed(2)}s`);
    console.log(`  Gen time: ${genTime.toFixed(2)}s, RTFx: ${rtfx.toFixed(2)}x`);
    console.log(`  Flow LM: ${(totalFlowLmTime / 1000).toFixed(2)}s (${(totalFlowLmTime / Math.max(1, generatedLatentCount)).toFixed(1)}ms/step)`);
    console.log(`  Decoder: ${(totalDecodeTime / 1000).toFixed(2)}s`);

    markTiming('finishedMs');
    postMessage({
        type: 'status',
        status: `Finished (RTFx: ${rtfx.toFixed(2)}x)`,
        state: 'idle',
        metrics: {
            rtfx,
            genTime,
            totalTime,
            audioDuration: audioSeconds,
            frames: generatedLatentCount,
            flowMsPerStep: totalFlowLmTime / Math.max(1, generatedLatentCount),
            decodeMs: totalDecodeTime,
            timings: { ...generationTimings },
        }
    });
    } finally {
        disposeState(flowLmState, protectedFlowTensors);
        disposeState(mimiState, protectedMimiTensors);
        disposeTensor(emptySeq);
        disposeTensor(emptyTextEmb);
    }
}

// Pre-allocated buffers for step counter updates (avoid GC pressure in hot loop)
const stepBuffers = {};

function updateStateSteps(state, increment) {
    // Update step counters in state dict - reuse buffers to avoid allocation
    const incBigInt = BigInt(increment);
    for (const key in state) {
        if (key.includes('step') && state[key]) {
            const tensor = state[key];
            if (tensor.data instanceof BigInt64Array) {
                // Reuse buffer if same size, otherwise create new one
                if (!stepBuffers[key] || stepBuffers[key].length !== tensor.data.length) {
                    stepBuffers[key] = new BigInt64Array(tensor.data.length);
                }
                const buf = stepBuffers[key];
                for (let i = 0; i < tensor.data.length; i++) {
                    buf[i] = tensor.data[i] + incBigInt;
                }
                state[key] = new ort.Tensor('int64', buf, tensor.dims);
            }
        }
    }
}
