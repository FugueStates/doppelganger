/**
 * Minimal WAV reader — enough to read the WAV files `renderPreFxAudio` produces.
 *
 * Why hand-roll this instead of using the `audio-decode` npm package?
 *  - We only ever read *our own* freshly rendered WAV here, so we don't need a
 *    full multi-format codec (mp3/ogg/flac/…).
 *  - `audio-decode` pulls in ~4 MB of dependencies; bundling that into the
 *    extension is a load-time risk and bloats the package. This is ~100 lines.
 *
 * Supports the formats a DAW render realistically uses: PCM integer (16/24/32-bit)
 * and IEEE float (32-bit), including the WAVE_FORMAT_EXTENSIBLE wrapper.
 */

export interface DecodedWav {
  sampleRate: number;
  numberOfChannels: number;
  duration: number; // seconds
  getChannelData(channel: number): Float32Array;
}

const FORMAT_PCM = 0x0001;
const FORMAT_IEEE_FLOAT = 0x0003;
const FORMAT_EXTENSIBLE = 0xfffe;

export function decodeWav(buffer: Buffer): DecodedWav {
  if (buffer.toString("ascii", 0, 4) !== "RIFF" || buffer.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error("Not a RIFF/WAVE file");
  }

  const fmt = readFmtChunk(buffer);
  const data = findChunk(buffer, "data");
  if (!data) throw new Error("WAV has no data chunk");

  const { numChannels, sampleRate, bitsPerSample, formatTag } = fmt;
  const bytesPerSample = bitsPerSample / 8;
  const frameCount = Math.floor(data.length / (bytesPerSample * numChannels));

  // De-interleave into one Float32Array per channel, normalized to [-1, 1].
  const channels: Float32Array[] = Array.from(
    { length: numChannels },
    () => new Float32Array(frameCount),
  );
  const readSample = sampleReader(formatTag, bitsPerSample);

  for (let frame = 0; frame < frameCount; frame++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const offset = data.start + (frame * numChannels + ch) * bytesPerSample;
      channels[ch]![frame] = readSample(buffer, offset);
    }
  }

  return {
    sampleRate,
    numberOfChannels: numChannels,
    duration: frameCount / sampleRate,
    getChannelData: (ch) => {
      const c = channels[ch];
      if (!c) throw new Error(`No such channel ${ch}`);
      return c;
    },
  };
}

interface FmtChunk {
  formatTag: number;
  numChannels: number;
  sampleRate: number;
  bitsPerSample: number;
}

function readFmtChunk(buffer: Buffer): FmtChunk {
  const fmt = findChunk(buffer, "fmt ");
  if (!fmt) throw new Error("WAV has no fmt chunk");
  const s = fmt.start;

  let formatTag = buffer.readUInt16LE(s);
  const numChannels = buffer.readUInt16LE(s + 2);
  const sampleRate = buffer.readUInt32LE(s + 4);
  const bitsPerSample = buffer.readUInt16LE(s + 14);

  // WAVE_FORMAT_EXTENSIBLE: the real format is the first 2 bytes of the SubFormat GUID.
  if (formatTag === FORMAT_EXTENSIBLE && fmt.length >= 26) {
    formatTag = buffer.readUInt16LE(s + 24);
  }

  return { formatTag, numChannels, sampleRate, bitsPerSample };
}

/** Returns a function that reads one normalized float sample at a byte offset. */
function sampleReader(
  formatTag: number,
  bitsPerSample: number,
): (buf: Buffer, offset: number) => number {
  if (formatTag === FORMAT_IEEE_FLOAT && bitsPerSample === 32) {
    return (buf, o) => buf.readFloatLE(o);
  }
  if (formatTag === FORMAT_PCM) {
    if (bitsPerSample === 16) return (buf, o) => buf.readInt16LE(o) / 32768;
    if (bitsPerSample === 24) return (buf, o) => read24LE(buf, o) / 8388608;
    if (bitsPerSample === 32) return (buf, o) => buf.readInt32LE(o) / 2147483648;
  }
  throw new Error(`Unsupported WAV format: tag=${formatTag}, bits=${bitsPerSample}`);
}

function read24LE(buf: Buffer, o: number): number {
  const v = buf[o]! | (buf[o + 1]! << 8) | (buf[o + 2]! << 16);
  return v & 0x800000 ? v - 0x1000000 : v; // sign-extend
}

interface ChunkRef {
  start: number; // byte offset of chunk *data*
  length: number;
}

/** Walks the RIFF chunk list and returns the first chunk with the given id. */
function findChunk(buffer: Buffer, id: string): ChunkRef | null {
  let pos = 12; // skip "RIFF" + size + "WAVE"
  while (pos + 8 <= buffer.length) {
    const chunkId = buffer.toString("ascii", pos, pos + 4);
    const chunkSize = buffer.readUInt32LE(pos + 4);
    const dataStart = pos + 8;
    if (chunkId === id) return { start: dataStart, length: chunkSize };
    // chunks are word-aligned: pad odd sizes by 1
    pos = dataStart + chunkSize + (chunkSize % 2);
  }
  return null;
}
