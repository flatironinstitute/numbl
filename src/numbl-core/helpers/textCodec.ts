/**
 * Byte <-> text conversion for the file I/O encodings numbl supports
 * (fopen's encoding argument, fileread, fread/fwrite of char data).
 *
 * Decoded text is a JS string of UTF-16 code units, which is what MATLAB
 * char arrays hold.
 */

export interface TextEncoding {
  /** Canonical name, as reported by `[~,~,~,enc] = fopen(fid)`. */
  readonly name: string;
  /** Decode bytes[start, end), stopping after `maxChars` chars if given.
   *  Returns the text and the byte offset where decoding stopped. */
  decode(
    bytes: Uint8Array,
    start: number,
    end: number,
    maxChars?: number
  ): { text: string; end: number };
  encode(text: string): Uint8Array;
}

/** MATLAB writes this byte for characters the encoding cannot represent. */
const SUBSTITUTE = 26;

function codesToString(codes: number[]): string {
  let s = "";
  const CHUNK = 8192;
  for (let i = 0; i < codes.length; i += CHUNK) {
    s += String.fromCharCode.apply(null, codes.slice(i, i + CHUNK));
  }
  return s;
}

/** UTF-8. With `lenient`, a byte that does not start a valid sequence
 *  decodes to its own value (MATLAB's default for files opened without an
 *  explicit encoding); otherwise it decodes to U+FFFD. */
function utf8(lenient: boolean): TextEncoding {
  const encoder = new TextEncoder();
  return {
    name: "UTF-8",
    decode(bytes, start, end, maxChars = Infinity) {
      const codes: number[] = [];
      let i = start;
      while (i < end && codes.length < maxChars) {
        const b = bytes[i];
        if (b < 0x80) {
          codes.push(b);
          i++;
          continue;
        }
        let len = 0;
        let lo = 0x80;
        let hi = 0xbf;
        if (b >= 0xc2 && b <= 0xdf) len = 2;
        else if (b >= 0xe0 && b <= 0xef) {
          len = 3;
          if (b === 0xe0) lo = 0xa0;
          else if (b === 0xed) hi = 0x9f;
        } else if (b >= 0xf0 && b <= 0xf4) {
          len = 4;
          if (b === 0xf0) lo = 0x90;
          else if (b === 0xf4) hi = 0x8f;
        }
        let ok = len > 0 && i + len <= end;
        for (let k = 1; ok && k < len; k++) {
          const c = bytes[i + k];
          if (k === 1 ? c < lo || c > hi : c < 0x80 || c > 0xbf) ok = false;
        }
        if (!ok) {
          codes.push(lenient ? b : 0xfffd);
          i++;
          continue;
        }
        let cp = b & (0xff >> (len + 1));
        for (let k = 1; k < len; k++) cp = (cp << 6) | (bytes[i + k] & 0x3f);
        if (cp > 0xffff) {
          if (codes.length + 2 > maxChars) break;
          cp -= 0x10000;
          codes.push(0xd800 + (cp >> 10), 0xdc00 + (cp & 0x3ff));
        } else {
          codes.push(cp);
        }
        i += len;
      }
      return { text: codesToString(codes), end: i };
    },
    encode: text => encoder.encode(text),
  };
}

/** A single-byte encoding given by its byte -> code point map. */
function singleByte(name: string, table: (b: number) => number): TextEncoding {
  const reverse = new Map<number, number>();
  for (let b = 255; b >= 0; b--) reverse.set(table(b), b);
  return {
    name,
    decode(bytes, start, end, maxChars = Infinity) {
      const stop = Math.min(end, start + maxChars);
      const codes: number[] = new Array(stop - start);
      for (let i = start; i < stop; i++) codes[i - start] = table(bytes[i]);
      return { text: codesToString(codes), end: stop };
    },
    encode(text) {
      const out = new Uint8Array(text.length);
      for (let i = 0; i < text.length; i++) {
        out[i] = reverse.get(text.charCodeAt(i)) ?? SUBSTITUTE;
      }
      return out;
    },
  };
}

// windows-1252 code points for bytes 0x80-0x9F (undefined slots map to
// themselves, as in the WHATWG decoder).
const CP1252_HIGH = [
  0x20ac, 0x81, 0x201a, 0x0192, 0x201e, 0x2026, 0x2020, 0x2021, 0x02c6, 0x2030,
  0x0160, 0x2039, 0x0152, 0x8d, 0x017d, 0x8f, 0x90, 0x2018, 0x2019, 0x201c,
  0x201d, 0x2022, 0x2013, 0x2014, 0x02dc, 0x2122, 0x0161, 0x203a, 0x0153, 0x9d,
  0x017e, 0x0178,
];

const ENCODINGS: Record<string, () => TextEncoding> = {
  "utf-8": () => utf8(false),
  "iso-8859-1": () => singleByte("ISO-8859-1", b => b),
  "windows-1252": () =>
    singleByte("windows-1252", b =>
      b >= 0x80 && b <= 0x9f ? CP1252_HIGH[b - 0x80] : b
    ),
  "us-ascii": () => singleByte("US-ASCII", b => (b < 0x80 ? b : 0xfffd)),
};

const ALIASES: Record<string, string> = {
  utf8: "utf-8",
  latin1: "iso-8859-1",
  "latin-1": "iso-8859-1",
  "iso8859-1": "iso-8859-1",
  iso_8859_1: "iso-8859-1",
  cp1252: "windows-1252",
  ascii: "us-ascii",
};

const cache = new Map<string, TextEncoding>();

/** The encoding for a name as accepted by fopen, or null if unsupported. */
export function lookupTextEncoding(name: string): TextEncoding | null {
  let key = name.trim().toLowerCase();
  key = ALIASES[key] ?? key;
  const make = ENCODINGS[key];
  if (!make) return null;
  let enc = cache.get(key);
  if (!enc) {
    enc = make();
    cache.set(key, enc);
  }
  return enc;
}

/** The encoding used when none is given: UTF-8 that falls back to one char
 *  per byte for invalid sequences, which is what MATLAB does. */
export const DEFAULT_TEXT_ENCODING: TextEncoding = utf8(true);
