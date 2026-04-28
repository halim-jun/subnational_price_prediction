// HMAC-signed cookie helpers for the prototype login barrier.
// Stateless: no DB, no session store. Cookie value = base64(expiry).hex(hmac).
// Runs on Cloudflare Workers via Web Crypto.

export const COOKIE_NAME = "ea_auth";
export const COOKIE_TTL_SECONDS = 60 * 60 * 24 * 7;

const enc = new TextEncoder();

async function hmacKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"]
  );
}

function bytesToHex(buf: ArrayBuffer): string {
  return [...new Uint8Array(buf)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function b64urlEncode(s: string): string {
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64urlDecode(s: string): string {
  const pad = s.length % 4 === 0 ? "" : "=".repeat(4 - (s.length % 4));
  return atob(s.replace(/-/g, "+").replace(/_/g, "/") + pad);
}

export async function issueCookieValue(secret: string): Promise<string> {
  const expires = Math.floor(Date.now() / 1000) + COOKIE_TTL_SECONDS;
  const payload = b64urlEncode(String(expires));
  const key = await hmacKey(secret);
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(payload));
  return `${payload}.${bytesToHex(sig)}`;
}

export async function verifyCookieValue(
  value: string | null | undefined,
  secret: string
): Promise<boolean> {
  if (!value || typeof value !== "string") return false;
  const dot = value.indexOf(".");
  if (dot < 1 || dot === value.length - 1) return false;
  const payload = value.slice(0, dot);
  const sigHex = value.slice(dot + 1);

  let expires: number;
  try {
    expires = parseInt(b64urlDecode(payload), 10);
  } catch {
    return false;
  }
  if (!Number.isFinite(expires) || expires < Math.floor(Date.now() / 1000)) {
    return false;
  }

  const key = await hmacKey(secret);
  const sigBytes = new Uint8Array(sigHex.length / 2);
  for (let i = 0; i < sigBytes.length; i++) {
    sigBytes[i] = parseInt(sigHex.substr(i * 2, 2), 16);
  }
  return crypto.subtle.verify("HMAC", key, sigBytes, enc.encode(payload));
}

export function readCookie(req: Request, name: string): string | null {
  const header = req.headers.get("Cookie");
  if (!header) return null;
  for (const part of header.split(";")) {
    const [k, ...rest] = part.trim().split("=");
    if (k === name) return rest.join("=");
  }
  return null;
}

export function setCookieHeader(value: string): string {
  return [
    `${COOKIE_NAME}=${value}`,
    `Max-Age=${COOKIE_TTL_SECONDS}`,
    "Path=/",
    "HttpOnly",
    "Secure",
    "SameSite=Lax",
  ].join("; ");
}

export function clearCookieHeader(): string {
  return `${COOKIE_NAME}=; Max-Age=0; Path=/; HttpOnly; Secure; SameSite=Lax`;
}

export async function constantTimeEqual(
  a: string,
  b: string
): Promise<boolean> {
  // Constant-time string compare via HMAC: identical inputs produce
  // identical macs, otherwise they differ. Robust against length leaks.
  const key = await crypto.subtle.generateKey(
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const macA = await crypto.subtle.sign("HMAC", key, enc.encode(a));
  const macB = await crypto.subtle.sign("HMAC", key, enc.encode(b));
  const arrA = new Uint8Array(macA);
  const arrB = new Uint8Array(macB);
  if (arrA.byteLength !== arrB.byteLength) return false;
  let diff = 0;
  for (let i = 0; i < arrA.byteLength; i++) diff |= arrA[i] ^ arrB[i];
  return diff === 0;
}
