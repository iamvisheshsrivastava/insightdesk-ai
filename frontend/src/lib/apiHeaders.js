// Optional shared API-key header for calls to state-mutating/expensive
// endpoints (see backend src/api/main.py `require_api_key`, and issue #15).
//
// IMPORTANT: this is NOT a real security boundary. VITE_API_KEY is a
// build-time value baked into the static JS bundle shipped to every visitor,
// so anyone can read it out of the deployed site's network/source tab. It's
// a light deterrent to keep casual bots/scrapers from hammering an
// unauthenticated public demo API with expensive requests — not a way to
// keep the API private or to distinguish real users. If this project ever
// needs real security (private data, per-user quotas, billing, etc.), it
// should be replaced with proper per-user authentication instead of relying
// on this shared key.
//
// If VITE_API_KEY isn't set at build time (the default, and the current
// Render deployment), this is a no-op and requests go out exactly as before.
const API_KEY = import.meta.env.VITE_API_KEY

export function apiHeaders(extra = {}) {
  return API_KEY
    ? { ...extra, 'X-API-Key': API_KEY }
    : extra
}
