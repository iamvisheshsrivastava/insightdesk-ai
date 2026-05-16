// CommonJS format required — PostCSS loads its config via require().
// Use ARRAY syntax with explicit require() calls so Node resolves tailwindcss
// from THIS file's directory (/app/frontend/node_modules), not from inside
// Vite's own node_modules (which is where the object-key lookup fails).
module.exports = {
  plugins: [
    require('tailwindcss'),
    require('autoprefixer'),
  ],
}
