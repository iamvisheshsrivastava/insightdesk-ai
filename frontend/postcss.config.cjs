// CommonJS format required — PostCSS loads config via require(), which is
// incompatible with ESM "export default" when "type":"module" is set in package.json.
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
