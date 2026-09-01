// Flat config (ESLint v9+) for the Node/ESM installer script under setup/.
export default [
  {
    files: ["setup/**/*.mjs"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: {
        process: "readonly",
        console: "readonly",
        fetch: "readonly",
        AbortSignal: "readonly",
        URL: "readonly",
        Buffer: "readonly",
        TextDecoder: "readonly",
        TextEncoder: "readonly",
      },
    },
    rules: {
      "no-unused-vars": "warn",
      "no-undef": "error",
    },
  },
];
