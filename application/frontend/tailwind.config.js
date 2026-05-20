/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#1f1d1a",
        bg: "#faf9f6",
        panel: "#ffffff",
        border: "#e5e3df",
        muted: "#6c6864",
        accent: "#cc785c",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui"],
      },
    },
  },
  plugins: [],
};
