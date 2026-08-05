import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        aila: {
          50: "#f2f5ff",
          100: "#e6ebff",
          200: "#c2cdff",
          300: "#9daeff",
          400: "#7488ff",
          500: "#4f63f5",
          600: "#3c4ad9",
          700: "#2e39ab",
          800: "#242c80",
          900: "#1a2059",
        },
      },
      animation: {
        "pulse-dot": "pulse-dot 1.4s ease-in-out infinite",
      },
      keyframes: {
        "pulse-dot": {
          "0%, 80%, 100%": { opacity: "0.25", transform: "scale(0.85)" },
          "40%": { opacity: "1", transform: "scale(1)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
