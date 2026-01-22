"use client";

import dynamic from "next/dynamic";

const LLMPage = dynamic(() => import("./llm/LLMPage"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-screen">
      <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
    </div>
  ),
});

export default function Home() {
  return <LLMPage />;
}
