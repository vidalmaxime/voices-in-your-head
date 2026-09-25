import type { Metadata } from "next";
import { Geist_Mono, IM_Fell_English } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";

/** The whole site speaks in one voice: a seventeenth-century book face. */
const fell = IM_Fell_English({
  variable: "--font-fell",
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const siteUrl = new URL("https://maximevidal.com/voices");

const previewImage = {
  url: new URL(`${siteUrl.href}/voices-social-preview.png`),
  width: 2886,
  height: 1376,
  alt: "Voices in your head — painted lettering above a chorus of colorful faces",
};

export const metadata: Metadata = {
  metadataBase: siteUrl,
  openGraph: {
    url: siteUrl,
    images: [previewImage],
  },
  twitter: {
    card: "summary_large_image",
    images: [previewImage],
  },
  title: "Voices in your head",
  description:
    "Speak a thought and hear it continued in your own cloned voice. Speech recognition, AI-generated continuations, and voice synthesis run locally in your browser.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${fell.variable} ${geistMono.variable} antialiased`}
      >
        {children}
        <Analytics
          scriptSrc="/voices/_vercel/insights/script.js"
          viewEndpoint="/voices/_vercel/insights/view"
          eventEndpoint="/voices/_vercel/insights/event"
          sessionEndpoint="/voices/_vercel/insights/session"
        />
      </body>
    </html>
  );
}
