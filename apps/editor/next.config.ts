import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // Allow the public demo host to reach dev resources (Next 15 blocks cross-origin
  // dev access by default). Extra hosts via NEXT_ALLOWED_DEV_ORIGINS (comma-sep).
  allowedDevOrigins: ['34.80.144.112', ...(process.env.NEXT_ALLOWED_DEV_ORIGINS?.split(',') ?? [])],
  logging: {
    browserToTerminal: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  transpilePackages: [
    'three',
    '@pascal-app/viewer',
    '@pascal-app/core',
    '@pascal-app/editor',
    '@pascal-app/ifc-converter',
    '@pascal-app/mcp',
  ],
  turbopack: {
    resolveAlias: {
      react: './node_modules/react',
      three: './node_modules/three',
      '@react-three/fiber': './node_modules/@react-three/fiber',
      '@react-three/drei': './node_modules/@react-three/drei',
    },
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '100mb',
    },
  },
  images: {
    unoptimized: process.env.NEXT_PUBLIC_ASSETS_CDN_URL?.startsWith('http://localhost') ?? false,
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
      {
        protocol: 'http',
        hostname: '**',
      },
    ],
  },
}

export default nextConfig
