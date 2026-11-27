# DocTalk — Frontend

Next.js 16 (App Router, React 19, TypeScript, Tailwind CSS v4) client for DocTalk. See the [root README](../README.md) for the full system.

## What it does

- **Auth** with Supabase (`/login`, `/signup`); `src/proxy.ts` guards private routes and checks the session cookie locally before calling Supabase.
- **Document dashboard** (`/dashboard`): drag-and-drop multi-file upload, live ingestion status polling (queued → parsing → chunking → embedding → ready), retry for failed files, delete.
- **Streaming chat** (`/chat/[sessionId]`): reads the backend's Server-Sent Events stream token by token, renders Markdown answers (GFM tables), page/row citations, the executed SQL for spreadsheet answers, and live chat titles.
- **Session sidebar**: rename, pin, delete; a backend wake-up check (`/api/health`) for hosts that sleep when idle.

## Run locally

```bash
npm ci
cp .env.example .env.local   # NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY, NEXT_PUBLIC_API_URL
npm run dev                  # http://localhost:3000
```

`npm run lint` and `npm run build` run in CI on every push and pull request.
