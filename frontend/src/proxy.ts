import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";

/**
 * Fast-path check: inspects Supabase auth cookie for an unexpired JWT.
 * Returns true if valid, false if expired, or null if no auth cookie exists.
 */
function checkLocalSession(request: NextRequest): { authenticated: boolean; needsRefresh: boolean } {
  const authCookies = request.cookies
    .getAll()
    .filter((c) => c.name.startsWith("sb-") && c.name.includes("-auth-token"))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));

  if (authCookies.length === 0) {
    return { authenticated: false, needsRefresh: false };
  }

  try {
    let raw = authCookies.map((c) => c.value).join("");
    if (raw.startsWith("base64-")) {
      raw = Buffer.from(raw.slice(7), "base64url").toString("utf-8");
    }
    const data = JSON.parse(raw);
    const token = Array.isArray(data) ? data[0] : data?.access_token;
    if (typeof token === "string" && token.includes(".")) {
      const payloadPart = token.split(".")[1];
      const payload = JSON.parse(Buffer.from(payloadPart, "base64url").toString("utf-8"));
      // If token is valid for at least another 60 seconds, no remote call needed
      if (payload.exp && payload.exp * 1000 > Date.now() + 60_000) {
        return { authenticated: true, needsRefresh: false };
      }
      // Token is near expiry or expired, needs Supabase refresh
      return { authenticated: false, needsRefresh: true };
    }
  } catch {
    // If parsing fails, fall back to remote refresh
    return { authenticated: false, needsRefresh: true };
  }

  return { authenticated: false, needsRefresh: false };
}

export async function proxy(request: NextRequest) {
  // Pass /api requests straight through to FastAPI backend via Next.js rewrites.
  if (request.nextUrl.pathname.startsWith("/api")) {
    return NextResponse.next();
  }

  const { authenticated, needsRefresh } = checkLocalSession(request);

  // Protected routes — /dashboard, /chat
  const isProtectedRoute =
    request.nextUrl.pathname.startsWith("/dashboard") ||
    request.nextUrl.pathname.startsWith("/chat");

  // Auth routes — /login, /signup
  const isAuthRoute =
    request.nextUrl.pathname === "/login" ||
    request.nextUrl.pathname === "/signup";

  const isRoot = request.nextUrl.pathname === "/";

  // 1. FAST PATH: Authenticated with unexpired token (< 0.2ms, 0 network calls)
  if (authenticated) {
    if (isAuthRoute || isRoot) {
      const url = request.nextUrl.clone();
      url.pathname = "/dashboard";
      return NextResponse.redirect(url);
    }
    return NextResponse.next();
  }

  // 2. FAST PATH: Unauthenticated and no refresh needed (< 0.1ms, 0 network calls)
  if (!needsRefresh) {
    if (isProtectedRoute || isRoot) {
      const url = request.nextUrl.clone();
      url.pathname = "/login";
      return NextResponse.redirect(url);
    }
    return NextResponse.next();
  }

  // 3. SLOW PATH (only when token expired / needs refresh): contact Supabase Auth
  let supabaseResponse = NextResponse.next({ request });

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value)
          );
          supabaseResponse = NextResponse.next({ request });
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options)
          );
        },
      },
    }
  );

  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (isProtectedRoute && !user) {
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    return NextResponse.redirect(url);
  }

  if (isAuthRoute && user) {
    const url = request.nextUrl.clone();
    url.pathname = "/dashboard";
    return NextResponse.redirect(url);
  }

  if (isRoot) {
    const url = request.nextUrl.clone();
    url.pathname = user ? "/dashboard" : "/login";
    return NextResponse.redirect(url);
  }

  return supabaseResponse;
}

export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - font files and image files
     */
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|woff|woff2|ttf|eot)$).*)",
  ],
};

