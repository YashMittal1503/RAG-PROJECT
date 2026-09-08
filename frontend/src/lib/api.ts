/**
 * Backend API client with authentication, retry logic, and wake-up detection.
 *
 * All requests automatically:
 * 1. Include the Supabase JWT in the Authorization header
 * 2. Prepend the backend URL
 * 3. Handle the "waking up" state (Render free tier cold starts)
 */

import { createClient } from "@/lib/supabase/client";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

/**
 * Get the current user's JWT token from Supabase.
 *
 * Deduplicates concurrent calls: when multiple API requests fire
 * in parallel on page load, they share a single getSession() call
 * instead of each triggering their own (which was the main cause
 * of slow initial load).
 */
let _tokenPromise: Promise<string | null> | null = null;
let _tokenTimestamp = 0;

async function getToken(): Promise<string | null> {
  const now = Date.now();

  // Reuse in-flight or recently-fetched token (cache for 60 seconds)
  if (_tokenPromise && now - _tokenTimestamp < 60_000) {
    return _tokenPromise;
  }

  _tokenTimestamp = now;
  _tokenPromise = (async () => {
    try {
      const supabase = createClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();
      const token = session?.access_token || null;
      if (!token) {
        _tokenPromise = null;
      }
      return token;
    } catch {
      _tokenPromise = null;
      return null;
    }
  })();

  return _tokenPromise;
}

/**
 * Make an authenticated API request to the backend.
 */
export async function apiRequest(
  path: string,
  options: RequestInit = {}
): Promise<Response> {
  const token = await getToken();

  const headers: HeadersInit = {
    ...((options.headers as Record<string, string>) || {}),
  };

  if (token) {
    (headers as Record<string, string>)["Authorization"] = `Bearer ${token}`;
  }

  // Don't set Content-Type for FormData (browser sets it with boundary)
  if (!(options.body instanceof FormData)) {
    (headers as Record<string, string>)["Content-Type"] = "application/json";
  }

  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers,
  });

  return response;
}

/**
 * Check if the backend is awake and ready.
 * Returns { awake: boolean, modelLoaded: boolean }
 */
export async function checkBackendHealth(): Promise<{
  awake: boolean;
  modelLoaded: boolean;
}> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);

    const response = await fetch(`${API_URL}/api/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (response.ok) {
      const data = await response.json();
      return { awake: true, modelLoaded: data.model_loaded };
    }
    return { awake: false, modelLoaded: false };
  } catch {
    return { awake: false, modelLoaded: false };
  }
}

/**
 * Upload files to the backend.
 */
export async function uploadFiles(files: File[]): Promise<{
  documents: any[];
  errors: any[];
}> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await apiRequest("/api/documents/upload", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = `Upload failed (${response.status})`;
    try {
      const error = await response.json();
      if (typeof error.detail === "string") {
        message = error.detail;
      } else if (Array.isArray(error.detail)) {
        message = error.detail.map((d: any) => d.msg || JSON.stringify(d)).join(", ");
      }
    } catch {
      const text = await response.text().catch(() => "");
      if (text) message = text;
    }
    throw new Error(message);
  }

  return response.json();
}

/**
 * Get the list of documents for the current user.
 */
export async function getDocuments(): Promise<any[]> {
  const response = await apiRequest("/api/documents");
  if (!response.ok) throw new Error("Failed to fetch documents");
  return response.json();
}

/**
 * Get the status of a specific document.
 */
export async function getDocumentStatus(
  docId: string
): Promise<{ id: string; status: string; failure_reason?: string; chunk_count?: number }> {
  const response = await apiRequest(`/api/documents/${docId}/status`);
  if (!response.ok) throw new Error("Failed to fetch document status");
  return response.json();
}

/**
 * Delete a document.
 */
export async function deleteDocument(docId: string): Promise<void> {
  const response = await apiRequest(`/api/documents/${docId}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error("Failed to delete document");
}

/**
 * Create a new chat session.
 */
export async function createChatSession(
  title: string = "New conversation"
): Promise<any> {
  const response = await apiRequest("/api/chat/sessions", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
  if (!response.ok) throw new Error("Failed to create chat session");
  return response.json();
}

/**
 * Get all chat sessions.
 */
export async function getChatSessions(): Promise<any[]> {
  const response = await apiRequest("/api/chat/sessions");
  if (!response.ok) throw new Error("Failed to fetch chat sessions");
  return response.json();
}

/**
 * Get a single chat session.
 */
export async function getChatSession(sessionId: string): Promise<any> {
  const response = await apiRequest(`/api/chat/sessions/${sessionId}`);
  if (!response.ok) throw new Error("Failed to fetch chat session");
  return response.json();
}

/**
 * Delete a chat session.
 */
export async function deleteChatSession(sessionId: string): Promise<void> {
  const response = await apiRequest(`/api/chat/sessions/${sessionId}`, {
    method: "DELETE",
  });
  if (!response.ok && response.status !== 204) {
    throw new Error("Failed to delete chat session");
  }
}

/**
 * Get messages and session title for a chat session in a single call.
 */
export async function getChatMessagesWithMetadata(
  sessionId: string
): Promise<{ messages: any[]; title?: string }> {
  const response = await apiRequest(
    `/api/chat/sessions/${sessionId}/messages`
  );
  if (!response.ok) throw new Error("Failed to fetch messages");
  const messages = await response.json();
  const title = response.headers.get("x-session-title") || undefined;
  return { messages, title };
}

/**
 * Get messages for a chat session.
 */
export async function getChatMessages(sessionId: string): Promise<any[]> {
  const { messages } = await getChatMessagesWithMetadata(sessionId);
  return messages;
}

/**
 * Send a query and return the SSE event source URL + token.
 * The caller handles the SSE stream directly.
 */
export async function sendQuery(
  sessionId: string,
  question: string
): Promise<Response> {
  const response = await apiRequest(
    `/api/chat/sessions/${sessionId}/query`,
    {
      method: "POST",
      body: JSON.stringify({ question }),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Query failed");
  }

  return response;
}
