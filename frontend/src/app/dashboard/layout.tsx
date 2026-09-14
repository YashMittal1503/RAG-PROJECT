"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import {
  FileText,
  MessageSquare,
  Plus,
  LogOut,
  Trash2,
  PanelLeftClose,
  PanelLeftOpen,
  Pin,
  PinOff,
  MoreHorizontal,
  Pencil,
  Check,
  X,
  ArrowUpDown,
} from "lucide-react";
import {
  getChatSessions,
  getChatMessagesWithMetadata,
  createChatSession,
  deleteChatSession,
  updateChatSession,
  checkBackendHealth,
} from "@/lib/api";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();

  // Sessions state
  const [sessions, setSessions] = useState<any[]>(() => {
    if (typeof window !== "undefined") {
      try {
        const cached = localStorage.getItem("rag_sessions_cache");
        if (cached) return JSON.parse(cached);
      } catch {}
    }
    return [];
  });

  // Pinned session IDs
  const [pinnedSessionIds, setPinnedSessionIds] = useState<string[]>(() => {
    if (typeof window !== "undefined") {
      try {
        const cached = localStorage.getItem("rag_pinned_sessions");
        if (cached) return JSON.parse(cached);
      } catch {}
    }
    return [];
  });

  // Sidebar width (movable / resizable)
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    if (typeof window !== "undefined") {
      try {
        const cached = localStorage.getItem("rag_sidebar_width");
        if (cached) return Math.min(Math.max(Number(cached), 200), 450);
      } catch {}
    }
    return 260;
  });

  // Sidebar collapsed state
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(() => {
    if (typeof window !== "undefined") {
      try {
        const cached = localStorage.getItem("rag_sidebar_open");
        if (cached !== null) return cached === "true";
      } catch {}
    }
    return true;
  });

  const [isResizing, setIsResizing] = useState(false);

  // Active dropdown menu session ID (••• menu)
  const [activeMenuSessionId, setActiveMenuSessionId] = useState<string | null>(null);

  // Inline rename state
  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  // Backend health & delete modal
  const [backendAwake, setBackendAwake] = useState<boolean | null>(true);
  const [sessionToDelete, setSessionToDelete] = useState<{
    id: string;
    title: string;
  } | null>(null);

  // User profile
  const [userProfile, setUserProfile] = useState<{ email: string; name: string; initials: string } | null>(null);

  // Load user profile from Supabase
  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then((res: any) => {
      const user = res?.data?.user;
      if (user) {
        const email = user.email || "";
        const name =
          user.user_metadata?.full_name ||
          user.user_metadata?.name ||
          email.split("@")[0] ||
          "User";
        const parts = name.trim().split(" ");
        let initials = "U";
        if (parts.length >= 2) {
          initials = (parts[0][0] + parts[1][0]).toUpperCase();
        } else if (name.length > 0) {
          initials = name.slice(0, 2).toUpperCase();
        }
        setUserProfile({ email, name, initials });
      }
    });
  }, []);

  // Check backend health
  useEffect(() => {
    let interval: NodeJS.Timeout;

    const check = async () => {
      const health = await checkBackendHealth();
      setBackendAwake(health.awake && health.modelLoaded);

      if (!health.awake || !health.modelLoaded) {
        interval = setInterval(async () => {
          const h = await checkBackendHealth();
          if (h.awake && h.modelLoaded) {
            setBackendAwake(true);
            clearInterval(interval);
          }
        }, 3000);
      }
    };

    check();
    return () => clearInterval(interval);
  }, []);

  // Load chat sessions
  const loadSessions = useCallback(async () => {
    try {
      const data = await getChatSessions();
      setSessions(data);
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("rag_sessions_cache", JSON.stringify(data));
        } catch {}
      }
    } catch {
      // Silently fail if waking up
    }
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Listen for session changes / deletions from other components
  useEffect(() => {
    const onSessionsChanged = () => {
      loadSessions();
      if (typeof window !== "undefined") {
        try {
          const cached = localStorage.getItem("rag_pinned_sessions");
          if (cached) setPinnedSessionIds(JSON.parse(cached));
        } catch {}
      }
    };
    const onSessionDeleted = (e: any) => {
      const deletedId = e.detail?.sessionId;
      if (deletedId) {
        setSessions((prev) => prev.filter((s) => s.id !== deletedId));
        setPinnedSessionIds((prev) => {
          const updated = prev.filter((id) => id !== deletedId);
          if (typeof window !== "undefined") {
            try {
              localStorage.setItem("rag_pinned_sessions", JSON.stringify(updated));
            } catch {}
          }
          return updated;
        });
        if (typeof window !== "undefined") {
          try {
            localStorage.removeItem(`rag_draft_${deletedId}`);
            localStorage.removeItem(`rag_msgs_${deletedId}`);
          } catch {}
        }
      }
    };

    window.addEventListener("chat-sessions-changed", onSessionsChanged);
    window.addEventListener("chat-session-deleted", onSessionDeleted as EventListener);

    return () => {
      window.removeEventListener("chat-sessions-changed", onSessionsChanged);
      window.removeEventListener(
        "chat-session-deleted",
        onSessionDeleted as EventListener
      );
    };
  }, [loadSessions]);

  // Close menus when clicking outside
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest("[data-menu-container]")) {
        setActiveMenuSessionId(null);
      }
    };
    window.addEventListener("click", handleOutsideClick);
    return () => window.removeEventListener("click", handleOutsideClick);
  }, []);

  // Drag resizer for movable sidebar
  const startResizing = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  useEffect(() => {
    if (!isResizing) return;
    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = Math.min(Math.max(e.clientX, 200), 450);
      setSidebarWidth(newWidth);
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("rag_sidebar_width", String(newWidth));
        } catch {}
      }
    };
    const handleMouseUp = () => {
      setIsResizing(false);
    };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing]);

  const toggleSidebar = () => {
    setIsSidebarOpen((prev) => {
      const next = !prev;
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("rag_sidebar_open", String(next));
        } catch {}
      }
      return next;
    });
  };

  // Pin / Unpin handler
  const handleTogglePin = (sessionId: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setActiveMenuSessionId(null);
    setPinnedSessionIds((prev) => {
      const isPinned = prev.includes(sessionId);
      const updated = isPinned
        ? prev.filter((id) => id !== sessionId)
        : [sessionId, ...prev];
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("rag_pinned_sessions", JSON.stringify(updated));
        } catch {}
      }
      return updated;
    });
    window.dispatchEvent(new Event("chat-sessions-changed"));
  };

  // Start inline rename
  const handleStartRename = (session: any, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setActiveMenuSessionId(null);
    setRenamingSessionId(session.id);
    setRenameValue(session.title || "New conversation");
  };

  // Save rename
  const handleSaveRename = async (sessionId: string, e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = renameValue.trim();
    setRenamingSessionId(null);
    if (!trimmed) return;

    // Optimistic update
    setSessions((prev) => {
      const updated = prev.map((s) => (s.id === sessionId ? { ...s, title: trimmed } : s));
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("rag_sessions_cache", JSON.stringify(updated));
        } catch {}
      }
      return updated;
    });

    window.dispatchEvent(new Event("chat-sessions-changed"));

    try {
      await updateChatSession(sessionId, trimmed);
    } catch {
      loadSessions();
    }
  };

  const handleNewChat = async () => {
    const tempId = crypto.randomUUID();
    const tempSession = { id: tempId, title: "New conversation", created_at: new Date().toISOString() };
    setSessions((prev) => [tempSession, ...prev]);
    router.push(`/chat/${tempId}`);

    try {
      const realSession = await createChatSession();
      if (typeof window !== "undefined") {
        try {
          const tempDraft = localStorage.getItem(`rag_draft_${tempId}`);
          if (tempDraft) {
            localStorage.setItem(`rag_draft_${realSession.id}`, tempDraft);
            localStorage.removeItem(`rag_draft_${tempId}`);
          }
        } catch {}
      }
      setSessions((prev) =>
        prev.map((s) => (s.id === tempId ? realSession : s))
      );
      router.replace(`/chat/${realSession.id}`);
    } catch {
      if (typeof window !== "undefined") {
        try {
          localStorage.removeItem(`rag_draft_${tempId}`);
        } catch {}
      }
      setSessions((prev) => prev.filter((s) => s.id !== tempId));
      router.push("/dashboard");
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    setSessions((prev) => {
      const updated = prev.filter((s) => s.id !== sessionId);
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("rag_sessions_cache", JSON.stringify(updated));
          localStorage.removeItem(`rag_draft_${sessionId}`);
          localStorage.removeItem(`rag_msgs_${sessionId}`);
        } catch {}
      }
      return updated;
    });

    setPinnedSessionIds((prev) => {
      const updated = prev.filter((id) => id !== sessionId);
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("rag_pinned_sessions", JSON.stringify(updated));
        } catch {}
      }
      return updated;
    });

    setSessionToDelete(null);

    if (pathname === `/chat/${sessionId}`) {
      router.push("/dashboard");
    }

    window.dispatchEvent(
      new CustomEvent("chat-session-deleted", { detail: { sessionId } })
    );

    try {
      await deleteChatSession(sessionId);
    } catch (err) {
      console.error("Failed to delete chat session:", err);
      loadSessions();
    }
  };

  const handleLogout = async () => {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/login");
    router.refresh();
  };

  // Group sessions into Pinned and Unpinned
  const pinnedSessions = sessions.filter((s) => pinnedSessionIds.includes(s.id));
  const recentSessions = sessions.filter((s) => !pinnedSessionIds.includes(s.id));

  return (
    <div className={`flex h-screen overflow-hidden bg-[var(--background)] text-[var(--foreground)] ${isResizing ? "select-none" : ""}`}>
      {/* Sidebar */}
      <aside
        style={{
          width: isSidebarOpen ? `${sidebarWidth}px` : "0px",
          minWidth: isSidebarOpen ? `${sidebarWidth}px` : "0px",
        }}
        className={`relative flex-shrink-0 bg-[var(--sidebar)] border-r border-[var(--border)] flex flex-col transition-[width] duration-200 ease-out overflow-hidden z-20`}
      >
        {/* Sidebar Header */}
        <div className="h-14 px-3 flex items-center justify-between border-b border-[var(--border)]/60 flex-shrink-0">
          <button
            onClick={() => router.push("/dashboard")}
            className="flex items-center gap-2 px-1.5 py-1 rounded-lg hover:bg-[var(--secondary)] transition-colors text-left"
          >
            <span className="font-serif text-lg font-medium tracking-tight text-[var(--foreground)]">
              DocuChat
            </span>
          </button>

          <button
            onClick={toggleSidebar}
            title="Collapse sidebar"
            aria-label="Collapse sidebar"
            className="p-1.5 rounded-lg text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors cursor-pointer"
          >
            <PanelLeftClose className="w-4 h-4" />
          </button>
        </div>

        {/* Top actions: + New chat, Documents */}
        <div className="p-3 pb-2 space-y-1.5 flex-shrink-0">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center justify-between px-3 py-2 rounded-xl text-sm font-medium border border-[var(--border)] hover:border-neutral-500 bg-[var(--card)]/50 hover:bg-[var(--secondary)] text-[var(--foreground)] transition-all cursor-pointer group shadow-xs"
          >
            <div className="flex items-center gap-2">
              <Plus className="w-4 h-4 text-[var(--primary)] group-hover:scale-110 transition-transform" />
              <span>New chat</span>
            </div>
            <span className="text-[10px] text-[var(--muted-foreground)] border border-[var(--border)] px-1.5 py-0.5 rounded font-mono">
              ⌘K
            </span>
          </button>

          <button
            onClick={() => router.push("/dashboard")}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-sm transition-colors cursor-pointer ${
              pathname === "/dashboard"
                ? "bg-[var(--secondary)] text-[var(--foreground)] font-medium"
                : "text-[var(--muted-foreground)] hover:bg-[var(--secondary)] hover:text-[var(--foreground)]"
            }`}
          >
            <FileText className="w-4 h-4 text-[var(--primary)]" />
            <span>Documents</span>
          </button>
        </div>

        {/* Chat sessions list (Pinned & Recents) */}
        <nav className="flex-1 overflow-y-auto px-2 py-1 space-y-4">
          {/* Pinned Section */}
          {pinnedSessions.length > 0 && (
            <div>
              <div className="px-2.5 py-1.5 flex items-center justify-between text-[11px] font-semibold text-[var(--muted-foreground)] uppercase tracking-wider">
                <div className="flex items-center gap-1.5">
                  <Pin className="w-3 h-3 text-[var(--primary)]" />
                  <span>Pinned</span>
                </div>
                <span className="text-[10px] font-mono text-[var(--muted-foreground)]">
                  {pinnedSessions.length}
                </span>
              </div>

              <div className="space-y-0.5 mt-0.5">
                {pinnedSessions.map((session) => (
                  <SessionItem
                    key={session.id}
                    session={session}
                    isActive={pathname === `/chat/${session.id}`}
                    isPinned={true}
                    isRenaming={renamingSessionId === session.id}
                    renameValue={renameValue}
                    isMenuOpen={activeMenuSessionId === session.id}
                    onSelect={() => router.push(`/chat/${session.id}`)}
                    onPrefetch={() => {
                      router.prefetch(`/chat/${session.id}`);
                      if (typeof window !== "undefined" && !localStorage.getItem(`rag_msgs_${session.id}`)) {
                        getChatMessagesWithMetadata(session.id)
                          .then(({ messages }) => {
                            if (Array.isArray(messages) && messages.length > 0) {
                              localStorage.setItem(`rag_msgs_${session.id}`, JSON.stringify(messages));
                            }
                          })
                          .catch(() => {});
                      }
                    }}
                    onToggleMenu={(e) => {
                      e.stopPropagation();
                      setActiveMenuSessionId((prev) => (prev === session.id ? null : session.id));
                    }}
                    onTogglePin={(e) => handleTogglePin(session.id, e)}
                    onStartRename={(e) => handleStartRename(session, e)}
                    onRenameChange={(val) => setRenameValue(val)}
                    onSaveRename={(e) => handleSaveRename(session.id, e)}
                    onCancelRename={() => setRenamingSessionId(null)}
                    onDeleteRequest={(e) => {
                      e.stopPropagation();
                      setActiveMenuSessionId(null);
                      setSessionToDelete({ id: session.id, title: session.title });
                    }}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Chats and Tasks (Recents) Section */}
          {recentSessions.length > 0 && (
            <div>
              <div className="px-2.5 py-1.5 flex items-center justify-between text-[11px] font-semibold text-[var(--muted-foreground)] uppercase tracking-wider">
                <span>Chats and tasks</span>
                <ArrowUpDown className="w-3 h-3 text-[var(--muted-foreground)] opacity-60" />
              </div>

              <div className="space-y-0.5 mt-0.5">
                {recentSessions.map((session) => (
                  <SessionItem
                    key={session.id}
                    session={session}
                    isActive={pathname === `/chat/${session.id}`}
                    isPinned={false}
                    isRenaming={renamingSessionId === session.id}
                    renameValue={renameValue}
                    isMenuOpen={activeMenuSessionId === session.id}
                    onSelect={() => router.push(`/chat/${session.id}`)}
                    onPrefetch={() => {
                      router.prefetch(`/chat/${session.id}`);
                      if (typeof window !== "undefined" && !localStorage.getItem(`rag_msgs_${session.id}`)) {
                        getChatMessagesWithMetadata(session.id)
                          .then(({ messages }) => {
                            if (Array.isArray(messages) && messages.length > 0) {
                              localStorage.setItem(`rag_msgs_${session.id}`, JSON.stringify(messages));
                            }
                          })
                          .catch(() => {});
                      }
                    }}
                    onToggleMenu={(e) => {
                      e.stopPropagation();
                      setActiveMenuSessionId((prev) => (prev === session.id ? null : session.id));
                    }}
                    onTogglePin={(e) => handleTogglePin(session.id, e)}
                    onStartRename={(e) => handleStartRename(session, e)}
                    onRenameChange={(val) => setRenameValue(val)}
                    onSaveRename={(e) => handleSaveRename(session.id, e)}
                    onCancelRename={() => setRenamingSessionId(null)}
                    onDeleteRequest={(e) => {
                      e.stopPropagation();
                      setActiveMenuSessionId(null);
                      setSessionToDelete({ id: session.id, title: session.title });
                    }}
                  />
                ))}
              </div>
            </div>
          )}
        </nav>

        {/* User Profile Bar at bottom */}
        <div className="p-2 border-t border-[var(--border)] flex-shrink-0 bg-[var(--sidebar)]">
          <div className="flex items-center justify-between p-2 rounded-xl hover:bg-[var(--secondary)] transition-colors group">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-8 h-8 rounded-full bg-[var(--primary)] text-white flex items-center justify-center font-medium text-xs flex-shrink-0 shadow-xs">
                {userProfile?.initials || "YM"}
              </div>
              <div className="truncate">
                <p className="text-xs font-medium text-[var(--foreground)] truncate">
                  {userProfile?.name || "Yash Mittal"}
                </p>
                <p className="text-[10px] text-[var(--muted-foreground)]">Free tier</p>
              </div>
            </div>

            <button
              onClick={handleLogout}
              title="Sign out"
              aria-label="Sign out"
              className="p-1.5 rounded-lg text-[var(--muted-foreground)] hover:text-red-400 hover:bg-red-500/10 transition-colors cursor-pointer"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Movable / Resizable Draggable Right Border */}
        <div
          onMouseDown={startResizing}
          onDoubleClick={() => {
            setSidebarWidth(260);
            localStorage.setItem("rag_sidebar_width", "260");
          }}
          title="Drag to resize sidebar (double-click to reset)"
          className="absolute top-0 right-0 w-1.5 h-full cursor-col-resize hover:bg-[var(--primary)]/60 transition-colors z-30 group"
        >
          <div className="w-full h-full group-hover:bg-[var(--primary)]" />
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col overflow-hidden bg-[var(--background)] min-w-0 relative">
        {/* Floating Sidebar Re-open Button when collapsed */}
        {!isSidebarOpen && (
          <div className="absolute top-3 left-4 z-30">
            <button
              onClick={toggleSidebar}
              title="Open sidebar"
              aria-label="Open sidebar"
              className="p-2 rounded-xl bg-[var(--card)] border border-[var(--border)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--secondary)] shadow-md transition-all cursor-pointer flex items-center gap-1.5"
            >
              <PanelLeftOpen className="w-4 h-4" />
              <span className="text-xs font-serif font-medium pr-1">DocuChat</span>
            </button>
          </div>
        )}

        {/* Backend wake-up banner */}
        {backendAwake === false && (
          <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2.5 flex items-center gap-3 animate-fade-in z-20">
            <div className="flex gap-1">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse-dot" style={{ animationDelay: "0ms" }} />
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse-dot" style={{ animationDelay: "300ms" }} />
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse-dot" style={{ animationDelay: "600ms" }} />
            </div>
            <p className="text-xs text-amber-300">
              The backend is waking up — this can take up to a minute on the free tier. Hang tight!
            </p>
          </div>
        )}

        <div className="flex-1 flex flex-col overflow-hidden">
          {children}
        </div>
      </main>

      {/* Delete Conversation Modal */}
      {sessionToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-xs p-4 animate-fade-in">
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl max-w-sm w-full p-6 shadow-2xl animate-fade-in">
            <div className="w-10 h-10 rounded-xl bg-red-500/10 text-red-400 flex items-center justify-center mb-4">
              <Trash2 className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-[var(--foreground)]">
              Delete conversation?
            </h3>
            <p className="text-sm text-[var(--muted-foreground)] mt-2 leading-relaxed">
              Are you sure you want to delete <span className="font-medium text-[var(--foreground)]">&ldquo;{sessionToDelete.title}&rdquo;</span>? All messages in this conversation will be permanently removed.
            </p>
            <div className="flex justify-end gap-2.5 mt-6">
              <button
                onClick={() => setSessionToDelete(null)}
                className="px-3.5 py-2 rounded-xl text-sm font-medium border border-[var(--border)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDeleteSession(sessionToDelete.id)}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-sm font-medium bg-red-600 hover:bg-red-500 text-white shadow-sm transition-colors cursor-pointer"
              >
                <span>Delete</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Session Item Component with Hover Options & Inline Rename ───────────────

interface SessionItemProps {
  session: any;
  isActive: boolean;
  isPinned: boolean;
  isRenaming: boolean;
  renameValue: string;
  isMenuOpen: boolean;
  onSelect: () => void;
  onPrefetch: () => void;
  onToggleMenu: (e: React.MouseEvent) => void;
  onTogglePin: (e: React.MouseEvent) => void;
  onStartRename: (e: React.MouseEvent) => void;
  onRenameChange: (val: string) => void;
  onSaveRename: (e: React.FormEvent) => void;
  onCancelRename: () => void;
  onDeleteRequest: (e: React.MouseEvent) => void;
}

function SessionItem({
  session,
  isActive,
  isPinned,
  isRenaming,
  renameValue,
  isMenuOpen,
  onSelect,
  onPrefetch,
  onToggleMenu,
  onTogglePin,
  onStartRename,
  onRenameChange,
  onSaveRename,
  onCancelRename,
  onDeleteRequest,
}: SessionItemProps) {
  if (isRenaming) {
    return (
      <form
        onSubmit={onSaveRename}
        className="flex items-center gap-1 px-2 py-1 bg-[var(--secondary)] rounded-xl border border-[var(--primary)]"
      >
        <input
          type="text"
          value={renameValue}
          onChange={(e) => onRenameChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") onCancelRename();
          }}
          autoFocus
          className="flex-1 bg-transparent text-xs text-[var(--foreground)] outline-none min-w-0"
        />
        <button
          type="submit"
          title="Save title"
          className="p-1 text-emerald-400 hover:text-emerald-300 transition-colors cursor-pointer"
        >
          <Check className="w-3.5 h-3.5" />
        </button>
        <button
          type="button"
          onClick={onCancelRename}
          title="Cancel"
          className="p-1 text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors cursor-pointer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </form>
    );
  }

  return (
    <div
      data-menu-container
      className={`group relative flex items-center rounded-xl text-xs transition-all ${
        isActive
          ? "bg-[var(--secondary)] text-[var(--foreground)] font-medium shadow-xs"
          : "text-[var(--muted-foreground)] hover:bg-[var(--secondary)]/60 hover:text-[var(--foreground)]"
      }`}
    >
      <button
        onClick={onSelect}
        onMouseEnter={onPrefetch}
        className="flex-1 flex items-center gap-2.5 px-3 py-2 text-left truncate min-w-0 cursor-pointer"
      >
        {isPinned ? (
          <Pin className="w-3.5 h-3.5 flex-shrink-0 text-[var(--primary)]" />
        ) : (
          <MessageSquare className="w-3.5 h-3.5 flex-shrink-0 opacity-70" />
        )}
        <span className="truncate">{session.title || "New conversation"}</span>
      </button>

      {/* Action menu trigger (•••) */}
      <div className="relative flex-shrink-0 pr-1">
        <button
          onClick={onToggleMenu}
          title="Chat options"
          aria-label="Chat options"
          className={`p-1.5 rounded-lg text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--card)] transition-all cursor-pointer ${
            isMenuOpen ? "opacity-100 bg-[var(--card)] text-[var(--foreground)]" : "opacity-0 group-hover:opacity-100"
          }`}
        >
          <MoreHorizontal className="w-3.5 h-3.5" />
        </button>

        {/* Options Dropdown Menu */}
        {isMenuOpen && (
          <div className="absolute right-0 top-full mt-1 w-40 bg-[var(--card)] border border-[var(--border)] rounded-xl shadow-xl py-1 z-50 animate-fade-in">
            <button
              onClick={onTogglePin}
              className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors text-left cursor-pointer"
            >
              {isPinned ? (
                <>
                  <PinOff className="w-3.5 h-3.5 text-[var(--muted-foreground)]" />
                  <span>Unpin</span>
                </>
              ) : (
                <>
                  <Pin className="w-3.5 h-3.5 text-[var(--primary)]" />
                  <span>Pin to top</span>
                </>
              )}
            </button>

            <button
              onClick={onStartRename}
              className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors text-left cursor-pointer"
            >
              <Pencil className="w-3.5 h-3.5 text-[var(--muted-foreground)]" />
              <span>Rename</span>
            </button>

            <div className="my-1 border-t border-[var(--border)]/60" />

            <button
              onClick={onDeleteRequest}
              className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-red-400 hover:bg-red-500/10 transition-colors text-left cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Delete</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
