
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
} from "react";

import "./App.css";

import {
  enrollFace,
  getBackendStatus,
  readFile,
  sendAssistantMessage,
  speakText,
  transcribeAudio,
  type BackendStatus,
  type ConversationMessage,
  type Detection,
  type Face,
} from "./services/api";

import Header from "./components/Header";
import ChatPanel from "./components/ChatPanel";
import CameraPanel from "./components/CameraPanel";
import FileBar from "./components/FileBar";
import VoiceStatus from "./components/VoiceStatus";
import EnrollModal from "./components/EnrollModal";


/* =========================================================
   TYPES
========================================================= */

export type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
};

export type CameraViewMode =
  | "compact"
  | "normal"
  | "large";

type BackendConnection =
  | "checking"
  | "online"
  | "offline";

type StoredConversation = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
};


/* =========================================================
   STORAGE
========================================================= */

const CONVERSATIONS_STORAGE_KEY =
  "fast-ai-conversations-v1";

const CURRENT_CONVERSATION_KEY =
  "fast-ai-current-conversation-v1";

const DEFAULT_MESSAGE: Message = {
  id: 1,
  role: "assistant",
  content:
    "Hello! I am your Fast AI Assistant. Ask me anything.",
};


/* =========================================================
   CONVERSATION HELPERS
========================================================= */

function createConversation(): StoredConversation {
  const now = Date.now();

  return {
    id:
      `${now}-${Math.random()
        .toString(36)
        .slice(2, 8)}`,

    title:
      "New Conversation",

    createdAt:
      now,

    updatedAt:
      now,

    messages: [
      {
        ...DEFAULT_MESSAGE,
        id: now,
      },
    ],
  };
}


function loadConversations(): StoredConversation[] {
  try {
    const raw =
      localStorage.getItem(
        CONVERSATIONS_STORAGE_KEY
      );

    if (!raw) {
      return [];
    }

    const parsed =
      JSON.parse(raw);

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.filter(
      (
        item: unknown
      ): item is StoredConversation => {
        if (
          !item ||
          typeof item !== "object"
        ) {
          return false;
        }

        const conversation =
          item as Partial<StoredConversation>;

        return (
          typeof conversation.id === "string" &&
          typeof conversation.title === "string" &&
          typeof conversation.createdAt === "number" &&
          typeof conversation.updatedAt === "number" &&
          Array.isArray(conversation.messages)
        );
      }
    );
  } catch (error) {
    console.error(
      "Conversation load error:",
      error
    );

    return [];
  }
}


function saveConversations(
  conversations: StoredConversation[]
) {
  try {
    localStorage.setItem(
      CONVERSATIONS_STORAGE_KEY,
      JSON.stringify(
        conversations
      )
    );
  } catch (error) {
    console.error(
      "Conversation save error:",
      error
    );
  }
}


const MEMORY_CANDIDATE_PATTERNS = [
  /\bmy name is\b/i,
  /\bcall me\b/i,
  /\bmy age is\b/i,
  /\b(?:i am|i'm|i’m)\s+(?:(?:currently|aged)\s+)?\d{1,3}\b/i,
  /\bmy favou?rite\b/i,
  /\bi (?:prefer|like|love|enjoy|dislike|hate)\b/i,
  /\b(?:remember that|remember this|please remember|keep in mind that)\b/i,
  /\b(?:i study|i am studying|i'm studying|i live in|i'm from|i am from|i work as)\b/i,
];

function collectMemoryHistory(
  conversations: StoredConversation[],
  currentConversationId: string,
  currentMessages: Message[]
): string[] {
  const allUserMessages = conversations
    .map((conversation) => ({
      ...conversation,
      messages:
        conversation.id === currentConversationId
          ? currentMessages
          : conversation.messages,
    }))
    .flatMap((conversation) =>
      conversation.messages
        .filter(
          (message) =>
            message.role === "user" &&
            message.content.trim()
        )
        .map((message) => ({
          id: message.id,
          content: message.content.trim(),
        }))
    )
    .sort((a, b) => a.id - b.id);

  return allUserMessages
    .filter((message) =>
      MEMORY_CANDIDATE_PATTERNS.some((pattern) =>
        pattern.test(message.content)
      )
    )
    .map((message) => message.content.slice(0, 1000))
    .slice(-300);
}


function makeConversationTitle(
  messages: Message[]
): string {
  const firstUserMessage =
    messages.find(
      (message) =>
        message.role === "user" &&
        message.content.trim()
    );

  if (!firstUserMessage) {
    return "New Conversation";
  }

  const cleaned =
    firstUserMessage.content
      .replace(/\s+/g, " ")
      .trim();

  if (cleaned.length <= 32) {
    return cleaned;
  }

  return (
    cleaned
      .slice(0, 32)
      .trimEnd() +
    "..."
  );
}


function getInitialConversation(
  savedConversations: StoredConversation[]
): StoredConversation {
  if (
    savedConversations.length === 0
  ) {
    return createConversation();
  }

  let savedId: string | null = null;

  try {
    savedId =
      localStorage.getItem(
        CURRENT_CONVERSATION_KEY
      );
  } catch {
    savedId = null;
  }

  const savedConversation =
    savedConversations.find(
      (conversation) =>
        conversation.id === savedId
    );

  if (savedConversation) {
    return savedConversation;
  }

  const sorted =
    savedConversations
      .slice()
      .sort(
        (a, b) =>
          b.updatedAt -
          a.updatedAt
      );

  return sorted[0];
}


/* =========================================================
   APP
========================================================= */

function App() {

  /* =======================================================
     CONVERSATIONS
  ======================================================= */

  const [conversations, setConversations] =
    useState<StoredConversation[]>(() => {
      const saved =
        loadConversations();

      if (saved.length > 0) {
        return saved;
      }

      const fresh =
        createConversation();

      saveConversations([
        fresh,
      ]);

      return [
        fresh,
      ];
    });


  const [currentConversationId, setCurrentConversationId] =
    useState<string>(() => {
      const saved =
        loadConversations();

      if (saved.length === 0) {
        return "";
      }

      try {
        const savedId =
          localStorage.getItem(
            CURRENT_CONVERSATION_KEY
          );

        if (
          savedId &&
          saved.some(
            (conversation) =>
              conversation.id ===
              savedId
          )
        ) {
          return savedId;
        }
      } catch {
        /* Ignore storage errors. */
      }

      return getInitialConversation(
        saved
      ).id;
    });


  const [historyPanelOpen, setHistoryPanelOpen] =
    useState(false);


  /* =======================================================
     CURRENT CHAT
  ======================================================= */

  const initialConversation =
    getInitialConversation(
      conversations
    );


  const [messages, setMessages] =
    useState<Message[]>(
      initialConversation.messages
    );


  const [messageInput, setMessageInput] =
    useState("");


  const [isThinking, setIsThinking] =
    useState(false);


  /* =======================================================
     BACKEND
  ======================================================= */

  const [backendConnection, setBackendConnection] =
    useState<BackendConnection>(
      "checking"
    );


  const [backendStatus, setBackendStatus] =
    useState<BackendStatus | null>(
      null
    );


  /* =======================================================
     CAMERA
  ======================================================= */

  const [cameraRunning, setCameraRunning] =
    useState(false);


  const [cameraStatus, setCameraStatus] =
    useState(
      "Camera is off"
    );


  const [detections, setDetections] =
    useState<Detection[]>([]);


  const [faces, setFaces] =
    useState<Face[]>([]);


  const [detectionFPS, setDetectionFPS] =
    useState(0);


  const [cameraViewMode, setCameraViewMode] =
    useState<CameraViewMode>(
      "normal"
    );


  /* =======================================================
     VOICE
  ======================================================= */

  const [isRecording, setIsRecording] =
    useState(false);


  const [voiceStatus, setVoiceStatus] =
    useState("");


  const [isTranscribing, setIsTranscribing] =
    useState(false);


  const [isConversationMode, setIsConversationMode] =
    useState(false);


  const [isWakeWordActive, setIsWakeWordActive] =
    useState(false);


  /* =======================================================
     TTS
  ======================================================= */

  const [speakingMessageId, setSpeakingMessageId] =
    useState<number | null>(
      null
    );


  /* =======================================================
     FILE
  ======================================================= */

  const [fileName, setFileName] =
    useState("");


  const [fileContext, setFileContext] =
    useState("");


  const [fileCharacters, setFileCharacters] =
    useState(0);


  const [fileLoading, setFileLoading] =
    useState(false);


  /* =======================================================
     ENROLLMENT
  ======================================================= */

  const [enrollPanelOpen, setEnrollPanelOpen] =
    useState(false);


  const [enrollName, setEnrollName] =
    useState("");


  const [enrollStatus, setEnrollStatus] =
    useState("");


  const [isEnrolling, setIsEnrolling] =
    useState(false);


  /* =======================================================
     REFS
  ======================================================= */

  const mediaRecorderRef =
    useRef<MediaRecorder | null>(
      null
    );


  const audioChunksRef =
    useRef<Blob[]>([]);


  const currentAudioRef =
    useRef<HTMLAudioElement | null>(
      null
    );


  const fileInputRef =
    useRef<HTMLInputElement | null>(
      null
    );


  const requestControllerRef =
    useRef<AbortController | null>(
      null
    );


  const conversationModeRef =
    useRef(false);


  const wakeWordActiveRef =
    useRef(false);


  const wakeSocketRef =
    useRef<WebSocket | null>(null);


  const wakeMediaStreamRef =
    useRef<MediaStream | null>(null);


  const wakeAudioContextRef =
    useRef<AudioContext | null>(null);


  const wakeSourceRef =
    useRef<MediaStreamAudioSourceNode | null>(null);


  const wakeProcessorRef =
    useRef<ScriptProcessorNode | null>(null);


  const handleSendRef =
    useRef<((voiceText?: string, speakResponse?: boolean) => Promise<void>) | null>(null);


  /* =======================================================
     CURRENT CONVERSATION FIX
  ======================================================= */

  useEffect(() => {
    const exists =
      conversations.some(
        (conversation) =>
          conversation.id ===
          currentConversationId
      );

    if (
      !currentConversationId ||
      !exists
    ) {
      setCurrentConversationId(
        getInitialConversation(
          conversations
        ).id
      );
    }
  }, [
    conversations,
    currentConversationId,
  ]);


  /* =======================================================
     SAVE CURRENT CONVERSATION
  ======================================================= */

  useEffect(() => {
    if (!currentConversationId) {
      return;
    }

    setConversations(
      (current) => {
        const exists =
          current.some(
            (conversation) =>
              conversation.id ===
              currentConversationId
          );

        if (!exists) {
          return current;
        }

        const updated =
          current.map(
            (conversation) => {
              if (
                conversation.id !==
                currentConversationId
              ) {
                return conversation;
              }

              return {
                ...conversation,

                title:
                  makeConversationTitle(
                    messages
                  ),

                updatedAt:
                  Date.now(),

                messages:
                  messages.map(
                    (message) => ({
                      ...message,
                    })
                  ),
              };
            }
          );

        saveConversations(
          updated
        );

        return updated;
      }
    );
  }, [
    messages,
    currentConversationId,
  ]);


  /* =======================================================
     SAVE CURRENT CONVERSATION ID
  ======================================================= */

  useEffect(() => {
    if (!currentConversationId) {
      return;
    }

    try {
      localStorage.setItem(
        CURRENT_CONVERSATION_KEY,
        currentConversationId
      );
    } catch (error) {
      console.error(
        "Current conversation save error:",
        error
      );
    }
  }, [
    currentConversationId,
  ]);


  /* =======================================================
     BACKEND STATUS
  ======================================================= */

  const checkBackend =
    useCallback(
      async () => {
        setBackendConnection(
          "checking"
        );

        try {
          const data =
            await getBackendStatus();

          setBackendStatus(
            data
          );

          setBackendConnection(
            "online"
          );

        } catch (error) {
          console.error(
            "Backend status error:",
            error
          );

          setBackendStatus(
            null
          );

          setBackendConnection(
            "offline"
          );
        }
      },
      []
    );


  useEffect(() => {
    void checkBackend();

    const timer =
      window.setInterval(
        () => {
          void checkBackend();
        },
        10000
      );

    return () => {
      window.clearInterval(
        timer
      );
    };
  }, [
    checkBackend,
  ]);


  /* =======================================================
     NEW CHAT
  ======================================================= */

  function createNewChat() {
    if (isThinking) {
      requestControllerRef.current?.abort();
    }

    const conversation =
      createConversation();

    setConversations(
      (current) => {
        const updated = [
          conversation,
          ...current,
        ];

        saveConversations(
          updated
        );

        return updated;
      }
    );

    setCurrentConversationId(
      conversation.id
    );

    setMessages(
      conversation.messages
    );

    setMessageInput("");

    setFileName("");

    setFileContext("");

    setFileCharacters(0);

    setHistoryPanelOpen(
      false
    );
  }


  /* =======================================================
     OPEN SAVED CHAT
  ======================================================= */

  function openConversation(
    conversationId: string
  ) {
    if (isThinking) {
      requestControllerRef.current?.abort();

      setIsThinking(
        false
      );
    }

    const conversation =
      conversations.find(
        (item) =>
          item.id ===
          conversationId
      );

    if (!conversation) {
      return;
    }

    setCurrentConversationId(
      conversation.id
    );

    setMessages(
      conversation.messages
    );

    setMessageInput("");

    setFileName("");

    setFileContext("");

    setFileCharacters(0);

    setHistoryPanelOpen(
      false
    );
  }


  /* =======================================================
     DELETE CHAT
  ======================================================= */

  function deleteConversation(
    conversationId: string
  ) {
    if (
      isThinking &&
      conversationId ===
        currentConversationId
    ) {
      requestControllerRef.current?.abort();

      setIsThinking(
        false
      );
    }

    const remaining =
      conversations.filter(
        (conversation) =>
          conversation.id !==
          conversationId
      );

    if (
      remaining.length ===
      0
    ) {
      const fresh =
        createConversation();

      const updated = [
        fresh,
      ];

      saveConversations(
        updated
      );

      setConversations(
        updated
      );

      setCurrentConversationId(
        fresh.id
      );

      setMessages(
        fresh.messages
      );

      setMessageInput("");

      setFileName("");

      setFileContext("");

      setFileCharacters(0);

      return;
    }

    saveConversations(
      remaining
    );

    setConversations(
      remaining
    );

    if (
      currentConversationId ===
      conversationId
    ) {
      const sorted =
        remaining
          .slice()
          .sort(
            (a, b) =>
              b.updatedAt -
              a.updatedAt
          );

      const next =
        sorted[0];

      setCurrentConversationId(
        next.id
      );

      setMessages(
        next.messages
      );

      setMessageInput("");

      setFileName("");

      setFileContext("");

      setFileCharacters(0);
    }
  }


  /* =======================================================
     CLEAR CURRENT CHAT
  ======================================================= */

  function clearCurrentChat() {
    if (isThinking) {
      requestControllerRef.current?.abort();

      setIsThinking(
        false
      );
    }

    const freshMessage: Message = {
      id:
        Date.now(),

      role:
        "assistant",

      content:
        "New conversation started. How can I help?",
    };

    setMessages([
      freshMessage,
    ]);

    setMessageInput("");

    setFileName("");

    setFileContext("");

    setFileCharacters(0);

    setConversations(
      (current) => {
        const updated =
          current.map(
            (conversation) => {
              if (
                conversation.id !==
                currentConversationId
              ) {
                return conversation;
              }

              return {
                ...conversation,

                title:
                  "New Conversation",

                updatedAt:
                  Date.now(),

                messages: [
                  freshMessage,
                ],
              };
            }
          );

        saveConversations(
          updated
        );

        return updated;
      }
    );
  }


  /* =======================================================
     SEND MESSAGE
  ======================================================= */

  async function handleSend(
    voiceText?: string,
    speakResponse = false
  ) {
    const text =
      (voiceText ?? messageInput).trim();

    if (
      !text ||
      isThinking ||
      fileLoading ||
      (isTranscribing && !voiceText)
    ) {
      return;
    }

    if (
      backendConnection !==
      "online"
    ) {
      setMessages(
        (current) => [
          ...current,
          {
            id:
              Date.now(),

            role:
              "assistant",

            content:
              "❌ The FastAPI backend is offline. Start the backend and try again.",
          },
        ]
      );

      if (speakResponse) {
        setVoiceStatus("❌ Assistant backend is offline");
        if (conversationModeRef.current) {
          resumeConversationListening();
        }
      }

      return;
    }

    const history:
      ConversationMessage[] =
      messages
        .filter(
          (message) =>
            message.content.trim()
        )
        .slice(-12)
        .map(
          (message) => ({
            role:
              message.role,

            content:
              message.content,
          })
        );

    // Recover explicit personal facts from saved chats. This is sent only
    // to the backend memory detector, not included in Ollama's prompt history.
    const memoryHistory = collectMemoryHistory(
      conversations,
      currentConversationId,
      messages
    );

    const userMessage: Message = {
      id:
        Date.now(),

      role:
        "user",

      content:
        text,
    };

    const thinkingId =
      Date.now() + 1;

    setMessages(
      (current) => [
        ...current,

        userMessage,

        {
          id:
            thinkingId,

          role:
            "assistant",

          content:
            "",
        },
      ]
    );

    setMessageInput("");

    setIsThinking(
      true
    );

    const controller =
      new AbortController();

    requestControllerRef.current =
      controller;

    try {
      const response =
        await sendAssistantMessage(
          text,

          {
            detections,
            faces,
          },

          fileContext ||
            null,

          history,

          controller.signal,

          (chunk) => {
            setMessages(
              (current) =>
                current.map(
                  (message) =>
                    message.id ===
                    thinkingId
                      ? {
                          ...message,

                          content:
                            message.content +
                            chunk,
                        }
                      : message
                )
            );
          },

          memoryHistory
        );

      if (
        response.response &&
        response.response.trim()
      ) {
        setMessages(
          (current) =>
            current.map(
              (message) =>
                message.id ===
                thinkingId
                  ? {
                      ...message,

                      content:
                        response.response ||
                        "No response received.",
                    }
                  : message
            )
        );

        if (speakResponse) {
          setVoiceStatus("🔊 Speaking response...");
          await handleSpeak(
            thinkingId,
            response.response,
            true
          );
        }
      } else if (speakResponse) {
        setVoiceStatus("⚠️ The assistant returned no spoken response");
        if (conversationModeRef.current) {
          resumeConversationListening();
        }
      }
    } catch (error) {
      if (
        error instanceof DOMException &&
        error.name ===
          "AbortError"
      ) {
        setMessages(
          (current) =>
            current.map(
              (message) =>
                message.id ===
                thinkingId
                  ? {
                      ...message,

                      content:
                        message.content.trim()
                          ? message.content +
                            "\n\n⏹ Generation stopped."
                          : "⏹ Generation stopped.",
                    }
                  : message
            )
        );

        if (speakResponse && conversationModeRef.current) {
          resumeConversationListening();
        }

        return;
      }

      console.error(
        "Assistant error:",
        error
      );

      const errorText =
        error instanceof Error
          ? error.message
          : "Unknown error";

      setMessages(
        (current) =>
          current.map(
            (message) =>
              message.id ===
              thinkingId
                ? {
                    ...message,

                    content:
                      `❌ ${errorText}`,
                  }
                : message
          )
      );

      if (speakResponse) {
        setVoiceStatus("❌ Voice request failed");
        if (conversationModeRef.current) {
          window.setTimeout(() => {
            if (conversationModeRef.current) {
              resumeConversationListening();
            }
          }, 1000);
        }
      }

      void checkBackend();
    } finally {
      setIsThinking(
        false
      );

      requestControllerRef.current =
        null;
    }
  }

  // Refresh the handler reference so later voice turns use current chat state.
  handleSendRef.current = handleSend;


  /* =======================================================
     STOP GENERATION
  ======================================================= */

  function stopGeneration() {
    requestControllerRef.current?.abort();
  }


  /* =======================================================
     VOICE RECORDING
  ======================================================= */

  /* =======================================================
     PHASE 22.3 — LOCAL WAKE-WORD MODE
  ======================================================= */

  function releaseWakeWordResources() {
    const processor = wakeProcessorRef.current;
    wakeProcessorRef.current = null;

    if (processor) {
      processor.onaudioprocess = null;
      try {
        processor.disconnect();
      } catch {
        /* Already disconnected. */
      }
    }

    const source = wakeSourceRef.current;
    wakeSourceRef.current = null;

    if (source) {
      try {
        source.disconnect();
      } catch {
        /* Already disconnected. */
      }
    }

    const stream = wakeMediaStreamRef.current;
    wakeMediaStreamRef.current = null;
    stream?.getTracks().forEach((track) => track.stop());

    const context = wakeAudioContextRef.current;
    wakeAudioContextRef.current = null;

    if (context && context.state !== "closed") {
      void context.close().catch((error) => {
        console.debug("Wake-word AudioContext cleanup:", error);
      });
    }

    const socket = wakeSocketRef.current;
    wakeSocketRef.current = null;

    if (socket && socket.readyState < WebSocket.CLOSING) {
      try {
        socket.close(1000, "Wake-word mode stopped");
      } catch {
        /* Socket may already be closing. */
      }
    }
  }


  function stopWakeWordMode(showStatus = true) {
    wakeWordActiveRef.current = false;
    setIsWakeWordActive(false);
    releaseWakeWordResources();

    if (showStatus) {
      setVoiceStatus("Wake-word standby stopped");
      window.setTimeout(() => setVoiceStatus(""), 1800);
    }
  }


  function toggleWakeWordMode() {
    if (wakeWordActiveRef.current) {
      stopWakeWordMode();
      return;
    }

    void startWakeWordMode();
  }


  async function startWakeWordMode() {
    if (
      backendConnection !== "online" ||
      isRecording ||
      isThinking ||
      isTranscribing ||
      isConversationMode ||
      fileLoading ||
      speakingMessageId !== null
    ) {
      setVoiceStatus(
        backendConnection !== "online"
          ? "❌ Start the backend before enabling wake-word mode"
          : "Finish the current action before enabling wake-word mode"
      );
      window.setTimeout(() => setVoiceStatus(""), 2400);
      return;
    }

    wakeWordActiveRef.current = true;
    setIsWakeWordActive(true);
    setVoiceStatus("👂 Connecting to the local wake-word detector...");

    let stream: MediaStream | null = null;
    let context: AudioContext | null = null;
    let socket: WebSocket | null = null;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      if (!wakeWordActiveRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      wakeMediaStreamRef.current = stream;

      socket = new WebSocket("ws://127.0.0.1:8000/wake-word");
      wakeSocketRef.current = socket;

      let wakeReady = false;

      socket.addEventListener("message", (event) => {
        if (!wakeWordActiveRef.current) {
          return;
        }

        let payload: {
          type?: string;
          phrase?: string;
          message?: string;
          score?: number;
        };

        try {
          payload = JSON.parse(String(event.data));
        } catch {
          return;
        }

        if (payload.type === "ready") {
          wakeReady = true;
          setVoiceStatus("👂 Wake-word standby ready · Say “Hey Jarvis”");
          return;
        }

        if (payload.type === "error") {
          const message = payload.message || "Local wake-word detector failed";
          stopWakeWordMode(false);
          setVoiceStatus("❌ " + message);
          return;
        }

        if (payload.type === "wake_word_detected") {
          stopWakeWordMode(false);

          // After the wake phrase, start the existing hands-free voice loop.
          // Speak the request after the Listening status appears.
          conversationModeRef.current = true;
          setIsConversationMode(true);
          setVoiceStatus("✅ Hey Jarvis detected · Listening for your request...");

          window.setTimeout(() => {
            if (conversationModeRef.current) {
              void startRecording();
            }
          }, 450);
        }
      });

      socket.addEventListener("close", () => {
        if (wakeWordActiveRef.current) {
          stopWakeWordMode(false);
          setVoiceStatus("⚠️ Wake-word connection closed");
        }
      });

      await new Promise<void>((resolve, reject) => {
        const timeoutId = window.setTimeout(() => {
          reject(new Error("Wake-word server connection timed out"));
        }, 10000);

        socket?.addEventListener(
          "open",
          () => {
            window.clearTimeout(timeoutId);
            resolve();
          },
          { once: true }
        );

        socket?.addEventListener(
          "error",
          () => {
            window.clearTimeout(timeoutId);
            reject(new Error("Could not connect to /wake-word"));
          },
          { once: true }
        );

        socket?.addEventListener(
          "close",
          () => {
            window.clearTimeout(timeoutId);
            reject(new Error("Wake-word connection closed before opening"));
          },
          { once: true }
        );
      });

      if (!wakeWordActiveRef.current) {
        return;
      }

      context = new AudioContext();
      await context.resume();

      if (!wakeWordActiveRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        if (context.state !== "closed") {
          await context.close();
        }
        return;
      }

      wakeAudioContextRef.current = context;

      const source = context.createMediaStreamSource(stream);
      wakeSourceRef.current = source;

      // ScriptProcessor is used for broad browser support. Resample the
      // browser's microphone rate to 16 kHz mono before sending short PCM frames.
      const processor = context.createScriptProcessor(4096, 1, 1);
      wakeProcessorRef.current = processor;

      let resampleCarry = new Float32Array(0);
      let resamplePosition = 0;
      let pendingPcm = new Int16Array(0);

      processor.onaudioprocess = (audioEvent) => {
        audioEvent.outputBuffer.getChannelData(0).fill(0);

        if (
          !wakeWordActiveRef.current ||
          !wakeReady ||
          socket?.readyState !== WebSocket.OPEN
        ) {
          return;
        }

        const input = audioEvent.inputBuffer.getChannelData(0);
        const combined = new Float32Array(resampleCarry.length + input.length);
        combined.set(resampleCarry);
        combined.set(input, resampleCarry.length);

        const step = context!.sampleRate / 16000;
        const resampled: number[] = [];

        while (resamplePosition + 1 < combined.length) {
          const left = Math.floor(resamplePosition);
          const fraction = resamplePosition - left;

          resampled.push(
            combined[left] * (1 - fraction) +
            combined[left + 1] * fraction
          );

          resamplePosition += step;
        }

        const consumedSamples = Math.floor(resamplePosition);
        resampleCarry = combined.slice(consumedSamples);
        resamplePosition -= consumedSamples;

        if (resampled.length === 0) {
          return;
        }

        const converted = new Int16Array(resampled.length);

        for (let i = 0; i < resampled.length; i += 1) {
          const sample = Math.max(-1, Math.min(1, resampled[i]));
          converted[i] = sample < 0
            ? Math.round(sample * 32768)
            : Math.round(sample * 32767);
        }

        const joined = new Int16Array(pendingPcm.length + converted.length);
        joined.set(pendingPcm);
        joined.set(converted, pendingPcm.length);

        let offset = 0;

        while (joined.length - offset >= 1280) {
          const frame = joined.slice(offset, offset + 1280);

          if (
            wakeWordActiveRef.current &&
            socket?.readyState === WebSocket.OPEN
          ) {
            socket.send(frame.buffer);
          }

          offset += 1280;
        }

        pendingPcm = joined.slice(offset);
      };

      source.connect(processor);
      processor.connect(context.destination);
    } catch (error) {
      console.error("Wake-word setup error:", error);

      stream?.getTracks().forEach((track) => track.stop());
      if (context && context.state !== "closed") {
        void context.close().catch((closeError) => {
          console.debug("Wake-word setup cleanup:", closeError);
        });
      }

      if (wakeWordActiveRef.current) {
        stopWakeWordMode(false);
        setVoiceStatus(
          error instanceof Error
            ? "❌ " + error.message
            : "❌ Could not start local wake-word mode"
        );
      } else {
        releaseWakeWordResources();
      }
    }
  }


  function toggleConversationMode() {
    if (conversationModeRef.current) {
      conversationModeRef.current = false;
      setIsConversationMode(false);

      if (mediaRecorderRef.current?.state === "recording") {
        setVoiceStatus("🧠 Finishing the current recording...");
        stopRecording();
      } else {
        setVoiceStatus("Conversation mode stopped");
        window.setTimeout(() => setVoiceStatus(""), 1800);
      }

      return;
    }

    if (backendConnection !== "online") {
      setVoiceStatus("❌ Start the backend before using conversation mode");
      window.setTimeout(() => setVoiceStatus(""), 2500);
      return;
    }

    if (
      isRecording ||
      isThinking ||
      isTranscribing ||
      isWakeWordActive ||
      fileLoading ||
      speakingMessageId !== null
    ) {
      setVoiceStatus("Finish the current action before starting conversation mode");
      window.setTimeout(() => setVoiceStatus(""), 2500);
      return;
    }

    conversationModeRef.current = true;
    setIsConversationMode(true);
    setVoiceStatus("🎤 Starting hands-free conversation...");
    void startRecording();
  }


  /* =======================================================
     VOICE RECORDING + LOCAL VOICE ACTIVITY DETECTION
  ======================================================= */

  async function startRecording(automaticRestart = false) {
    if (
      mediaRecorderRef.current ||
      (
        !automaticRestart &&
        (
          isRecording ||
          isThinking ||
          isTranscribing ||
          fileLoading
        )
      )
    ) {
      return;
    }

    let stream: MediaStream | null = null;
    let audioContext: AudioContext | null = null;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      const recordingStream = stream;
      let analyser: AnalyserNode | null = null;
      let sourceNode: MediaStreamAudioSourceNode | null = null;
      let vadIntervalId: number | null = null;
      let vadBuffer = new Float32Array(0);
      let voiceCandidateSince = 0;
      let lastSpeechAt = 0;
      let speechDetected = false;

      let mimeType = "audio/webm";

      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = "audio/webm;codecs=opus";
      }

      const recorder = new MediaRecorder(recordingStream, {
        mimeType,
      });

      if (conversationModeRef.current) {
        audioContext = new AudioContext();
        await audioContext.resume();

        analyser = audioContext.createAnalyser();
        analyser.fftSize = 1024;

        sourceNode = audioContext.createMediaStreamSource(recordingStream);
        sourceNode.connect(analyser);

        vadBuffer = new Float32Array(analyser.fftSize);
      }

      audioChunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        if (vadIntervalId !== null) {
          window.clearInterval(vadIntervalId);
          vadIntervalId = null;
        }

        recordingStream.getTracks().forEach((track) => track.stop());
        sourceNode?.disconnect();

        if (audioContext && audioContext.state !== "closed") {
          try {
            await audioContext.close();
          } catch (error) {
            console.debug("AudioContext cleanup:", error);
          }
        }

        setIsRecording(false);

        const blob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });

        setIsTranscribing(true);
        setVoiceStatus("🧠 Converting speech to text...");

        try {
          const text = await transcribeAudio(blob);

          if (text.trim()) {
            const transcript = text.trim();

            setMessageInput(transcript);
            setVoiceStatus("🧠 Processing your request...");

            // Keep the existing router, assistant response, and Piper TTS flow.
            await handleSendRef.current?.(transcript, true);
          } else {
            setVoiceStatus("⚠️ No speech detected");
            if (conversationModeRef.current) {
              resumeConversationListening();
            } else {
              window.setTimeout(() => setVoiceStatus(""), 1800);
            }
          }
        } catch (error) {
          console.error("Transcription error:", error);
          setVoiceStatus("❌ Transcription failed");
          if (conversationModeRef.current) {
            resumeConversationListening();
          } else {
            window.setTimeout(() => setVoiceStatus(""), 2200);
          }
        } finally {
          setIsTranscribing(false);
        }
      };

      recorder.start();

      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      setVoiceStatus(
        conversationModeRef.current
          ? "🎤 Listening... speak naturally, then pause to send"
          : "🎤 Listening..."
      );

      // Detect a complete utterance locally: sustained speech followed by
      // 1.2 seconds of silence. Audio is only uploaded when recording stops.
      if (analyser) {
        vadIntervalId = window.setInterval(() => {
          if (
            !conversationModeRef.current ||
            recorder.state !== "recording" ||
            !analyser
          ) {
            return;
          }

          analyser.getFloatTimeDomainData(vadBuffer);

          let energy = 0;

          for (let i = 0; i < vadBuffer.length; i += 1) {
            energy += vadBuffer[i] * vadBuffer[i];
          }

          const rms = Math.sqrt(energy / vadBuffer.length);
          const now = Date.now();

          if (rms > 0.025) {
            if (voiceCandidateSince === 0) {
              voiceCandidateSince = now;
            }

            if (now - voiceCandidateSince >= 180) {
              speechDetected = true;
              lastSpeechAt = now;
            }
          } else {
            voiceCandidateSince = 0;
          }

          if (speechDetected && now - lastSpeechAt >= 1200) {
            stopRecording();
          }
        }, 50);
      }
    } catch (error) {
      console.error("Microphone error:", error);

      stream?.getTracks().forEach((track) => track.stop());

      if (audioContext && audioContext.state !== "closed") {
        void audioContext.close().catch((closeError) => {
          console.debug("AudioContext cleanup:", closeError);
        });
      }

      if (conversationModeRef.current) {
        conversationModeRef.current = false;
        setIsConversationMode(false);
      }

      setVoiceStatus("❌ Microphone access failed");
      alert("Microphone access failed. Please allow microphone permission.");
      window.setTimeout(() => setVoiceStatus(""), 2500);
    }
  }


  /* =======================================================
     STOP RECORDING
  ======================================================= */

  function stopRecording() {
    const recorder =
      mediaRecorderRef.current;

    if (!recorder) {
      return;
    }

    if (
      recorder.state !==
      "inactive"
    ) {
      recorder.stop();
    }

    mediaRecorderRef.current =
      null;

    setIsRecording(
      false
    );

    setVoiceStatus(
      "🧠 Converting speech to text..."
    );
  }


  /* =======================================================
     MANUAL VOICE BUTTON
  ======================================================= */

  function handleVoiceClick() {
    if (isRecording) {
      stopRecording();
    } else {
      void startRecording();
    }
  }


  function resumeConversationListening() {
    if (!conversationModeRef.current) {
      return;
    }

    window.setTimeout(() => {
      if (
        !conversationModeRef.current ||
        mediaRecorderRef.current
      ) {
        return;
      }

      setVoiceStatus("🎤 Listening for your next request...");
      void startRecording(true);
    }, 500);
  }


  /* =======================================================
     TEXT TO SPEECH
  ======================================================= */

  async function handleSpeak(
    messageId: number,
    text: string,
    voiceTurn = false
  ) {
    if (!text.trim()) {
      return;
    }

    if (
      speakingMessageId ===
      messageId
    ) {
      currentAudioRef.current?.pause();

      currentAudioRef.current =
        null;

      setSpeakingMessageId(
        null
      );

      return;
    }

    currentAudioRef.current?.pause();

    currentAudioRef.current =
      null;

    setSpeakingMessageId(
      messageId
    );

    try {
      const audio =
        await speakText(
          text
        );

      currentAudioRef.current =
        audio;

      audio.onended =
        () => {
          setSpeakingMessageId(
            null
          );

          currentAudioRef.current =
            null;

          if (voiceTurn) {
            if (conversationModeRef.current) {
              setVoiceStatus("✅ Reply finished · Listening again...");
              resumeConversationListening();
            } else {
              setVoiceStatus("✅ Voice reply finished");
              window.setTimeout(() => setVoiceStatus(""), 2200);
            }
          }
        };

      audio.onerror =
        () => {
          setSpeakingMessageId(
            null
          );

          currentAudioRef.current =
            null;

          if (voiceTurn) {
            if (conversationModeRef.current) {
              setVoiceStatus("⚠️ Audio playback failed · Listening again...");
              resumeConversationListening();
            } else {
              setVoiceStatus("❌ Audio playback failed");
              window.setTimeout(() => setVoiceStatus(""), 3000);
            }
          }
        };

      await audio.play();
    } catch (error) {
      console.error(
        "TTS error:",
        error
      );

      setSpeakingMessageId(
        null
      );

      currentAudioRef.current =
        null;

      if (voiceTurn) {
        if (conversationModeRef.current) {
          setVoiceStatus("⚠️ Could not play response · Listening again...");
          window.setTimeout(() => {
            if (conversationModeRef.current) {
              resumeConversationListening();
            }
          }, 1000);
        } else {
          setVoiceStatus("❌ Could not play the voice response");
          window.setTimeout(() => setVoiceStatus(""), 3000);
        }
      }
    }
  }


  /* =======================================================
     FILE PICKER
  ======================================================= */

  function openFilePicker() {
    if (
      fileLoading ||
      isThinking
    ) {
      return;
    }

    fileInputRef.current?.click();
  }


  /* =======================================================
     FILE SELECTED
  ======================================================= */

  async function handleFileSelected(
    event: ChangeEvent<HTMLInputElement>
  ) {
    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    const extension =
      "." +
      (
        file.name
          .split(".")
          .pop() ||
        ""
      ).toLowerCase();

    const allowed = [
      ".txt",
      ".pdf",
      ".docx",
    ];

    if (
      !allowed.includes(
        extension
      )
    ) {
      alert(
        "Supported files: TXT, PDF and DOCX"
      );

      event.target.value =
        "";

      return;
    }

    setFileLoading(
      true
    );

    setFileName(
      file.name
    );

    setFileContext(
      ""
    );

    setFileCharacters(
      0
    );

    try {
      const data =
        await readFile(
          file
        );

      if (
        !data.text?.trim()
      ) {
        throw new Error(
          "No readable text was found in this file."
        );
      }

      setFileContext(
        data.text
      );

      setFileCharacters(
        data.characters ||
        data.text.length
      );

      setMessages(
        (current) => [
          ...current,
          {
            id:
              Date.now(),

            role:
              "assistant",

            content:
              `📎 ${
                data.filename ||
                file.name
              } is ready.\n\nAsk me anything about this file.`,
          },
        ]
      );
    } catch (error) {
      console.error(
        "File read error:",
        error
      );

      const errorText =
        error instanceof Error
          ? error.message
          : "Unable to read file.";

      setFileName("");

      setFileContext("");

      setFileCharacters(
        0
      );

      setMessages(
        (current) => [
          ...current,
          {
            id:
              Date.now(),

            role:
              "assistant",

            content:
              `❌ File error: ${errorText}`,
          },
        ]
      );
    } finally {
      setFileLoading(
        false
      );

      event.target.value =
        "";
    }
  }


  /* =======================================================
     REMOVE FILE
  ======================================================= */

  function removeFile() {
    setFileName("");

    setFileContext("");

    setFileCharacters(
      0
    );

    if (
      fileInputRef.current
    ) {
      fileInputRef.current.value =
        "";
    }
  }


  /* =======================================================
     ENROLLMENT
  ======================================================= */

  function openEnrollPanel() {
    if (
      !cameraRunning
    ) {
      alert(
        "Start the camera before enrolling a face."
      );

      return;
    }

    setEnrollName("");

    setEnrollStatus("");

    setEnrollPanelOpen(
      true
    );
  }


  function closeEnrollPanel() {
    if (
      isEnrolling
    ) {
      return;
    }

    setEnrollPanelOpen(
      false
    );
  }


  async function handleEnroll() {
    const name =
      enrollName.trim();

    if (!name) {
      setEnrollStatus(
        "❌ Enter your name."
      );

      return;
    }

    const video =
      document.querySelector(
        ".camera-video"
      ) as HTMLVideoElement |
      null;

    if (
      !video ||
      video.readyState <
        2
    ) {
      setEnrollStatus(
        "❌ Camera is not ready."
      );

      return;
    }

    setIsEnrolling(
      true
    );

    setEnrollStatus(
      "🧠 Capturing face..."
    );

    try {
      const canvas =
        document.createElement(
          "canvas"
        );

      canvas.width =
        video.videoWidth;

      canvas.height =
        video.videoHeight;

      const context =
        canvas.getContext(
          "2d"
        );

      if (!context) {
        throw new Error(
          "Could not capture camera frame."
        );
      }

      context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
      );

      const blob =
        await new Promise<
          Blob | null
        >(
          (resolve) =>
            canvas.toBlob(
              resolve,
              "image/jpeg",
              0.85
            )
        );

      if (!blob) {
        throw new Error(
          "Could not create image."
        );
      }

      setEnrollStatus(
        "🧠 Processing face..."
      );

      const result =
        await enrollFace(
          blob,
          name
        );

      setEnrollStatus(
        `✅ ${
          result.message ||
          `${name} was enrolled successfully.`
        }`
      );

      setCameraStatus(
        `✅ ${name} enrolled successfully`
      );

      window.setTimeout(
        () => {
          setEnrollPanelOpen(
            false
          );

          setEnrollStatus(
            ""
          );
        },
        1500
      );
    } catch (error) {
      console.error(
        "Enrollment error:",
        error
      );

      setEnrollStatus(
        `❌ ${
          error instanceof Error
            ? error.message
            : "Enrollment failed."
        }`
      );
    } finally {
      setIsEnrolling(
        false
      );
    }
  }


  function handleEnrollKeyDown(
    event: KeyboardEvent<HTMLInputElement>
  ) {
    if (
      event.key ===
        "Enter" &&
      !isEnrolling
    ) {
      event.preventDefault();

      void handleEnroll();
    }
  }


  /* =======================================================
     CLEANUP
  ======================================================= */

  useEffect(() => {
    return () => {
      requestControllerRef.current?.abort();
      conversationModeRef.current = false;
      wakeWordActiveRef.current = false;

      if (wakeProcessorRef.current) {
        wakeProcessorRef.current.onaudioprocess = null;
        wakeProcessorRef.current.disconnect();
      }
      wakeSourceRef.current?.disconnect();
      wakeMediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      if (
        wakeAudioContextRef.current &&
        wakeAudioContextRef.current.state !== "closed"
      ) {
        void wakeAudioContextRef.current.close();
      }
      if (
        wakeSocketRef.current &&
        wakeSocketRef.current.readyState < WebSocket.CLOSING
      ) {
        wakeSocketRef.current.close(1000, "App unmounted");
      }

      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !==
          "inactive"
      ) {
        mediaRecorderRef.current.stop();
      }

      currentAudioRef.current?.pause();
    };
  }, []);


  /* =======================================================
     HISTORY
  ======================================================= */

  const sortedConversations =
    conversations
      .slice()
      .sort(
        (a, b) =>
          b.updatedAt -
          a.updatedAt
      );


  const currentConversation =
    conversations.find(
      (conversation) =>
        conversation.id ===
        currentConversationId
    );


  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <div className="app">

      <Header
        backendConnection={
          backendConnection
        }

        backendStatus={
          backendStatus
        }

        onRetry={() =>
          void checkBackend()
        }

        onClear={
          clearCurrentChat
        }
      />


      {/* =================================================
         HISTORY TOOLBAR
      ================================================= */}

      <div className="history-toolbar">

        <button
          className="history-button"
          type="button"
          onClick={() =>
            setHistoryPanelOpen(
              (open) =>
                !open
            )
          }
        >
          {
            historyPanelOpen
              ? "✕ Close"
              : "☰ History"
          }
        </button>


        <button
          className="new-chat-button"
          type="button"
          onClick={
            createNewChat
          }
        >
          ＋ New Chat
        </button>


        <div className="current-chat-title">
          {
            currentConversation?.title ||
            "New Conversation"
          }
        </div>

      </div>


      {/* =================================================
         HISTORY PANEL
      ================================================= */}

      {historyPanelOpen && (
        <aside
          className="history-panel"
        >

          <div
            className="history-panel-header"
          >

            <span>
              Conversations
            </span>

            <span className="history-count">
              {
                conversations.length
              }
            </span>

          </div>


          <div className="history-list">

            {sortedConversations.map(
              (
                conversation
              ) => (

                <div
                  className={
                    `history-item ${
                      conversation.id ===
                      currentConversationId
                        ? "active"
                        : ""
                    }`
                  }

                  key={
                    conversation.id
                  }
                >

                  <button
                    className="history-open"
                    type="button"
                    onClick={() =>
                      openConversation(
                        conversation.id
                      )
                    }
                  >

                    <span className="history-item-title">
                      {
                        conversation.title
                      }
                    </span>


                    <span className="history-item-date">
                      {
                        new Date(
                          conversation.updatedAt
                        ).toLocaleString()
                      }
                    </span>

                  </button>


                  <button
                    className="history-delete"
                    type="button"
                    title="Delete conversation"
                    onClick={() =>
                      deleteConversation(
                        conversation.id
                      )
                    }
                  >
                    🗑
                  </button>

                </div>
              )
            )}

          </div>

        </aside>
      )}


      {/* =================================================
         WORKSPACE
      ================================================= */}

      <main
        className={
          `workspace workspace-${cameraViewMode}`
        }
      >

        <ChatPanel
          messages={
            messages
          }

          speakingMessageId={
            speakingMessageId
          }

          onSpeak={
            handleSpeak
          }
        />


        <CameraPanel
          cameraRunning={
            cameraRunning
          }

          setCameraRunning={
            setCameraRunning
          }

          cameraStatus={
            cameraStatus
          }

          setCameraStatus={
            setCameraStatus
          }

          detections={
            detections
          }

          setDetections={
            setDetections
          }

          faces={
            faces
          }

          setFaces={
            setFaces
          }

          detectionFPS={
            detectionFPS
          }

          setDetectionFPS={
            setDetectionFPS
          }

          cameraViewMode={
            cameraViewMode
          }

          setCameraViewMode={
            setCameraViewMode
          }

          onEnroll={
            openEnrollPanel
          }
        />

      </main>


      {/* =================================================
         FILE BAR
      ================================================= */}

      <FileBar
        fileName={
          fileName
        }

        characters={
          fileCharacters
        }

        loading={
          fileLoading
        }

        onRemove={
          removeFile
        }
      />


      {/* =================================================
         VOICE STATUS
      ================================================= */}

      <VoiceStatus
        text={
          voiceStatus
        }
      />


      {/* =================================================
         ENROLL MODAL
      ================================================= */}

      {enrollPanelOpen && (
        <EnrollModal

          open={
            enrollPanelOpen
          }

          name={
            enrollName
          }

          status={
            enrollStatus
          }

          loading={
            isEnrolling
          }

          onNameChange={
            setEnrollName
          }

          onKeyDown={
            handleEnrollKeyDown
          }

          onEnroll={() =>
            void handleEnroll()
          }

          onClose={
            closeEnrollPanel
          }
        />
      )}


      {/* =================================================
         COMPOSER
      ================================================= */}

      <footer
        className="composer"
      >

        <div
          className="composer-inner"
        >

          {/* FILE BUTTON */}

          <button
            className="composer-icon"
            type="button"
            title="Upload TXT, PDF or DOCX"
            onClick={
              openFilePicker
            }

            disabled={
              fileLoading ||
              isThinking
            }
          >
            {
              fileLoading
                ? "⏳"
                : "📎"
            }
          </button>


          <input
            ref={
              fileInputRef
            }

            type="file"

            accept=".txt,.pdf,.docx"

            hidden

            onChange={
              handleFileSelected
            }
          />


          {/* HANDS-FREE CONVERSATION MODE */}

          <button
            className={
              `composer-icon conversation-control ${isConversationMode ? "active" : ""}`
            }
            type="button"
            title={
              isConversationMode
                ? "Stop hands-free conversation"
                : "Start hands-free conversation"
            }
            aria-label={
              isConversationMode
                ? "Stop hands-free conversation"
                : "Start hands-free conversation"
            }
            aria-pressed={isConversationMode}
            onClick={toggleConversationMode}
            disabled={
              !isConversationMode &&
              (
                isRecording ||
                isThinking ||
                isTranscribing ||
                isWakeWordActive ||
                fileLoading ||
                speakingMessageId !== null
              )
            }
          >
            🔁
          </button>


          {/* LOCAL WAKE-WORD MODE */}

          <button
            className={
              `composer-icon wake-word-control ${isWakeWordActive ? "active" : ""}`
            }
            type="button"
            title={
              isWakeWordActive
                ? "Stop wake-word standby"
                : "Enable wake-word standby (Hey Jarvis)"
            }
            aria-label={
              isWakeWordActive
                ? "Stop wake-word standby"
                : "Enable wake-word standby"
            }
            aria-pressed={isWakeWordActive}
            onClick={toggleWakeWordMode}
            disabled={
              !isWakeWordActive &&
              (
                isRecording ||
                isThinking ||
                isTranscribing ||
                isConversationMode ||
                fileLoading ||
                speakingMessageId !== null
              )
            }
          >
            👂
          </button>


          {/* MICROPHONE */}

          <button
            className={
              `composer-icon mic-control ${
                isRecording
                  ? "recording"
                  : ""
              }`
            }

            type="button"

            title={
              isRecording
                ? "Stop recording"
                : "Voice input"
            }

            onClick={
              handleVoiceClick
            }

            disabled={
              isThinking ||
              isTranscribing ||
              fileLoading ||
              isConversationMode
            }
          >

            {
              isRecording
                ? "⏹"
                : "🎤"
            }

          </button>


          {/* TEXT INPUT */}

          <input
            className="composer-input"

            type="text"

            value={
              messageInput
            }

            placeholder={
              fileName
                ? "Ask about your file..."
                : "Ask me anything..."
            }

            onChange={
              (event) =>
                setMessageInput(
                  event.target.value
                )
            }

            onKeyDown={
              (event) => {
                if (
                  event.key ===
                    "Enter" &&
                  !event.shiftKey &&
                  !isThinking
                ) {
                  event.preventDefault();

                  void handleSend();
                }
              }
            }

            disabled={
              isTranscribing ||
              fileLoading
            }
          />


          {/* =================================================
             STOP / SEND
          ================================================= */}

          {isThinking ? (
            <button
              className="composer-stop"
              type="button"
              title="Stop generation"
              onClick={
                stopGeneration
              }
            >
              ■
            </button>
          ) : (
            <button
              className="composer-send"
              type="button"
              title="Send"
              onClick={() =>
                void handleSend()
              }

              disabled={
                isTranscribing ||
                fileLoading ||
                !messageInput.trim()
              }
            >
              ➤
            </button>
          )}

        </div>


        {/* COMPOSER STATUS */}

        <div
          className="composer-hint"
        >
          {
            isThinking

              ? "Generating response · Click ■ to stop"

              : isTranscribing

                ? "🧠 Converting speech to text..."

                : isWakeWordActive

                ? "👂 Wake-word standby · Say “Hey Jarvis”"

                : isRecording

                  ? isConversationMode
                    ? "🎤 Hands-free listening · Pause briefly when you finish speaking"
                    : "🎤 Listening..."

                  : isConversationMode
                    ? "Hands-free mode on · Speak naturally; a pause sends your request"

                    : backendConnection ===
                      "offline"

                  ? "Backend offline · Start FastAPI to continue"

                  : fileName

                    ? "File attached · Questions are answered using the uploaded content"

                    : "Local AI · Streaming · Persistent History · Vision · Voice · Memory"
          }
        </div>

      </footer>

    </div>
  );
}


export default App;
